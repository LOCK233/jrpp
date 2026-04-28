from typing import Dict, Tuple

import torch

from models.mol.candidate_index import TopKModule
from models.mol.mol import MoLSimilarity


class MoLTopKModule(TopKModule):

    def __init__(
        self,
        mol_module: MoLSimilarity, 
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        flatten_item_ids_and_embeddings: bool,
        keep_component_level_item_embeddings: bool,
    ) -> None:
        super().__init__()

        self._mol_module: MoLSimilarity = mol_module
        self._item_embeddings: torch.Tensor = item_embeddings if not flatten_item_ids_and_embeddings else item_embeddings.squeeze(0)
        
        if keep_component_level_item_embeddings:
            # (X, D) -> (X, K_I, D) -> (K_I, X, D)
            self._mol_item_embeddings: torch.Tensor = (
                mol_module.get_item_component_embeddings(
                    self._item_embeddings.squeeze(0) if not flatten_item_ids_and_embeddings else self._item_embeddings
                ).permute(1, 0, 2)
            )

        self._item_ids: torch.Tensor = item_ids if not flatten_item_ids_and_embeddings else item_ids.squeeze(0)
        
    @property
    def mol_module(self) -> MoLSimilarity:
        return self._mol_module

class MoLAvgTopK(MoLTopKModule):
    def __init__(
        self,
        mol_module: MoLSimilarity, 
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        avg_top_k: int,
    ) -> None:
        super().__init__(
            mol_module=mol_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            flatten_item_ids_and_embeddings=True,
            keep_component_level_item_embeddings=True,
        )
        P_X, _, D_prime = self._mol_item_embeddings.size()
        self._avg_mol_item_embeddings_t = (self._mol_item_embeddings.sum(0) / P_X).transpose(0, 1)  # (P_X, X, D') -> (X, D') -> (D', X)
        self._avg_top_k: int = avg_top_k

    def forward(
        self, 
        query_embeddings: torch.Tensor,
        k: int,
        sorted: bool = True,
        exclude_item_ids: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D = query_embeddings.size()
        mol_query_embeddings, _ = self.mol_module.get_query_component_embeddings(query_embeddings, **kwargs)  # (B, P_Q, D_prime)
        _, P_Q, D_prime = mol_query_embeddings.size()
        _, N, _ = self._mol_item_embeddings.size()
        candidate_k = min(self._avg_top_k, N)
        output_k = min(k, candidate_k)

        avg_sim_values = torch.mm(mol_query_embeddings.sum(1) / P_Q, self._avg_mol_item_embeddings_t)  # (B, D_prime) * (D_prime, X)
        excluded = None
        item_ids = self._item_ids.reshape(1, -1).to(avg_sim_values.device)
        if exclude_item_ids is not None:
            excluded = exclude_item_ids.reshape(-1, 1).to(device=avg_sim_values.device, dtype=self._item_ids.dtype)
            excluded_mask = item_ids.eq(excluded)
            avg_sim_values = avg_sim_values.masked_fill(excluded_mask, torch.finfo(avg_sim_values.dtype).min)

        _, avg_sim_top_k_indices = torch.topk(avg_sim_values, k=candidate_k, dim=1)

        # queries averaged results
        avg_filtered_item_embeddings = self._item_embeddings[avg_sim_top_k_indices].reshape(B, candidate_k, D)
        candidate_scores, _ = self.mol_module(query_embeddings, avg_filtered_item_embeddings, item_sideinfo=None, item_ids=None, **kwargs)
        if excluded is not None:
            avg_filtered_item_ids = self._item_ids[avg_sim_top_k_indices].to(candidate_scores.device)
            candidate_scores = candidate_scores.masked_fill(
                avg_filtered_item_ids.eq(excluded.to(candidate_scores.device)),
                torch.finfo(candidate_scores.dtype).min,
            )
        top_k_logits, top_k_indices = torch.topk(input=candidate_scores, k=output_k, dim=1, largest=True, sorted=sorted)
        top_k_item_indices = torch.gather(avg_sim_top_k_indices, dim=1, index=top_k_indices)
        return top_k_logits, top_k_item_indices

    def topk_ids(
        self, 
        query_embeddings: torch.Tensor,
        sorted: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D = query_embeddings.size()
        mol_query_embeddings, _ = self.mol_module.get_query_component_embeddings(query_embeddings, **kwargs)  # (B, P_Q, D_prime)
        _, P_Q, D_prime = mol_query_embeddings.size()
        _, N, _ = self._mol_item_embeddings.size()
        candidate_k = min(self._avg_top_k, N)

        avg_sim_values = torch.mm(mol_query_embeddings.sum(1) / P_Q, self._avg_mol_item_embeddings_t)  # (B, D_prime) * (D_prime, X)
        _, avg_sim_top_k_indices = torch.topk(avg_sim_values, k=candidate_k, dim=1)
        return self._item_ids[avg_sim_top_k_indices]
