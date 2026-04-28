import torch
import abc
from typing import Dict,Tuple

class TopKModule(torch.nn.Module):

    @abc.abstractmethod
    def forward(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        aux_payloads: Dict[str, torch.Tensor],
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, X, ...). Implementation-specific.
            k: int. top k to return.
            sorted: bool.

        Returns:
            Tuple of (top_k_scores, top_k_indices), both of shape (B, K,)
        """
        pass
