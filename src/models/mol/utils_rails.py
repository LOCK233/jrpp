import re

import torch

from models.mol.candidate_index import TopKModule
from models.mol.mol_top_k import MoLAvgTopK


def get_top_k_module(
    top_k_method: str,
    model: torch.nn.Module,
    item_embeddings: torch.Tensor,
    item_ids: torch.Tensor,
) -> TopKModule:
    match = re.fullmatch(r"MoLAvgTopK(\d+)", top_k_method)
    if not match:
        raise ValueError(f"Invalid top-k method {top_k_method!r}. Expected e.g. MoLAvgTopK100.")

    return MoLAvgTopK(
        mol_module=model.mol_similarity,
        item_embeddings=item_embeddings,
        item_ids=item_ids,
        avg_top_k=int(match.group(1)),
    )
