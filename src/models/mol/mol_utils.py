from typing import Optional, List, Tuple

import gin

import torch

from models.mol.mol import MoLSimilarity, GeGLU, SwiGLU, SoftmaxDropoutCombiner, IdentityMLPProjectionFn
from models.mol.mol_query_embeddings import RecoMoLQueryEmbeddingsFn


@gin.configurable
def create_mol_interaction_module(
    query_embedding_dim: int,
    item_embedding_dim: int,
    dot_product_dimension: int,
    query_dot_product_groups: int,
    item_dot_product_groups: int,
    temperature: float,
    query_use_identity_fn: bool,
    query_dropout_rate: float,
    query_hidden_dim: int,
    item_use_identity_fn: bool,
    item_dropout_rate: float,
    item_hidden_dim: int,
    gating_query_hidden_dim: int,
    gating_qi_hidden_dim: int,
    gating_item_hidden_dim: int,
    softmax_dropout_rate: float,
    bf16_training: bool,
    gating_query_fn: bool = True,
    gating_item_fn: bool = True,
    dot_product_l2_norm: bool = True,
    query_nonlinearity: str = "geglu",
    item_nonlinearity: str = "geglu",
    uid_dropout_rate: float = 0.5,
    uid_embedding_hash_sizes: Optional[List[int]] = None,
    uid_embedding_level_dropout: bool = False,
    gating_combination_type: str = "glu_silu",
    gating_item_dropout_rate: float = 0.0,
    gating_qi_dropout_rate: float = 0.0,
    eps: float = 1e-6,
) -> Tuple[MoLSimilarity, str]:
    mol_module = MoLSimilarity(
        query_embedding_dim=query_embedding_dim,
        item_embedding_dim=item_embedding_dim,
        dot_product_dimension=dot_product_dimension,
        query_dot_product_groups=query_dot_product_groups,
        item_dot_product_groups=item_dot_product_groups,
        temperature=temperature,
        dot_product_l2_norm=dot_product_l2_norm,
        item_sideinfo_dim=0,
        query_embeddings_fn=RecoMoLQueryEmbeddingsFn(
            query_embedding_dim=query_embedding_dim,
            query_dot_product_groups=query_dot_product_groups,
            dot_product_dimension=dot_product_dimension,
            dot_product_l2_norm=dot_product_l2_norm,
            proj_fn=lambda input_dim, output_dim: IdentityMLPProjectionFn(
                input_dim=input_dim,
                output_num_features=query_dot_product_groups,
                output_dim=output_dim // query_dot_product_groups,
                input_dropout_rate=query_dropout_rate,
            ) if query_use_identity_fn else
                (
                    torch.nn.Sequential(
                        torch.nn.Dropout(p=query_dropout_rate),
                        GeGLU(in_features=input_dim, out_features=query_hidden_dim,) if query_nonlinearity == "geglu"
                        else SwiGLU(in_features=input_dim, out_features=query_hidden_dim,),
                        torch.nn.Linear(
                            in_features=query_hidden_dim,
                            out_features=output_dim,
                        ),
                    ) if query_hidden_dim > 0 else torch.nn.Sequential(
                        torch.nn.Dropout(p=query_dropout_rate),
                        torch.nn.Linear(
                            in_features=input_dim,
                            out_features=output_dim,
                       ),
                    )
                ).apply(init_mlp_xavier_weights_zero_bias),
            uid_embedding_hash_sizes=uid_embedding_hash_sizes or [],
            uid_dropout_rate=uid_dropout_rate,
            uid_embedding_level_dropout=uid_embedding_level_dropout,
            eps=eps,
        ),
        item_proj_fn=lambda input_dim, output_dim: IdentityMLPProjectionFn(
            input_dim=input_dim,
            output_num_features=item_dot_product_groups,
            output_dim=output_dim // item_dot_product_groups,
            input_dropout_rate=item_dropout_rate,
        ) if item_use_identity_fn else
            (
                torch.nn.Sequential(
                    torch.nn.Dropout(p=item_dropout_rate),
                    GeGLU(in_features=input_dim, out_features=item_hidden_dim,) if item_nonlinearity == "geglu"
                    else SwiGLU(in_features=input_dim, out_features=item_hidden_dim),
                    torch.nn.Linear(
                        in_features=item_hidden_dim,
                        out_features=output_dim,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias) if item_hidden_dim > 0 else
                torch.nn.Sequential(
                    torch.nn.Dropout(p=item_dropout_rate),
                    torch.nn.Linear(
                        in_features=input_dim,
                        out_features=output_dim,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias)
            ),
        gating_query_only_partial_fn=lambda input_dim, output_dim: torch.nn.Sequential(
            torch.nn.Linear(
                in_features=input_dim,
                out_features=gating_query_hidden_dim,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=gating_query_hidden_dim,
                out_features=output_dim,
                bias=False,
            ),
        ).apply(init_mlp_xavier_weights_zero_bias) if gating_query_fn else None,
        gating_item_only_partial_fn=lambda input_dim, output_dim: torch.nn.Sequential(
            torch.nn.Dropout(p=gating_item_dropout_rate),
            torch.nn.Linear(
                in_features=input_dim,
                out_features=gating_item_hidden_dim,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=gating_item_hidden_dim,
                out_features=output_dim,
                bias=False,
            ),
        ).apply(init_mlp_xavier_weights_zero_bias) if gating_item_fn else None,
        gating_qi_partial_fn=lambda input_dim, output_dim: torch.nn.Sequential(
            torch.nn.Dropout(p=gating_qi_dropout_rate),
            torch.nn.Linear(
                in_features=input_dim,
                out_features=gating_qi_hidden_dim,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=gating_qi_hidden_dim,
                out_features=output_dim,
            ),
        ).apply(init_mlp_xavier_weights_zero_bias) if gating_qi_hidden_dim > 0 else torch.nn.Sequential(
            torch.nn.Dropout(p=gating_qi_dropout_rate),
            torch.nn.Linear(
                in_features=input_dim,
                out_features=output_dim,
            ),
        ).apply(init_mlp_xavier_weights_zero_bias),
        gating_combination_type=gating_combination_type,
        gating_normalization_fn=lambda _: SoftmaxDropoutCombiner(dropout_rate=softmax_dropout_rate, eps=1e-6),
        eps=eps,
        bf16_training=bf16_training,
    )
    interaction_module_debug_str = (
        f"MoL-{query_dot_product_groups}x{item_dot_product_groups}x{dot_product_dimension}"
        + f"-t{temperature}-d{softmax_dropout_rate}"
        + f"{'-l2' if dot_product_l2_norm else ''}"
        + (f"-q{query_hidden_dim}d{query_dropout_rate}{query_nonlinearity}" if query_hidden_dim > 0 else f"-cd{query_dropout_rate}")
        + (
            "-i_id" if item_use_identity_fn else
            (f"-{item_hidden_dim}d{item_dropout_rate}-{item_nonlinearity}" if item_hidden_dim > 0 else f"-id{item_dropout_rate}")
        )
        + (f"-gq{gating_query_hidden_dim}" if gating_query_fn else "")
        + (f"-gi{gating_item_hidden_dim}d{gating_item_dropout_rate}" if gating_item_fn else "")
        + f"-gqi{gating_qi_hidden_dim}d{gating_qi_dropout_rate}-x-{gating_combination_type}"
    )
    if uid_embedding_hash_sizes is not None:
        if uid_embedding_hash_sizes is not None:
            interaction_module_debug_str += f"-uids{'-'.join([str(x) for x in uid_embedding_hash_sizes])}"
        if uid_dropout_rate > 0.0:
            interaction_module_debug_str += f"d{uid_dropout_rate}"
        if uid_embedding_level_dropout:
            interaction_module_debug_str += "-el"
    return mol_module, interaction_module_debug_str

def init_mlp_xavier_weights_zero_bias(m) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.fill_(0.0)