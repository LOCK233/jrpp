import math

import torch
from torch import nn
import torch.nn.functional as F

from models.TransformerBlock import TransformerBlock
from models.gumbel_ib_filter import GumbelIBFilter
from models.mol_similarity import MoLSimilarity
from models.retrieval import Retrieval


class JRPP(nn.Module):
    """Joint retrieval and popularity prediction model."""

    def __init__(self, args, config, meta_dim: int, dropout: float = 0.2):
        super().__init__()

        self.config = config
        self.emb_size = int(args.embSize)
        self.data_name = args.data_name
        self.meta_dim = int(meta_dim)
        self.input_dim = self.emb_size * 2 + self.meta_dim
        self.fusion_dim = self.emb_size + self.emb_size // 2

        retrieval_config = config["retrieval"]
        filter_config = config.get("filter", {})
        top_k = int(retrieval_config.get("top_k", 50))
        keep_ratio = float(filter_config.get("keep_ratio", 0.8))
        select_k = max(1, int(top_k * keep_ratio))
        ib_dim = int(filter_config.get("ib_dim", 512))

        self.text_projection = nn.Linear(768, self.emb_size)
        self.image_projection = nn.Linear(768, self.emb_size)

        self.mol_similarity = MoLSimilarity(retrieval_config["mol"])
        self.retrieval = Retrieval(retrieval_config, model=self)
        self.gumbel_ib_filter = GumbelIBFilter(
            cand_dim=self.input_dim,
            query_dim=self.input_dim,
            hidden_dim=int(filter_config.get("hidden_dim", 256)),
            ib_dim=ib_dim,
            select_k=select_k,
            tau=float(filter_config.get("tau", 1.0)),
            beta=float(filter_config.get("beta", 1e-11)),
            use_score=bool(filter_config.get("use_score", True)),
        )

        self.query_projection = nn.Linear(self.input_dim, self.fusion_dim)
        self.context_projection = nn.Linear(ib_dim, self.fusion_dim)
        self.joint_projection = nn.Linear(self.input_dim + ib_dim, self.fusion_dim)
        self.fusing_attn = TransformerBlock(input_size=self.fusion_dim, n_heads=4, attn_dropout=dropout)
        self.regressor = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_dim, 1),
        )

        self.loss_function = nn.MSELoss()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.emb_size)
        for parameter in self.parameters():
            parameter.data.uniform_(-stdv, stdv)

    def _project_items(
        self,
        items_text: torch.Tensor,
        items_img: torch.Tensor,
        items_meta: torch.Tensor,
    ) -> torch.Tensor:
        text = self.text_projection(items_text)
        image = self.image_projection(items_img)
        meta = items_meta.squeeze(1)
        return torch.cat([text, image, meta], dim=-1).unsqueeze(0)

    def _project_query(
        self,
        text_vec: torch.Tensor,
        img_vec_cls: torch.Tensor,
        meta_features: torch.Tensor,
    ) -> torch.Tensor:
        text = self.text_projection(text_vec)
        image = self.image_projection(img_vec_cls)
        return torch.cat([text, image, meta_features], dim=-1)

    def forward(
        self,
        text_vec,
        img_vec_pool,
        meta_features,
        img_vec_cls,
        user_id,
        image_id,
        items_id,
        items_text,
        items_img,
        items_meta,
    ):
        del img_vec_pool

        input_embeddings = self._project_query(text_vec, img_vec_cls, meta_features)
        items_embeddings = self._project_items(items_text, items_img, items_meta)

        topk_model = self.retrieval.get_topk_related_items(items_embeddings, items_id)
        top_k_scores, top_k_indices = topk_model(
            input_embeddings,
            int(self.config["retrieval"]["top_k"]),
            user_ids=user_id,
            exclude_item_ids=image_id,
        )
        top_k_embeddings = items_embeddings.squeeze(0)[top_k_indices]

        refined, _, kl_loss, _, _, row_score = self.gumbel_ib_filter(
            query_emb=input_embeddings,
            cand_emb=top_k_embeddings,
            cand_score=top_k_scores,
        )
        attn_weight = F.softmax(row_score, dim=1)
        context_raw = torch.bmm(attn_weight.unsqueeze(1), refined)

        query = self.query_projection(input_embeddings).unsqueeze(1)
        context = self.context_projection(context_raw.squeeze(1)).unsqueeze(1)
        joint = self.joint_projection(torch.cat([input_embeddings.unsqueeze(1), context_raw], dim=-1))

        output = self.fusing_attn(query, context, joint)
        output = self.regressor(output.squeeze(1))
        return output, kl_loss
