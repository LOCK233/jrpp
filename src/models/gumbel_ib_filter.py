import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.gumbel import Gumbel


def gumbel_softmax_topk(
    logits: torch.Tensor,
    k: int,
    tau: float = 1.0,
    hard: bool = True,
    training: bool = True,
):
    k = min(k, logits.size(1))
    batch_size, num_candidates = logits.shape
    if training:
        gumbel = Gumbel(0, 1).sample(logits.shape).to(logits.device)
        selection_logits = logits + gumbel
    else:
        selection_logits = logits

    topk_idx = torch.topk(selection_logits, k, dim=-1)[1]
    hard_selection = torch.zeros(
        batch_size,
        k,
        num_candidates,
        dtype=logits.dtype,
        device=logits.device,
    ).scatter_(-1, topk_idx.unsqueeze(-1), 1.0)

    if not training:
        return hard_selection, hard_selection, topk_idx

    selected_mass = torch.zeros_like(logits)
    soft_rows = []
    for _ in range(k):
        remaining = torch.clamp(1.0 - selected_mass, min=1e-6)
        row_logits = selection_logits + remaining.log()
        row = F.softmax(row_logits / tau, dim=-1)
        soft_rows.append(row)
        selected_mass = selected_mass + row
    soft_selection = torch.stack(soft_rows, dim=1)

    if not hard:
        return soft_selection, soft_selection, topk_idx

    straight_through_selection = soft_selection - soft_selection.detach() + hard_selection
    return hard_selection, straight_through_selection, topk_idx


class InfoBottleneckCompressor(nn.Module):
    def __init__(self, d_selected: int, d_query: int, d_z: int, beta: float = 1e-3):
        super().__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(d_selected + d_query, 512),
            nn.ReLU(),
            nn.Linear(512, d_z * 2),
        )

    def forward(self, selected: torch.Tensor, query: torch.Tensor):
        _, k, _ = selected.shape
        query = query.unsqueeze(1).expand(-1, k, -1)
        params = self.encoder(torch.cat([selected, query], dim=-1))
        mu, log_sigma = params.chunk(2, dim=-1)
        std = torch.exp(log_sigma.clamp(-10, 10))
        z = mu + std * torch.randn_like(std) if self.training else mu
        kl = 0.5 * (mu.pow(2) + std.pow(2) - 1 - 2 * log_sigma).sum(-1).mean()
        return z, kl * self.beta


class GumbelIBFilter(nn.Module):
    def __init__(
        self,
        cand_dim: int,
        query_dim: int,
        hidden_dim: int = 256,
        ib_dim: int = 512,
        select_k: int = 50,
        tau: float = 1.0,
        beta: float = 1e-3,
        use_score: bool = False,
    ):
        super().__init__()
        self.select_k = select_k
        self.tau = tau
        self.use_score = use_score

        in_dim = cand_dim + query_dim + (1 if use_score else 0)
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.compressor = InfoBottleneckCompressor(
            d_selected=cand_dim,
            d_query=query_dim,
            d_z=ib_dim,
            beta=beta,
        )

    def forward(self, query_emb: torch.Tensor, cand_emb: torch.Tensor, cand_score: torch.Tensor):
        _, k, d_c = cand_emb.shape
        select_k = min(self.select_k, k)

        query = query_emb.unsqueeze(1).expand(-1, k, -1)
        scorer_input = torch.cat([cand_emb, query], dim=-1)
        if self.use_score and cand_score is not None:
            scorer_input = torch.cat([scorer_input, cand_score.unsqueeze(-1)], dim=-1)
        logits = self.scorer(scorer_input).squeeze(-1)

        hard_selection, selection, _ = gumbel_softmax_topk(
            logits,
            select_k,
            self.tau,
            hard=True,
            training=self.training,
        )
        selected = torch.bmm(selection, cand_emb)
        selected_score = torch.bmm(selection, logits.unsqueeze(-1)).squeeze(-1)
        selected_raw_score = (
            torch.bmm(selection, cand_score.unsqueeze(-1)).squeeze(-1)
            if cand_score is not None
            else selected_score
        )

        refined, kl_loss = self.compressor(selected, query_emb)
        return refined, selection, kl_loss, hard_selection, selected_score, selected_raw_score
