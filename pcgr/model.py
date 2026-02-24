from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Graph

@dataclass
class PCGROutput:
    p_concept: torch.Tensor     # (B, K) raw concept probabilities p_i
    p_agg: torch.Tensor         # (B, K) aggregated probabilities p_hat_i
    y_hat: torch.Tensor         # (B,) final prediction (root concept)
    attn: List[Dict[int, torch.Tensor]]  # per child i: attention weights over parents (B, |Pa(i)|)

class PCGRModel(nn.Module):
    """A practical implementation of PCGR's core inference:
    - CLIP encodes (text, image)
    - Per-concept probability computed via low-rank interaction
    - Top-down hierarchical attention + multiplicative aggregation
    """

    def __init__(
        self,
        clip_model,
        clip_dim: int,
        concept_texts: List[str],
        sbert_embed_dim: int,
        d_model: int = 256,
        r_rank: int = 64,
        attn_dim: int = 256,
        rho: float = 0.35,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.clip_dim = clip_dim

        self.concept_texts = list(concept_texts)
        self.K = len(concept_texts)

        # Project CLIP embeddings into d_model
        self.text_proj = nn.Linear(clip_dim, d_model)
        self.image_proj = nn.Linear(clip_dim, d_model)

        # Trainable concept embeddings (initialized from SBERT in build_concept_embeddings())
        self.concept_q = nn.Parameter(torch.empty(self.K, attn_dim))
        nn.init.normal_(self.concept_q, std=0.02)

        self.concept_proj = nn.Linear(sbert_embed_dim, d_model)

        # Prototype networks (shared)
        in_dim = d_model * 3  # v ⊕ t ⊕ d (concept-conditioned text)
        hidden = d_model * 2
        self.phi_plus = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.phi_minus = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.tau_logits = nn.Parameter(torch.zeros(self.K))  # learnable per-concept mixing

        # Low-rank interaction (Eq.3-style)
        self.U = nn.Parameter(torch.randn(d_model, r_rank) * 0.02)
        self.V = nn.Parameter(torch.randn(d_model, r_rank) * 0.02)
        self.g_mlp = nn.Sequential(nn.Linear(sbert_embed_dim, r_rank), nn.GELU(), nn.Linear(r_rank, r_rank))
        self.w = nn.Parameter(torch.randn(self.K, r_rank) * 0.02)
        self.b = nn.Parameter(torch.zeros(self.K))

        self.rho = float(rho)

        # Buffers set later
        self.register_buffer("concept_sbert", torch.empty(self.K, sbert_embed_dim), persistent=False)

    @torch.no_grad()
    def init_concept_q_from_sbert(self) -> None:
        # Simple linear projection + normalize into attn space
        # If dimensions differ, we use a random orthogonal-ish projection.
        K, sdim = self.concept_sbert.shape
        # create a fixed random projection for init
        proj = torch.randn(sdim, self.concept_q.shape[1], device=self.concept_sbert.device) / (sdim ** 0.5)
        q = self.concept_sbert @ proj
        q = F.normalize(q, dim=-1)
        self.concept_q.copy_(q)

    def forward(self, batch: Dict[str, torch.Tensor], graph: Optional[Graph] = None) -> PCGROutput:
        # CLIP encodings
        clip_out = self.clip_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            return_dict=True,
        )
        t = clip_out.text_embeds  # (B, clip_dim)
        v = clip_out.image_embeds # (B, clip_dim)
        t = F.normalize(t, dim=-1)
        v = F.normalize(v, dim=-1)

        t = self.text_proj(t)     # (B, d)
        v = self.image_proj(v)    # (B, d)

        # Concept-conditioned representation h_i (Eq.1-2 in spirit)
        # Use the concept text embedding as the "description embedding" by default.
        e = self.concept_sbert  # (K, sdim)
        d = self.concept_proj(e)  # (K, d)

        B, K, D = t.shape[0], self.K, t.shape[1]
        t_exp = t[:, None, :].expand(B, K, D)
        v_exp = v[:, None, :].expand(B, K, D)
        d_exp = d[None, :, :].expand(B, K, D)
        x = torch.cat([v_exp, t_exp, d_exp], dim=-1)  # (B, K, 3d)

        h_plus = self.phi_plus(x)   # (B,K,d)
        h_minus = self.phi_minus(x) # (B,K,d)
        tau = torch.sigmoid(self.tau_logits)[None, :, None]  # (1,K,1)
        h = tau * h_plus + (1.0 - tau) * h_minus             # (B,K,d)

        # Low-rank interaction to get per-concept probability (Eq.3-like)
        u = torch.einsum("bkd,dr->bkr", h, self.U)
        vv = torch.einsum("bkd,dr->bkr", h, self.V)
        g = self.g_mlp(e)  # (K,r)
        inter = u * vv * g[None, :, :]  # (B,K,r)
        logits = (inter * self.w[None, :, :]).sum(dim=-1) + self.b[None, :]  # (B,K)
        p = torch.sigmoid(logits)

        if graph is None:
            # Flat fallback: assume last concept is root.
            root = self.K - 1
            y_hat = p[:, root]
            return PCGROutput(p_concept=p, p_agg=p, y_hat=y_hat, attn=[])

        p_hat, attn = self._top_down_infer(p, graph)
        y_hat = p_hat[:, graph.root]
        return PCGROutput(p_concept=p, p_agg=p_hat, y_hat=y_hat, attn=attn)

    def _top_down_infer(self, p: torch.Tensor, graph: Graph) -> Tuple[torch.Tensor, List[Dict[int, torch.Tensor]]]:
        """Top-down hierarchical attention + multiplicative aggregation (Eq.5-6 style)."""
        B, K = p.shape
        device = p.device
        p_hat = p.clone()
        attn_list: List[Dict[int, torch.Tensor]] = [dict() for _ in range(K)]

        # Use trainable concept_q as attention embeddings
        q = F.normalize(self.concept_q, dim=-1)  # (K, attn_dim)

        order = graph.topological_order_top_down()  # highest layer -> lowest layer
        for node in order:
            parents = graph.parents[node]
            if not parents:
                continue
            # attention weights α_{node,parent} = softmax(q_node · q_parent)
            qn = q[node]  # (attn_dim,)
            qp = q[parents]  # (P, attn_dim)
            # (P,)
            scores = torch.matmul(qp, qn)  # dot with node, shape (P,)
            alpha = torch.softmax(scores, dim=0)  # (P,)
            # Broadcast to batch
            alpha_b = alpha[None, :].expand(B, -1)  # (B,P)
            parent_vals = p_hat[:, parents]         # (B,P)

            # multiplicative "AND"-like aggregation of parent priors
            prod = torch.prod(alpha_b * parent_vals, dim=-1)  # (B,)

            # confusion parameter rho mixes local evidence p(node) and parent priors
            p_hat[:, node] = self.rho * p[:, node] + (1.0 - self.rho) * prod

            attn_list[node] = {parents[i]: alpha_b[:, i].detach().clone() for i in range(len(parents))}

        return p_hat, attn_list

    def orthogonality_loss(self, eps: float = 1e-8) -> torch.Tensor:
        """Concept orthogonality regularizer similar to Eq.(8) on trainable q."""
        q = self.concept_q
        q = q / (q.norm(dim=-1, keepdim=True) + eps)
        sim = torch.matmul(q, q.t())
        K = sim.shape[0]
        mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
        return sim[mask].abs().mean()
