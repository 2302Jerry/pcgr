from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .utils import Graph, pairwise_cosine, safe_log

@dataclass
class GraphBuildConfig:
    alpha_sem: float = 0.6
    beta_pmi: float = 0.6
    gamma_ent: float = 0.4
    delta_contr: float = 0.4
    zeta: float = 0.55
    use_nli: bool = True
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base"
    max_parents: int = 4

class NLIScorer:
    def __init__(self, model_name: str, device: torch.device):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        # Most NLI models use label mapping: contradiction, neutral, entailment (but may vary).
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def score(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        inputs = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        # Map to entail/neutral/contradiction with best-effort
        ent = neu = con = 0.0
        for i, p in enumerate(probs):
            lab = self.id2label.get(i, str(i))
            if "entail" in lab:
                ent = float(p)
            elif "neutral" in lab:
                neu = float(p)
            elif "contrad" in lab:
                con = float(p)
        # Fallback if mapping isn't present (assume 0:contr,1:neu,2:ent)
        if (ent + neu + con) == 0.0 and len(probs) >= 3:
            con, neu, ent = float(probs[0]), float(probs[1]), float(probs[2])
        return ent, neu, con

def soft_pmi(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute Soft-PMI matrix from probabilities p over samples.
    p: (N, K) in [0,1]
    returns: (K, K) pmi_ij
    """
    N, K = p.shape
    pi = p.mean(dim=0)  # (K,)
    pij = (p[:, :, None] * p[:, None, :]).mean(dim=0)  # (K,K)
    pmi = safe_log(pij + eps) - safe_log(pi[:, None] * pi[None, :] + eps)
    return pmi

def build_layered_graph(
    concept_texts: List[str],
    concept_emb: torch.Tensor,  # (K, d_emb) semantic/attention embedding
    layers: List[List[int]],
    p_val: torch.Tensor,        # (N, K) probabilities on a validation set
    cfg: GraphBuildConfig,
    device: torch.device,
) -> Graph:
    """Build edges only between adjacent layers (L_r -> L_{r+1}) using Eq.(4)-style score."""
    K = len(concept_texts)
    parents: List[List[int]] = [[] for _ in range(K)]
    children: List[List[int]] = [[] for _ in range(K)]

    # Precompute components
    concept_emb_n = F.normalize(concept_emb, dim=-1)
    pmi = soft_pmi(p_val.to(device)).detach()

    nli = NLIScorer(cfg.nli_model_name, device=device) if cfg.use_nli else None

    for r in range(len(layers) - 1):
        lower = layers[r]
        upper = layers[r + 1]
        for i in lower:
            scores = []
            for j in upper:
                sem = float((concept_emb_n[i] * concept_emb_n[j]).sum().detach().cpu())
                spmi = float(pmi[i, j].detach().cpu())
                ent = con = 0.0
                if nli is not None:
                    ent, _, con = nli.score(concept_texts[i], concept_texts[j])
                s = cfg.alpha_sem * sem + cfg.beta_pmi * spmi + cfg.gamma_ent * ent - cfg.delta_contr * con
                scores.append((j, s))

            # pick parents above threshold, keep top max_parents
            scores.sort(key=lambda x: x[1], reverse=True)
            kept = [(j, s) for (j, s) in scores if s > cfg.zeta][: cfg.max_parents]
            for j, s in kept:
                parents[i].append(j)
                children[j].append(i)

    # Define root as the last layer's single node if possible; else the last concept
    root = layers[-1][0] if layers and len(layers[-1]) == 1 else (K - 1)
    return Graph(layers=layers, parents=parents, children=children, root=root)
