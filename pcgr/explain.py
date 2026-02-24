from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .utils import Graph

@dataclass
class Explanation:
    y_hat: float
    top_concepts: List[Tuple[str, float]]
    chains: List[List[Tuple[str, float]]]

def _dominant_parent(node: int, parents: List[int], attn: Dict[int, torch.Tensor], p_hat_row: torch.Tensor) -> Optional[int]:
    if not parents:
        return None
    # Choose parent maximizing α * p_hat(parent) for this sample
    best = None
    best_val = -1.0
    for p in parents:
        a = float(attn.get(p, torch.tensor(0.0))[0].item()) if p in attn else 0.0
        val = a * float(p_hat_row[p].item())
        if val > best_val:
            best_val = val
            best = p
    return best

def explain_sample(
    concept_texts: List[str],
    graph: Graph,
    p_hat: torch.Tensor,     # (K,)
    attn_list: List[Dict[int, torch.Tensor]],
    topk: int = 6,
    max_chain_len: int = 5,
) -> Explanation:
    # Top concepts excluding root for readability
    K = len(concept_texts)
    root = graph.root
    idxs = [i for i in range(K) if i != root]
    vals = [(i, float(p_hat[i].item())) for i in idxs]
    vals.sort(key=lambda x: x[1], reverse=True)
    top = vals[:topk]
    top_concepts = [(concept_texts[i], s) for i, s in top]

    chains: List[List[Tuple[str, float]]] = []
    for i, _ in top:
        chain: List[Tuple[str, float]] = [(concept_texts[i], float(p_hat[i].item()))]
        cur = i
        for _ in range(max_chain_len):
            ps = graph.parents[cur]
            if not ps:
                break
            dom = _dominant_parent(cur, ps, attn_list[cur], p_hat)
            if dom is None:
                break
            chain.append((concept_texts[dom], float(p_hat[dom].item())))
            cur = dom
            if cur == root:
                break
        # Ensure root at end (if reachable)
        if chain[-1][0] != concept_texts[root]:
            chain.append((concept_texts[root], float(p_hat[root].item())))
        chains.append(chain)

    return Explanation(
        y_hat=float(p_hat[root].item()),
        top_concepts=top_concepts,
        chains=chains,
    )
