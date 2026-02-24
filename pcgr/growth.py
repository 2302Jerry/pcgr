from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

# --------- Concept proposal (plug-in) ----------

ConceptGenerator = Callable[[List[str]], List[str]]
"""Given a list of *seed texts* (e.g., misclassified samples), return a list of new concept strings."""

def heuristic_generator(seed_texts: List[str], max_new: int = 5) -> List[str]:
    """Offline, runnable concept generator:
    - Extracts TF-IDF keywords from seed texts
    - Wrap them into interrogative, reusable concept questions
    """
    if not seed_texts:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=50, ngram_range=(1, 2))
    X = vec.fit_transform(seed_texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    top = vocab[np.argsort(scores)[::-1][:max_new]]
    concepts = []
    for kw in top:
        kw_clean = re.sub(r"\s+", " ", kw.strip())
        concepts.append(f"Does the text emphasize or mention '{kw_clean}' in a potentially misleading way?")
    return concepts

# --------- Paper-style filters ----------

@dataclass
class GrowthFilterConfig:
    semantic_uniqueness_thresh: float = 0.8  # keep if max cosine <= 0.8
    corr_thresh: float = 0.9                 # keep if |corr| <= 0.9
    activation_range: Tuple[float, float] = (0.05, 0.95)

def filter_concepts(
    new_concepts: List[str],
    existing_concepts: List[str],
    sbert_embedder,
    p_val_existing: torch.Tensor,  # (N, K_existing)
    p_val_new: torch.Tensor,       # (N, K_new) predicted with the *same model* after adding candidate heads
    cfg: GrowthFilterConfig,
) -> List[str]:
    """Implements the 3-stage filter:
    1) semantic uniqueness via cosine similarity
    2) statistical independence via Pearson correlation on validation predictions
    3) informative activation via mean probability range
    """
    if not new_concepts:
        return []

    # 1) Semantic uniqueness
    ex_emb = torch.tensor(sbert_embedder.encode(existing_concepts, normalize_embeddings=True), dtype=torch.float32)
    new_emb = torch.tensor(sbert_embedder.encode(new_concepts, normalize_embeddings=True), dtype=torch.float32)
    cos = torch.matmul(new_emb, ex_emb.t())  # (K_new, K_old)
    max_sim = cos.max(dim=1).values
    keep_sem = max_sim <= cfg.semantic_uniqueness_thresh

    kept = []
    kept_idx = []
    for i, ok in enumerate(keep_sem.tolist()):
        if ok:
            kept.append(new_concepts[i])
            kept_idx.append(i)

    if not kept:
        return []

    # 2) Statistical independence (Pearson corr)
    # Compare each new concept prob with existing concept probs; keep if all |corr| <= thresh
    p_old = p_val_existing.detach().cpu().numpy()
    p_new = p_val_new[:, kept_idx].detach().cpu().numpy()
    ok2 = []
    for j in range(p_new.shape[1]):
        x = p_new[:, j]
        good = True
        for k in range(p_old.shape[1]):
            y = p_old[:, k]
            if np.std(x) < 1e-6 or np.std(y) < 1e-6:
                continue
            corr = np.corrcoef(x, y)[0, 1]
            if abs(corr) > cfg.corr_thresh:
                good = False
                break
        ok2.append(good)

    kept2 = [c for c, ok in zip(kept, ok2) if ok]
    if not kept2:
        return []

    # 3) Informative activation (mean prob in [low, high])
    low, high = cfg.activation_range
    means = p_new.mean(axis=0)
    final = []
    for c, m, ok in zip(kept, means, ok2):
        if ok and (low <= float(m) <= high):
            final.append(c)
    return final

"""To plug in an MLLM:
  - implement a function like `openai_generator(seed_texts)->concepts`
  - pass it to your training loop
"""
