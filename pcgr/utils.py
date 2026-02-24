from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict, Optional

import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device_of(t: torch.Tensor) -> torch.device:
    return t.device

def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))

def pairwise_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: (..., d), y: (..., d)
    x_n = x / (x.norm(dim=-1, keepdim=True) + eps)
    y_n = y / (y.norm(dim=-1, keepdim=True) + eps)
    return (x_n * y_n).sum(dim=-1)

@dataclass
class Graph:
    # Directed edges are stored as "child -> parent" (lower layer -> higher layer).
    # For top-down inference we need parents[child] (higher layer indices).
    layers: List[List[int]]
    parents: List[List[int]]  # len K
    children: List[List[int]] # len K
    root: int

    def topological_order_top_down(self) -> List[int]:
        # Parents are higher-layer nodes, so compute from top layer -> bottom layer
        order: List[int] = []
        for layer in reversed(self.layers):  # highest -> lowest
            order.extend(layer)
        return order

