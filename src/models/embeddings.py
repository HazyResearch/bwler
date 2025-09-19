"""
Embedding functions for input preprocessing.
Ported from JAX implementation in cheb_embedding.py
"""

import torch
import torch.nn as nn
import math
from typing import Optional


import torch
import torch.nn as nn
import math
from typing import Optional

def lobatto_nodes_weights(N: int, dtype: torch.dtype = torch.float64, device: str = "cpu"):
    j = torch.arange(N + 1, dtype=dtype, device=device)
    x = torch.cos(math.pi * j / N)
    delta = torch.where((j == 0) | (j == N), 0.5, 1.0)
    w = ((-1.0) ** j) * delta
    return x, w  # shapes: (N+1,), (N+1,)

def _local_spacings(nodes: torch.Tensor) -> torch.Tensor:
    """
    Per-node local spacing h_j = min(|x_j - x_{j-1}|, |x_{j+1} - x_j|)
    Endpoints use the one-sided neighbor.
    """
    dx = torch.abs(nodes[1:] - nodes[:-1])                # (M-1,)
    left = torch.cat([dx[:1], dx])                        # (M,)
    right = torch.cat([dx, dx[-1:]])                      # (M,)
    return torch.minimum(left, right)

def barycentric_lambdas(
    x: torch.Tensor,
    nodes: torch.Tensor,
    w: torch.Tensor,
    *,
    snap_eps_abs: float = 1e-15,
    snap_radii: Optional[torch.Tensor] = None,
    tiny_den: float = 1e-300,
):
    """
    Compute λ(x) row-wise for a batch x:(B,) against fixed nodes/w:(M,).
    Returns (B,M). If x is within a snap radius of a node, returns exact one-hot.

    Args:
      x: (B,) or (B,1) tensor
      nodes: (M,)
      w: (M,)
      snap_eps_abs: absolute snap epsilon (ignored if snap_radii provided)
      snap_radii: optional (M,) per-node snap radii
      tiny_den: floor for |x - x_j| in non-snapped rows to avoid inf/NaN
    """
    x = x.flatten().to(nodes.dtype)
    B, M = x.shape[0], nodes.shape[0]

    # distances to each node
    diff = x[:, None] - nodes[None, :]                   # (B,M)
    absdiff = torch.abs(diff)

    # nearest node per row
    k_min = torch.argmin(absdiff, dim=1)                 # (B,)

    if snap_radii is None:
        # broadcast scalar absolute radius to nearest node
        r_sel = torch.full_like(k_min, fill_value=snap_eps_abs, dtype=nodes.dtype)
    else:
        # pick the per-node radius of the nearest node for each row
        r_sel = snap_radii[k_min]                        # (B,)

    # decide which rows to snap
    d_sel = absdiff.gather(1, k_min[:, None])[:, 0]      # (B,)
    snap_mask = d_sel <= r_sel                           # (B,)

    # --- barycentric formula on non-snapped rows ---
    # clamp tiny denominators away from zero to avoid overflow
    safe_diff = torch.where(absdiff < tiny_den, torch.sign(diff) * tiny_den, diff)
    z = w[None, :] / safe_diff                           # (B,M)
    S = torch.sum(z, dim=1, keepdims=True)               # (B,1)
    lam = z / S                                          # (B,M)

    # overwrite snapped rows with exact one-hot at k_min
    onehot = torch.nn.functional.one_hot(k_min, M).to(nodes.dtype)
    lam = torch.where(snap_mask[:, None], onehot, lam)
    return lam

class EmbedNone(nn.Module):
    """No embedding - input is x ∈ [-1, 1]"""
    
    def __init__(self, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.dtype = dtype
        
    def forward(self, x):
        # x: (B,) or (B, 1)
        if x.dim() == 1:
            x = x[:, None]
        return x.to(self.dtype)  # (B,1)
    
    @property
    def output_dim(self):
        return 1


class EmbedTheta(nn.Module):
    """Theta embedding - input is θ = arccos(x) ∈ [0, π]"""
    
    def __init__(self, eps: float = 1e-7, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        
    def forward(self, x):
        # θ = arccos(x_clipped), shape (B,1)
        x = x.flatten()
        x_clip = torch.clamp(x, -1.0 + self.eps, 1.0 - self.eps)
        theta = torch.arccos(x_clip)
        return theta[:, None].to(self.dtype)
    
    @property
    def output_dim(self):
        return 1


class EmbedCheb(nn.Module):
    """Chebyshev embedding - features = [cos(θ), cos(2θ), ..., cos(Mθ)]"""
    
    def __init__(self, M: int, eps: float = 1e-7, dtype: torch.dtype = torch.float64):
        super().__init__()
        if M <= 0:
            raise ValueError("M must be > 0 for Chebyshev embedding")
        self.M = M
        self.eps = eps
        self.dtype = dtype
        # Pre-compute the indices for efficiency
        self.register_buffer('js', torch.arange(1, M + 1, dtype=dtype))
        
    def forward(self, x):
        # features = [cos(θ), cos(2θ), ..., cos(Mθ)], shape (B,M)
        x = x.flatten()
        x_clip = torch.clamp(x, -1.0 + self.eps, 1.0 - self.eps)
        theta = torch.arccos(x_clip)  # (B,)
        feats = torch.cos(theta[:, None] * self.js[None, :])  # (B,M)
        return feats.to(self.dtype)
    
    @property
    def output_dim(self):
        return self.M



class EmbedBary(nn.Module):
    """Barycentric embedding using Chebyshev–Lobatto nodes with ε-snap near nodes."""
    def __init__(
        self,
        N: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        snap_eps_rel: float = 0.0,
        snap_eps_abs: float = 1e-14,
        tiny_den: float = 1e-300,
    ):
        super().__init__()
        if N <= 0:
            raise ValueError("N must be > 0 for barycentric embedding")
        self.N = N
        self.dtype = dtype
        self.snap_eps_rel = snap_eps_rel
        self.snap_eps_abs = snap_eps_abs
        self.tiny_den = tiny_den

        # Pre-compute nodes, weights, and per-node snap radii
        nodes, w = lobatto_nodes_weights(N, dtype=dtype, device=device)
        self.register_buffer('nodes', nodes)
        self.register_buffer('w', w)

        # Per-node local spacing, then radius = abs + rel * spacing
        h = _local_spacings(nodes)                                        # (N+1,)
        snap_r = snap_eps_abs + snap_eps_rel * h
        self.register_buffer('snap_radii', snap_r)

    def forward(self, x):
        # x: (B,) -> lambdas: (B, N+1)
        return barycentric_lambdas(
            x, self.nodes, self.w,
            snap_eps_abs=self.snap_eps_abs,
            snap_radii=self.snap_radii,
            tiny_den=self.tiny_den,
        )

    @property
    def output_dim(self):
        return self.N + 1




def make_embedding(kind: str, M: Optional[int] = None, dtype: torch.dtype = torch.float64, device: str = "cpu"):
    """Factory function to create embeddings"""
    if kind == "none":
        return EmbedNone(dtype=dtype)
    elif kind == "theta":
        return EmbedTheta(dtype=dtype)
    elif kind == "cheb":
        if M is None or M <= 0:
            raise ValueError("For 'cheb' embedding you must set M > 0.")
        return EmbedCheb(M=M, dtype=dtype)
    elif kind == "bary":
        if M is None or M <= 0:
            raise ValueError("For 'bary' embedding you must set M > 0 (degree N).")
        # Here M is interpreted as degree N => N+1 lambdas
        return EmbedBary(N=M, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown embedding kind: {kind}")


def infer_embedding_dim(embed_kind: str, M: Optional[int] = None):
    """Infer the output dimension of an embedding"""
    if embed_kind == "none":  
        return 1
    if embed_kind == "theta": 
        return 1
    if embed_kind == "cheb":  
        if M is None or M <= 0:
            raise ValueError("M must be specified and > 0 for 'cheb' embedding")
        return M
    if embed_kind == "bary":  
        if M is None or M <= 0:
            raise ValueError("M must be specified and > 0 for 'bary' embedding")
        return M + 1  # N+1 lambdas
    raise ValueError(f"Unknown embedding kind: {embed_kind}")
