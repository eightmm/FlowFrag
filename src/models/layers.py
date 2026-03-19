"""Shared GatedGCN layer used by both ProteinEncoder and LigandEncoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing


class GatedGCNLayer(MessagePassing):
    """Gated Graph Convolutional layer with edge gating.

    Node update:
        h_i' = h_i + ReLU(BN(A h_i + sum_j(B h_j * sigma(e_ij))))
    Edge update:
        e_ij' = e_ij + ReLU(BN(C h_i + D h_j + E e_ij))
        sigma(e_ij) = sigmoid(e_ij') / (sum_j sigmoid(e_ij') + eps)
    """

    def __init__(self, node_dim: int, edge_dim: int, eps: float = 1e-6) -> None:
        super().__init__(aggr="add")
        self.eps = eps

        self.A = nn.Linear(node_dim, node_dim, bias=False)
        self.B = nn.Linear(node_dim, node_dim, bias=False)
        self.C = nn.Linear(node_dim, edge_dim, bias=False)
        self.D = nn.Linear(node_dim, edge_dim, bias=False)
        self.E = nn.Linear(edge_dim, edge_dim, bias=False)

        self.bn_node = nn.BatchNorm1d(node_dim)
        self.bn_edge = nn.BatchNorm1d(edge_dim)

        self.node_bias = nn.Parameter(torch.zeros(node_dim))
        self.edge_bias = nn.Parameter(torch.zeros(edge_dim))

    def forward(
        self, h: Tensor, edge_index: Tensor, e: Tensor
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]

        # Edge update
        e_new = self.C(h[src]) + self.D(h[dst]) + self.E(e) + self.edge_bias
        e_new = torch.relu(self.bn_edge(e_new))
        e = e + e_new  # residual

        # Compute normalized edge gates sigma(e_ij)
        gate = torch.sigmoid(e)
        # Normalize gates per destination node
        gate_sum = torch.zeros(h.shape[0], gate.shape[1], device=h.device, dtype=h.dtype)
        gate_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(gate), gate)
        gate_norm = gate / (gate_sum[dst] + self.eps)

        # Node update: aggregate gated neighbor messages
        msg = self.B(h[src]) * gate_norm  # [E, node_dim]
        agg = torch.zeros(h.shape[0], msg.shape[1], device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)

        h_new = self.A(h) + agg + self.node_bias
        h_new = torch.relu(self.bn_node(h_new))
        h = h + h_new  # residual

        return h, e

    def message(self, x_j: Tensor) -> Tensor:  # type: ignore[override]
        # Not used (manual scatter for gate normalization)
        return x_j


def rbf_encode(
    dist: Tensor,
    n_rbf: int = 16,
    d_min: float = 0.0,
    d_max: float = 16.0,
) -> Tensor:
    """Radial basis function encoding of distances.

    Args:
        dist: Tensor of shape ``[E]`` or ``[E, 1]``.
        n_rbf: Number of RBF centers.
        d_min: Minimum distance.
        d_max: Maximum distance.

    Returns:
        Tensor of shape ``[E, n_rbf]``.
    """
    dist = dist.view(-1, 1)
    centers = torch.linspace(d_min, d_max, n_rbf, device=dist.device, dtype=dist.dtype)
    width = (d_max - d_min) / n_rbf
    return torch.exp(-((dist - centers) ** 2) / (2 * width ** 2))


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (scale + shift) from a condition."""

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feature_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return h * (1.0 + scale) + shift


def sinusoidal_embedding(t: Tensor, dim: int = 32) -> Tensor:
    """Sinusoidal time embedding.

    Args:
        t: Scalar time values ``[B]`` or ``[B, 1]``.
        dim: Embedding dimension (must be even).

    Returns:
        ``[B, dim]``.
    """
    import math

    t = t.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def scatter_mean(src: Tensor, idx: Tensor, n: int) -> Tensor:
    """Scatter-mean aggregation.

    Args:
        src: Source features ``[E, D]``.
        idx: Target indices ``[E]``.
        n: Number of target nodes.

    Returns:
        ``[n, D]``.
    """
    out = torch.zeros(n, src.shape[-1], device=src.device, dtype=src.dtype)
    count = torch.zeros(n, 1, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, idx.unsqueeze(-1).expand_as(src), src)
    count.scatter_add_(0, idx.unsqueeze(-1), torch.ones(idx.shape[0], 1, device=src.device, dtype=src.dtype))
    return out / count.clamp_min(1.0)


def build_radius_graph(
    pos: Tensor,
    r: float,
    batch: Tensor | None = None,
    max_num_neighbors: int = 32,
) -> Tensor:
    """Build self-radius graph via cdist (pure-torch, no torch_cluster).

    Returns:
        edge_index ``[2, E]``.
    """
    dists = torch.cdist(pos, pos)
    mask = (dists <= r) & (dists > 0)  # exclude self-loops

    if batch is not None:
        mask = mask & (batch.unsqueeze(1) == batch.unsqueeze(0))

    dists_inf = dists.masked_fill(~mask, float("inf"))
    k = min(max_num_neighbors, dists_inf.shape[1])
    _, topk_idx = dists_inf.topk(k, dim=1, largest=False)

    n = pos.shape[0]
    row = torch.arange(n, device=pos.device).unsqueeze(1).expand(-1, k)
    valid = torch.gather(dists_inf, 1, topk_idx) < float("inf")

    src = topk_idx[valid]
    dst = row[valid]
    return torch.stack([src, dst], dim=0)


__all__ = [
    "GatedGCNLayer", "FiLM", "build_radius_graph",
    "rbf_encode", "scatter_mean", "sinusoidal_embedding",
]
