"""Reusable SE(3)-equivariant layers and utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

import cuequivariance as cue
import cuequivariance_torch as cuet

from .layers import scatter_mean


class EquivariantTPConv(nn.Module):
    """SE(3)-equivariant tensor product message passing.

    Given source features, edge indices, edge vectors, and scalar edge context,
    computes: SH(edge_vec) → weight_MLP(edge_scalars) → TP(src, SH, weights) → scatter_mean.

    Args:
        src_irreps: Source node irreps (scalar-only for external weights).
        dst_irreps: Output irreps after aggregation.
        sh_lmax: Maximum spherical harmonics degree.
        edge_scalar_dim: Dimension of scalar edge features fed to weight MLP.
        weight_mlp_hidden: Hidden dim for the weight MLP.
        n_rbf: Number of RBF basis functions for distance encoding.
    """

    def __init__(
        self,
        src_irreps: cue.Irreps,
        dst_irreps: cue.Irreps,
        sh_lmax: int,
        edge_scalar_dim: int,
        weight_mlp_hidden: int = 128,
        n_rbf: int = 16,
    ) -> None:
        super().__init__()
        self.n_rbf = n_rbf

        ls = list(range(sh_lmax + 1))
        sh_irreps = cue.Irreps("O3", " + ".join(f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in ls))

        self.sh = cuet.SphericalHarmonics(ls=ls, normalize=True)

        self.tp = cuet.FullyConnectedTensorProduct(
            src_irreps,
            sh_irreps,
            dst_irreps,
            layout_in1=cue.mul_ir,
            layout_in2=cue.mul_ir,
            layout_out=cue.mul_ir,
            internal_weights=False,
            shared_weights=False,
        )

        self.weight_mlp = nn.Sequential(
            nn.Linear(edge_scalar_dim, weight_mlp_hidden),
            nn.SiLU(),
            nn.Linear(weight_mlp_hidden, self.tp.weight_numel),
        )

    def forward(
        self,
        h_src: Tensor,
        edge_vec: Tensor,
        edge_scalars: Tensor,
        src_idx: Tensor,
        dst_idx: Tensor,
        n_dst: int,
    ) -> Tensor:
        """Forward pass."""
        edge_sh = self.sh(edge_vec)
        weights = self.weight_mlp(edge_scalars)
        msg = self.tp(h_src[src_idx], edge_sh, weights)
        return scatter_mean(msg, dst_idx, n_dst)


class ScalarActivation(nn.Module):
    """Applies activation only to scalar channels (0e)."""

    def __init__(self, irreps: cue.Irreps, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.irreps = irreps
        self.activation = activation or nn.SiLU()

        # Identify slices for 0e
        self.scalar_indices = []
        offset = 0
        for mul, ir in irreps:
            ir_dim = ir.dim
            if ir.l == 0 and ir.p == 1:  # 0e
                self.scalar_indices.append((offset, offset + mul * ir_dim))
            offset += mul * ir_dim

    def forward(self, x: Tensor) -> Tensor:
        if not self.scalar_indices:
            return x
        
        # In-place activation on scalar parts
        # (Though cuet tensors might be immutable in some contexts, 
        # standard torch operations work fine here)
        out = x.clone()
        for start, end in self.scalar_indices:
            out[..., start:end] = self.activation(x[..., start:end])
        return out


class EquivariantDropout(nn.Module):
    """Dropout for equivariant features.
    
    Scalars get element-wise dropout.
    Vectors get channel-wise dropout (dropping the entire vector).
    """

    def __init__(self, irreps: cue.Irreps, p: float = 0.1) -> None:
        super().__init__()
        self.irreps = irreps
        self.p = p
        
        self.slices = []
        offset = 0
        for mul, ir in irreps:
            ir_dim = ir.dim
            self.slices.append((offset, mul, ir_dim))
            offset += mul * ir_dim

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0:
            return x
        
        out = x.clone()
        for offset, mul, ir_dim in self.slices:
            if ir_dim == 1:
                # Scalar element-wise dropout
                mask = torch.rand(x.shape[:-1] + (mul,), device=x.device, dtype=x.dtype) > self.p
                out[..., offset : offset + mul] = x[..., offset : offset + mul] * mask.to(x.dtype) / (1 - self.p)
            else:
                # Vector channel-wise dropout
                mask = torch.rand(x.shape[:-1] + (mul, 1), device=x.device, dtype=x.dtype) > self.p
                mask = mask.expand(-1, -1, ir_dim).reshape(x.shape[:-1] + (mul * ir_dim,))
                out[..., offset : offset + mul * ir_dim] = x[..., offset : offset + mul * ir_dim] * mask.to(x.dtype) / (1 - self.p)
        return out


class EquivariantAdaLN(nn.Module):
    """Adaptive LayerNorm for equivariant features.

    Scalars: LayerNorm + (1 + scale) * h + shift from condition.
    Vectors: identity-preserving gating via (1 + tanh(gate)).

    Zero-init ensures identity at initialization for both scalar and vector paths.
    """

    def __init__(self, irreps: cue.Irreps, cond_dim: int) -> None:
        super().__init__()
        self.irreps = irreps

        self.scalar_slices: list[tuple[int, int]] = []
        self.vector_slices: list[tuple[int, int, int]] = []

        n_scalar = 0
        n_vector_channels = 0

        offset = 0
        for mul, ir in irreps:
            ir_dim = ir.dim
            if ir.l == 0 and ir.p == 1:  # 0e
                self.scalar_slices.append((offset, mul))
                n_scalar += mul
            else:
                self.vector_slices.append((offset, mul, ir_dim))
                n_vector_channels += mul
            offset += mul * ir_dim

        self.n_scalar = n_scalar
        self.n_vector_channels = n_vector_channels

        if n_scalar > 0:
            self.ln = nn.LayerNorm(n_scalar)
        else:
            self.ln = nn.Identity()

        # condition → [2*n_scalar (scale, shift), n_vector_channels (gates)]
        out_dim = 2 * n_scalar + n_vector_channels
        self.ada_mlp = nn.Linear(cond_dim, out_dim)

        # Zero-init: scale=0 → (1+0)*h + 0 = h, tanh(0)=0 → (1+0)*v = v
        nn.init.zeros_(self.ada_mlp.weight)
        nn.init.zeros_(self.ada_mlp.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        ada_out = self.ada_mlp(cond)
        out = x.clone()

        # 1. Scalars: LN → (1 + scale) * h + shift
        if self.scalar_slices:
            scalar_parts = [x[..., off : off + m] for off, m in self.scalar_slices]
            h_scalar = torch.cat(scalar_parts, dim=-1)
            h_scalar = self.ln(h_scalar)

            ns = self.n_scalar
            scale = ada_out[..., :ns]
            shift = ada_out[..., ns : 2 * ns]
            h_scalar = h_scalar * (1.0 + scale) + shift

            curr = 0
            for off, m in self.scalar_slices:
                out[..., off : off + m] = h_scalar[..., curr : curr + m]
                curr += m

        # 2. Vectors: bounded gate = (1 + 0.1 * tanh(g)), range [0.9, 1.1]
        # Prevents complete vector suppression that kills 1o channels.
        if self.vector_slices:
            gate_vals = ada_out[..., 2 * self.n_scalar :]
            gates = 1.0 + 0.1 * torch.tanh(gate_vals)
            curr = 0
            for off, m, ir_dim in self.vector_slices:
                g = gates[..., curr : curr + m].unsqueeze(-1)
                g = g.expand(*g.shape[:-1], ir_dim).reshape(x.shape[:-1] + (m * ir_dim,))
                out[..., off : off + m * ir_dim] = x[..., off : off + m * ir_dim] * g
                curr += m

        return out


__all__ = [
    "EquivariantTPConv", 
    "ScalarActivation", 
    "EquivariantDropout", 
    "EquivariantAdaLN"
]
