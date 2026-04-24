"""Per-pose confidence head for FlowFrag.

Scores a single pose (no cross-pose dependencies). Takes per-atom
SO(3)-invariant features from the main model's final hidden at t=1 plus
optional per-pose scalar stats, and predicts:

  atom     :  per-atom displacement (log1p Å) + pLDDT (P(|disp| < 2 Å))
  fragment :  per-fragment RMSD (log1p Å)   + P(frag_rmsd > 2 Å)
  pose     :  pose RMSD (log1p Å)           + P(pose_rmsd < 2 Å)

Fragment and pose heads consume pooled per-atom trunk features (pose uses
mean + max pool), so no separate fragment-hidden feature extraction is
required from the main model.

All heads operate on ONE pose at a time conceptually; multiple poses in a
batch are just concatenated along the atom axis with CSR pointers marking
pose boundaries.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .layers import scatter_mean, segment_max_csr, segment_mean_csr


# ---------------------------------------------------------------------------
# Confidence head
# ---------------------------------------------------------------------------
class ConfidenceHead(nn.Module):
    """Multi-head per-pose confidence predictor."""

    def __init__(
        self,
        scalar_dim: int = 512,
        norms_dim: int = 192,
        pose_stats_dim: int = 6,
        hidden: int = 256,
        trunk_depth: int = 3,
        head_depth: int = 2,
        dropout: float = 0.2,
        pool_mode: str = "mean_max",
        n_pool_queries: int = 4,
        n_pool_heads: int = 4,
    ) -> None:
        super().__init__()
        self.scalar_dim = scalar_dim
        self.norms_dim = norms_dim
        self.pose_stats_dim = pose_stats_dim
        self.hidden = hidden
        self.pool_mode = pool_mode
        self.n_pool_queries = n_pool_queries

        # --- Shared trunk ------------------------------------------------
        in_dim = scalar_dim + norms_dim
        trunk: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(trunk_depth - 1):
            trunk += [nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*trunk)

        def _head(in_d: int, out_d: int = 2) -> nn.Sequential:
            layers: list[nn.Module] = []
            d = in_d
            for _ in range(head_depth - 1):
                layers += [nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(dropout)]
                d = hidden
            layers.append(nn.Linear(d, out_d))
            return nn.Sequential(*layers)

        # --- Atom head: (log1p_disp, plddt_logit) ------------------------
        self.atom_head = _head(hidden, out_d=2)

        # --- Fragment head: pool atom trunk → (log1p_frag_rmsd, bad_logit)
        self.frag_head = _head(hidden, out_d=2)

        # --- Attention pool for pose head (PMA: learnable queries + cross-attn) ---
        assert pool_mode in ("mean_max", "attention", "both"), (
            f"pool_mode must be one of {{mean_max, attention, both}}, got {pool_mode!r}"
        )
        if pool_mode in ("attention", "both"):
            self.pool_queries = nn.Parameter(torch.randn(n_pool_queries, hidden) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                hidden, num_heads=n_pool_heads, batch_first=True, dropout=dropout,
            )
        pool_dim = {
            "mean_max": 2 * hidden,
            "attention": n_pool_queries * hidden,
            "both": (2 + n_pool_queries) * hidden,
        }[pool_mode]

        # --- Pose head -------------------------------------------------------
        self.pose_head = _head(pool_dim + pose_stats_dim, out_d=2)

    # ---------------------------------------------------------------------
    def _attention_pool(self, h_atom: Tensor, atom_pose_ptr: Tensor) -> Tensor:
        """PMA over variable-length atom sequences per pose.  Returns [P, K·D]."""
        sizes = atom_pose_ptr[1:] - atom_pose_ptr[:-1]
        P = int(sizes.shape[0])
        max_s = int(sizes.max().item())

        # Vectorized padding via scatter
        seg_id = torch.repeat_interleave(torch.arange(P, device=h_atom.device), sizes)
        range_all = torch.arange(h_atom.shape[0], device=h_atom.device)
        pos_in_pose = range_all - atom_pose_ptr[:-1][seg_id]

        padded = torch.zeros(
            P, max_s, h_atom.shape[-1], device=h_atom.device, dtype=h_atom.dtype,
        )
        padded[seg_id, pos_in_pose] = h_atom

        mask = (
            torch.arange(max_s, device=h_atom.device).unsqueeze(0)
            >= sizes.unsqueeze(-1)
        )  # True = pad (MultiheadAttention convention)

        q = self.pool_queries.unsqueeze(0).expand(P, -1, -1)   # [P, K, D]
        out, _ = self.pool_attn(q, padded, padded, key_padding_mask=mask)
        return out.reshape(P, -1)                              # [P, K*D]

    # ---------------------------------------------------------------------
    def forward(
        self,
        atom_scalar: Tensor,       # [N, scalar_dim]
        atom_norms: Tensor,        # [N, norms_dim]
        atom_pose_ptr: Tensor,     # [P + 1]  CSR for pose segmentation
        atom_frag_id: Tensor,      # [N]      GLOBAL fragment id per atom (0..F-1)
        pose_stats: Tensor | None = None,   # [P, pose_stats_dim] or None
    ) -> dict[str, Tensor]:
        atom_feats = torch.cat([atom_scalar, atom_norms], dim=-1)   # [N, D_in]
        h_atom = self.trunk(atom_feats)                             # [N, hidden]

        # --- Atom outputs -----------------------------------------------
        atom_out = self.atom_head(h_atom)
        atom_disp_log1p = atom_out[:, 0]
        atom_plddt_logit = atom_out[:, 1]
        atom_disp = torch.expm1(atom_disp_log1p.clamp(-2.0, 5.0)).clamp_min(0.0)

        # --- Fragment outputs -------------------------------------------
        n_frags = int(atom_frag_id.max().item()) + 1
        h_frag = scatter_mean(h_atom, atom_frag_id, n_frags)         # [F, hidden]
        frag_out = self.frag_head(h_frag)
        frag_rmsd_log1p = frag_out[:, 0]
        frag_bad_logit = frag_out[:, 1]
        frag_rmsd = torch.expm1(frag_rmsd_log1p.clamp(-2.0, 5.0)).clamp_min(0.0)

        # --- Pose outputs ------------------------------------------------
        P = atom_pose_ptr.shape[0] - 1
        pool_parts: list[Tensor] = []
        if self.pool_mode in ("mean_max", "both"):
            pool_parts.append(segment_mean_csr(h_atom, atom_pose_ptr))   # [P, hidden]
            pool_parts.append(segment_max_csr(h_atom, atom_pose_ptr))    # [P, hidden]
        if self.pool_mode in ("attention", "both"):
            pool_parts.append(self._attention_pool(h_atom, atom_pose_ptr))  # [P, K*D]

        if pose_stats is None:
            pose_stats = torch.zeros(P, self.pose_stats_dim,
                                     device=h_atom.device, dtype=h_atom.dtype)
        pose_feat = torch.cat(pool_parts + [pose_stats], dim=-1)
        pose_out = self.pose_head(pose_feat)
        pose_rmsd_log1p = pose_out[:, 0]
        pose_prob_logit = pose_out[:, 1]
        pose_rmsd = torch.expm1(pose_rmsd_log1p.clamp(-2.0, 5.0)).clamp_min(0.0)

        # Also a "derived" pose RMSD from per-atom predictions (ensemble view)
        pose_rmsd_from_atoms = segment_mean_csr(
            atom_disp.pow(2).unsqueeze(-1), atom_pose_ptr
        ).squeeze(-1).clamp_min(0.0).sqrt()

        return {
            "atom_disp_log1p": atom_disp_log1p,
            "atom_plddt_logit": atom_plddt_logit,
            "atom_disp": atom_disp,
            "frag_rmsd_log1p": frag_rmsd_log1p,
            "frag_bad_logit": frag_bad_logit,
            "frag_rmsd": frag_rmsd,
            "pose_rmsd_log1p": pose_rmsd_log1p,
            "pose_prob_logit": pose_prob_logit,
            "pose_rmsd": pose_rmsd,
            "pose_rmsd_from_atoms": pose_rmsd_from_atoms,
        }


__all__ = ["ConfidenceHead"]
