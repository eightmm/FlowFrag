"""Unified equivariant model: single GNN over all node/edge types.

All nodes (ligand atoms, dummy atoms, fragments, protein atoms, CA virtuals)
live in the same irreps space and exchange messages through edge-type-specific
tensor product convolutions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

import cuequivariance as cue
import cuequivariance_torch as cuet

from .equivariant import EquivariantTPConv, EquivariantAdaLN, ScalarActivation, EquivariantDropout
from .layers import rbf_encode, sinusoidal_embedding, scatter_mean
from ..geometry.se3 import quaternion_to_matrix


def _build_node_irreps(scalar_dim: int, vec_dim: int, l2_dim: int, l2o_dim: int = 0) -> cue.Irreps:
    """Build node irreps string, omitting 2e/2o if dims are 0."""
    parts = [f"{scalar_dim}x0e", f"{vec_dim}x1o", f"{vec_dim}x1e"]
    if l2_dim > 0:
        parts.append(f"{l2_dim}x2e")
    if l2o_dim > 0:
        parts.append(f"{l2o_dim}x2o")
    return cue.Irreps("O3", " + ".join(parts))


# ---------------------------------------------------------------------------
# Newton-Euler aggregation: atom forces → fragment (v, omega)
# ---------------------------------------------------------------------------

def newton_euler_aggregate(
    f_atom: Tensor,
    atom_pos: Tensor,
    T_frag: Tensor,
    frag_id: Tensor,
    n_frag: int,
    frag_sizes: Tensor,
) -> tuple[Tensor, Tensor]:
    """Aggregate per-atom forces to fragment translation/angular velocities.

    Args:
        f_atom: Per-atom force vectors [N_atom, 3].
        atom_pos: Atom positions at time t [N_atom, 3].
        T_frag: Fragment centers [N_frag, 3].
        frag_id: Fragment assignment per atom [N_atom].
        n_frag: Number of fragments.
        frag_sizes: Fragment sizes [N_frag].

    Returns:
        v_frag: Translation velocity [N_frag, 3].
        omega_frag: Angular velocity [N_frag, 3].
    """
    # Translation: mean force per fragment
    v_frag = scatter_mean(f_atom, frag_id, n_frag)

    # Lever arm: atom position relative to fragment center
    r_arm = atom_pos - T_frag[frag_id]  # [N_atom, 3]

    # Torque: sum of cross(r, f) per fragment
    torque_per_atom = torch.cross(r_arm, f_atom, dim=-1)  # [N_atom, 3]
    torque = torch.zeros(n_frag, 3, device=f_atom.device, dtype=f_atom.dtype)
    torque.scatter_add_(0, frag_id.unsqueeze(-1).expand_as(torque_per_atom), torque_per_atom)

    # Inertia tensor per fragment: I = sum(|r|^2 * I_3 - r ⊗ r)
    r2 = (r_arm * r_arm).sum(-1, keepdim=True)  # [N_atom, 1]
    # Diagonal: sum(|r|^2) per fragment
    I_diag = torch.zeros(n_frag, 1, device=f_atom.device, dtype=f_atom.dtype)
    I_diag.scatter_add_(0, frag_id.unsqueeze(-1), r2)
    I_diag = I_diag.expand(-1, 3)  # [N_frag, 3]

    # Off-diagonal: sum(r_i * r_j) per fragment — build full 3x3
    rr = r_arm.unsqueeze(-1) * r_arm.unsqueeze(-2)  # [N_atom, 3, 3]
    I_off = torch.zeros(n_frag, 3, 3, device=f_atom.device, dtype=f_atom.dtype)
    I_off.scatter_add_(0, frag_id.view(-1, 1, 1).expand_as(rr), rr)

    I_tensor = torch.diag_embed(I_diag) - I_off  # [N_frag, 3, 3]

    # Regularize for single-atom fragments (degenerate inertia)
    eps = 1e-4
    I_tensor = I_tensor + eps * torch.eye(3, device=f_atom.device, dtype=f_atom.dtype)

    # Solve I @ omega = torque
    omega_frag = torch.linalg.solve(I_tensor, torque.unsqueeze(-1)).squeeze(-1)

    # Zero out single-atom fragments
    single_mask = (frag_sizes <= 1).unsqueeze(-1)
    omega_frag = omega_frag.masked_fill(single_mask, 0.0)

    return v_frag, omega_frag


# ---------------------------------------------------------------------------
# Node type constants (must match build_static_complex_graph)
# ---------------------------------------------------------------------------
NTYPE_LIG_ATOM = 0
NTYPE_LIG_DUMMY = 1
NTYPE_FRAGMENT = 2
NTYPE_PROT_ATOM = 3
NTYPE_PROT_CA = 4
NUM_NODE_TYPES = 5

# Edge type names (from src/preprocess/graph.py EDGE_TYPES)
# 0=ligand_bond, 1=ligand_tri, 2=ligand_cut, 3=ligand_atom_frag,
# 4=ligand_frag_frag, 5=protein_bond, 6=protein_atom_ca,
# 7=protein_ca_ca, 8=protein_ca_frag, 9=dynamic_contact (runtime)
ETYPE_DYNAMIC_CONTACT = 9
NUM_EDGE_TYPES = 10


# ---------------------------------------------------------------------------
# Unified node embedding
# ---------------------------------------------------------------------------

class UnifiedNodeEmbedding(nn.Module):
    """Embed all node types into a shared scalar space."""

    NUM_ELEMENTS = 13
    NUM_HYBRIDIZATIONS = 6
    NUM_AMINO_ACIDS = 21

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        # Atom-level features (shared across lig/prot atoms)
        self.elem_emb = nn.Embedding(self.NUM_ELEMENTS, 32)
        self.charge_proj = nn.Linear(1, 8)
        self.aromatic_emb = nn.Embedding(2, 8)
        self.hybrid_emb = nn.Embedding(self.NUM_HYBRIDIZATIONS, 16)
        self.ring_emb = nn.Embedding(2, 8)
        # 32+8+8+16+8 = 72

        # Node type embedding
        self.type_emb = nn.Embedding(NUM_NODE_TYPES, 16)

        # Amino acid (for protein nodes)
        self.aa_emb = nn.Embedding(self.NUM_AMINO_ACIDS, 16)

        # Boolean pharmacophore features
        self.bool_proj = nn.Linear(8, 16)  # donor/acceptor/positive/negative/hydrophobe/halogen/backbone/dummy

        # Projection: 72 + 16 + 16 + 16 = 120 -> hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(120, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, graph: dict[str, Tensor]) -> Tensor:
        h = self.elem_emb(graph["node_element"].clamp(max=self.NUM_ELEMENTS - 1))
        h = h + nn.functional.pad(self.charge_proj(graph["node_charge"].float().unsqueeze(-1)), (0, 24))
        h = h + nn.functional.pad(self.aromatic_emb(graph["node_aromatic"].long()), (0, 24))
        h = h + nn.functional.pad(self.hybrid_emb(graph["node_hybridization"].clamp(max=self.NUM_HYBRIDIZATIONS - 1).long()), (0, 16))
        h = h + nn.functional.pad(self.ring_emb(graph["node_in_ring"].long()), (0, 24))

        # Node type
        h_type = self.type_emb(graph["node_type"])
        # Amino acid
        h_aa = self.aa_emb(graph["node_amino_acid"].clamp(max=self.NUM_AMINO_ACIDS - 1))

        # Boolean features
        bools = torch.stack([
            graph["node_is_donor"].float(),
            graph["node_is_acceptor"].float(),
            graph["node_is_positive"].float(),
            graph["node_is_negative"].float(),
            graph["node_is_hydrophobe"].float(),
            graph["node_is_halogen"].float(),
            graph["node_is_backbone"].float(),
            graph["node_is_dummy"].float(),
        ], dim=-1)
        h_bool = self.bool_proj(bools)

        h_cat = torch.cat([h[:, :32], h_type, h_aa, h_bool,
                           self.charge_proj(graph["node_charge"].float().unsqueeze(-1)),
                           self.aromatic_emb(graph["node_aromatic"].long()),
                           self.hybrid_emb(graph["node_hybridization"].clamp(max=self.NUM_HYBRIDIZATIONS - 1).long()),
                           self.ring_emb(graph["node_in_ring"].long()),
                           ], dim=-1)  # 32+16+16+16+8+8+16+8 = 120

        return self.proj(h_cat)


# ---------------------------------------------------------------------------
# Unified interaction layer
# ---------------------------------------------------------------------------

class UnifiedInteractionLayer(nn.Module):
    """Single equivariant interaction layer over the unified graph.

    Uses ONE TP convolution for all edge types.  Edge-type specialization
    comes from the edge scalar features: edge_type embedding + bond-specific
    features (type, conjugation, ring, stereo) + RBF distance + node scalars.
    """

    NUM_BOND_TYPES = 5
    NUM_BOND_STEREO = 4

    def __init__(
        self,
        scalar_dim: int,
        vec_dim: int,
        l2_dim: int,
        t_emb_dim: int,
        n_edge_types: int = NUM_EDGE_TYPES,
        n_rbf: int = 16,
        sh_lmax: int = 2,
        dropout: float = 0.0,
        l2o_dim: int = 0,
    ) -> None:
        super().__init__()
        self.scalar_dim = scalar_dim
        self.n_rbf = n_rbf

        node_irreps = _build_node_irreps(scalar_dim, vec_dim, l2_dim, l2o_dim)

        # Edge type embedding
        self.edge_type_emb = nn.Embedding(n_edge_types, 16)

        # Bond feature embeddings (only active for bond-type edges)
        self.bond_type_emb = nn.Embedding(self.NUM_BOND_TYPES + 1, 8)  # +1 for "no bond"
        self.bond_conj_emb = nn.Embedding(3, 4)  # 0=no, 1=yes, 2=N/A
        self.bond_ring_emb = nn.Embedding(3, 4)  # 0=no, 1=yes, 2=N/A
        self.bond_stereo_emb = nn.Embedding(self.NUM_BOND_STEREO + 1, 4)  # +1 for "N/A"

        # Triangulation ref_dist feature
        self.ref_dist_proj = nn.Linear(3, 8)  # rbf(|delta_d|) + delta_d + has_ref

        # Edge scalar dim:
        #   rbf(16) + edge_type(16) + bond(8+4+4+4) + ref_dist(8)
        #   + src_scalar + dst_scalar + t_emb
        edge_scalar_dim = n_rbf + 16 + 20 + 8 + scalar_dim + scalar_dim + t_emb_dim

        # Single unified TP conv
        self.conv = EquivariantTPConv(
            node_irreps, node_irreps,
            sh_lmax=sh_lmax,
            edge_scalar_dim=edge_scalar_dim,
            n_rbf=n_rbf,
        )

        # Post-aggregation
        self.proj = cuet.Linear(node_irreps, node_irreps, layout=cue.mul_ir)
        self.act = ScalarActivation(node_irreps)
        self.dropout = EquivariantDropout(node_irreps, p=dropout)
        self.ada_ln = EquivariantAdaLN(node_irreps, t_emb_dim)

    def forward(
        self,
        h: Tensor,
        coords: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_bond_type: Tensor,
        edge_bond_conjugated: Tensor,
        edge_bond_in_ring: Tensor,
        edge_bond_stereo: Tensor,
        edge_ref_dist: Tensor,
        t_emb: Tensor,
    ) -> Tensor:
        n = h.shape[0]
        S = self.scalar_dim
        h_s = h[:, :S]

        src = edge_index[0]
        dst = edge_index[1]

        diff = coords[dst] - coords[src]
        dist = torch.linalg.vector_norm(diff, dim=-1)

        # Build edge scalar features
        # Preprocessing convention: -1 = "not a bond edge" sentinel.
        # Model convention: index 0 = "no bond / N/A", so shift -1 → 0.
        # For conjugated/in_ring (stored as bool), non-bond edges get 2 (N/A).
        e_type = self.edge_type_emb(edge_type.long())
        is_bond_edge = (edge_bond_type >= 0)  # True for actual bond edges
        conj_idx = edge_bond_conjugated.long()  # 0=no, 1=yes
        conj_idx = torch.where(is_bond_edge, conj_idx, torch.full_like(conj_idx, 2))  # non-bond → 2 (N/A)
        ring_idx = edge_bond_in_ring.long()
        ring_idx = torch.where(is_bond_edge, ring_idx, torch.full_like(ring_idx, 2))
        e_bond = torch.cat([
            self.bond_type_emb(edge_bond_type.long() + 1),    # -1→0(no bond), 0→1, ...
            self.bond_conj_emb(conj_idx),                      # 0=no, 1=yes, 2=N/A
            self.bond_ring_emb(ring_idx),                      # 0=no, 1=yes, 2=N/A
            self.bond_stereo_emb(edge_bond_stereo.long() + 1), # -1→0(N/A), 0→1, ...
        ], dim=-1)  # [E, 20]

        # Triangulation ref_dist: delta_d = |current_dist - ref_dist|
        has_ref = (edge_ref_dist > 0).float()
        delta_d = dist - edge_ref_dist
        e_ref = self.ref_dist_proj(torch.stack([
            delta_d.abs(), delta_d, has_ref,
        ], dim=-1))  # [E, 8]

        edge_scalars = torch.cat([
            rbf_encode(dist, self.n_rbf),
            e_type, e_bond, e_ref,
            h_s[src], h_s[dst], t_emb[dst],
        ], dim=-1)

        msg = self.conv(h, diff, edge_scalars, src, dst, n)

        # Update
        update = self.proj(msg)
        update = self.act(update)
        update = self.dropout(update)
        h = h + update
        h = self.ada_ln(h, t_emb)

        return h


# ---------------------------------------------------------------------------
# Unified FlowFrag model
# ---------------------------------------------------------------------------

class UnifiedFlowFrag(nn.Module):
    """Unified equivariant model for fragment-based flow matching.

    All node types share the same irreps space and exchange messages through
    edge-type-specific tensor product convolutions.

    Args:
        hidden_dim: Scalar dimension for node embedding.
        hidden_vec_dim: Vector (1o, 1e) channel count.
        l2_dim: Quadrupole (2e) channel count.
        num_layers: Number of interaction layers.
        n_rbf: RBF basis count.
        t_emb_dim: Time embedding dimension.
        sh_lmax: Maximum spherical harmonics degree.
        max_frag_size: Maximum fragment size for size embedding.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        hidden_vec_dim: int = 32,
        l2_dim: int = 16,
        l2o_dim: int = 0,
        num_layers: int = 4,
        n_rbf: int = 16,
        t_emb_dim: int = 128,
        sh_lmax: int = 2,
        max_frag_size: int = 30,
        dropout: float = 0.0,
        omega_mode: str = "direct",
        contact_cutoff: float = 0.0,
        prune_edge_types: list[int] | None = None,
        rt_injection: str = "per_layer",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_vec_dim = hidden_vec_dim
        self.l2_dim = l2_dim
        self.l2o_dim = l2o_dim
        self.n_rbf = n_rbf
        self.omega_mode = omega_mode
        self.contact_cutoff = contact_cutoff
        self.prune_edge_types = set(prune_edge_types) if prune_edge_types else set()
        self.rt_injection = rt_injection

        # Node embedding (scalar)
        self.node_emb = UnifiedNodeEmbedding(hidden_dim)

        # Time embedding
        self.t_emb_mlp = nn.Sequential(
            nn.Linear(32, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        # Fragment-specific: size embedding
        self.frag_size_emb = nn.Embedding(max_frag_size + 1, 16)
        self.frag_init_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 16 + t_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Vector init gates
        self.vec_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_vec_dim),
            nn.Tanh(),
        )

        # Fragment rotation injection gate (R_t columns → 1o)
        self.frag_rot_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_vec_dim * 3),
            nn.Tanh(),
        )

        # Interaction layers
        self.layers = nn.ModuleList([
            UnifiedInteractionLayer(
                scalar_dim=hidden_dim,
                vec_dim=hidden_vec_dim,
                l2_dim=l2_dim,
                t_emb_dim=t_emb_dim,
                n_rbf=n_rbf,
                sh_lmax=sh_lmax,
                dropout=dropout,
                l2o_dim=l2o_dim,
            )
            for _ in range(num_layers)
        ])

        # Output heads
        node_irreps = _build_node_irreps(hidden_dim, hidden_vec_dim, l2_dim, l2o_dim)
        head_irreps = _build_node_irreps(hidden_dim // 2, hidden_vec_dim, l2_dim, l2o_dim)

        if omega_mode in ("newton_euler", "hybrid"):
            # Atom-level force head: predict f_atom (1o) from ligand atom nodes
            self.atom_head_pre = cuet.Linear(node_irreps, head_irreps, layout=cue.mul_ir)
            self.atom_head_act = ScalarActivation(head_irreps)
            self.f_atom_linear = cuet.Linear(head_irreps, cue.Irreps("O3", "1x1o"), layout=cue.mul_ir)

        if omega_mode in ("direct", "hybrid"):
            # Direct fragment-level head
            self.head_pre = cuet.Linear(node_irreps, head_irreps, layout=cue.mul_ir)
            self.head_act = ScalarActivation(head_irreps)
            self.v_linear = cuet.Linear(head_irreps, cue.Irreps("O3", "1x1o"), layout=cue.mul_ir)
            if omega_mode == "direct":
                self.omega_linear = cuet.Linear(head_irreps, cue.Irreps("O3", "1x1e"), layout=cue.mul_ir)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Forward pass on a collated batch from UnifiedDataset.

        The batch dict contains all graph.pt tensors (node_coords already
        updated with flow matching poses), plus T_frag, q_frag, frag_sizes,
        t, batch (node→sample), frag_batch (frag→sample).
        """
        device = batch["node_coords"].device
        n_nodes = batch["node_coords"].shape[0]
        coords = batch["node_coords"]
        node_type = batch["node_type"]
        edge_index = batch["edge_index"]
        edge_type = batch["edge_type"]

        # Edge features (bond-specific, 0/default for non-bond edges)
        n_edges = edge_index.shape[1]
        edge_bond_type = batch.get("edge_bond_type", torch.zeros(n_edges, dtype=torch.long, device=device))
        edge_bond_conjugated = batch.get("edge_bond_conjugated", torch.full((n_edges,), 2, dtype=torch.long, device=device))
        edge_bond_in_ring = batch.get("edge_bond_in_ring", torch.full((n_edges,), 2, dtype=torch.long, device=device))
        edge_bond_stereo = batch.get("edge_bond_stereo", torch.full((n_edges,), 4, dtype=torch.long, device=device))
        edge_ref_dist = batch.get("edge_ref_dist", torch.zeros(n_edges, dtype=torch.float32, device=device))

        # Prune specified edge types
        if self.prune_edge_types:
            keep_mask = torch.ones(n_edges, dtype=torch.bool, device=device)
            for et in self.prune_edge_types:
                keep_mask &= (edge_type != et)
            edge_index = edge_index[:, keep_mask]
            edge_type = edge_type[keep_mask]
            edge_bond_type = edge_bond_type[keep_mask]
            edge_bond_conjugated = edge_bond_conjugated[keep_mask]
            edge_bond_in_ring = edge_bond_in_ring[keep_mask]
            edge_bond_stereo = edge_bond_stereo[keep_mask]
            edge_ref_dist = edge_ref_dist[keep_mask]
            n_edges = edge_index.shape[1]

        # Dynamic protein_atom ↔ ligand_atom contact edges (rebuilt per forward)
        if self.contact_cutoff > 0:
            prot_atom_mask = node_type == NTYPE_PROT_ATOM
            lig_atom_mask = node_type == NTYPE_LIG_ATOM
            prot_idx_all = prot_atom_mask.nonzero(as_tuple=True)[0]
            lig_idx_all = lig_atom_mask.nonzero(as_tuple=True)[0]

            if prot_idx_all.shape[0] > 0 and lig_idx_all.shape[0] > 0:
                dists = torch.cdist(coords[prot_idx_all], coords[lig_idx_all])
                contact_mask = dists <= self.contact_cutoff
                pi, li = contact_mask.nonzero(as_tuple=True)
                if pi.shape[0] > 0:
                    c_src = torch.cat([prot_idx_all[pi], lig_idx_all[li]])
                    c_dst = torch.cat([lig_idx_all[li], prot_idx_all[pi]])
                    n_contact = c_src.shape[0]

                    edge_index = torch.cat([edge_index, torch.stack([c_src, c_dst])], dim=1)
                    edge_type = torch.cat([edge_type, torch.full((n_contact,), ETYPE_DYNAMIC_CONTACT, dtype=edge_type.dtype, device=device)])
                    edge_bond_type = torch.cat([edge_bond_type, torch.full((n_contact,), -1, dtype=edge_bond_type.dtype, device=device)])
                    edge_bond_conjugated = torch.cat([edge_bond_conjugated, torch.zeros(n_contact, dtype=edge_bond_conjugated.dtype, device=device)])
                    edge_bond_in_ring = torch.cat([edge_bond_in_ring, torch.zeros(n_contact, dtype=edge_bond_in_ring.dtype, device=device)])
                    edge_bond_stereo = torch.cat([edge_bond_stereo, torch.full((n_contact,), -1, dtype=edge_bond_stereo.dtype, device=device)])
                    edge_ref_dist = torch.cat([edge_ref_dist, torch.zeros(n_contact, dtype=edge_ref_dist.dtype, device=device)])
                    n_edges = edge_index.shape[1]

        frag_mask = node_type == NTYPE_FRAGMENT
        frag_idx = frag_mask.nonzero(as_tuple=True)[0]
        n_frag = frag_idx.shape[0]

        # Fragment data
        q_frag = batch["q_frag"]
        frag_sizes = batch["frag_sizes"]

        # --- Time embedding ---
        # batch["t"] is [B, 1], broadcast to each node via batch["batch"]
        t_per_sample = batch["t"].view(-1)  # [B]
        t_sin = sinusoidal_embedding(t_per_sample, dim=32)  # [B, 32]
        t_emb = self.t_emb_mlp(t_sin)  # [B, t_emb_dim]
        node_batch_idx = batch["batch"]  # [N_nodes]
        t_emb_nodes = t_emb[node_batch_idx]  # [N_nodes, t_emb_dim]

        # Per-fragment time embedding
        frag_batch_idx = batch["frag_batch"]  # [N_frag]
        t_emb_frags = t_emb[frag_batch_idx]  # [N_frag, t_emb_dim]

        # --- Node scalar embedding ---
        h_scalar = self.node_emb(batch)  # [N, hidden_dim]

        # Fragment-specific init: add size + time
        size_clamped = frag_sizes.clamp(max=30).long()
        size_feat = self.frag_size_emb(size_clamped)
        h_frag_init = self.frag_init_mlp(
            torch.cat([h_scalar[frag_idx], size_feat, t_emb_frags], dim=-1)
        )
        h_scalar = h_scalar.clone()
        h_scalar[frag_idx] = h_frag_init

        # --- Build equivariant state: 0e + 1o + 1e [+ 2e] ---
        vec_dim = self.hidden_vec_dim
        l2 = self.l2_dim

        # 1o init: gated displacement from per-sample center
        n_samples = t_per_sample.shape[0]
        center_per_sample = scatter_mean(coords, node_batch_idx, n_samples)  # [B, 3]
        r = coords - center_per_sample[node_batch_idx]

        gate = self.vec_gate(h_scalar)  # [N, vec_dim]
        h_1o = (gate.unsqueeze(-1) * r.unsqueeze(1)).reshape(n_nodes, vec_dim * 3)

        # Fragment rotation injection (R_t columns → 1o)
        R_frag = quaternion_to_matrix(q_frag)
        R_cols = R_frag.transpose(-1, -2)  # [N_frag, 3, 3]
        h_1o_clone = h_1o.clone()
        if self.rt_injection != "off":
            rot_gate = self.frag_rot_gate(h_scalar[frag_idx])
            rot_gate = rot_gate.view(n_frag, 3, vec_dim)
            rot_vecs = (rot_gate.unsqueeze(-1) * R_cols.unsqueeze(2)).sum(dim=1)
            h_1o_clone[frag_idx] = h_1o_clone[frag_idx] + rot_vecs.reshape(n_frag, vec_dim * 3)

        # 1e [+ 2e] [+ 2o] zero init
        h_1e = torch.zeros(n_nodes, vec_dim * 3, device=device, dtype=coords.dtype)
        parts = [h_scalar, h_1o_clone, h_1e]
        if l2 > 0:
            parts.append(torch.zeros(n_nodes, l2 * 5, device=device, dtype=coords.dtype))
        l2o = self.l2o_dim
        if l2o > 0:
            parts.append(torch.zeros(n_nodes, l2o * 5, device=device, dtype=coords.dtype))
        h = torch.cat(parts, dim=-1)

        # --- Interaction layers with per-layer R_t re-injection ---
        S = self.hidden_dim
        _1o_start = S
        _1o_end = S + vec_dim * 3

        for layer in self.layers:
            h = layer(
                h, coords, edge_index, edge_type,
                edge_bond_type, edge_bond_conjugated, edge_bond_in_ring,
                edge_bond_stereo, edge_ref_dist, t_emb_nodes,
            )

            # Re-inject R_t columns into fragment 1o (non-inplace)
            if self.rt_injection == "per_layer":
                layer_rot_gate = self.frag_rot_gate(h[frag_idx, :S])
                layer_rot_gate = layer_rot_gate.view(n_frag, 3, vec_dim)
                layer_rot_vecs = (layer_rot_gate.unsqueeze(-1) * R_cols.unsqueeze(2)).sum(dim=1)
                h_new_1o = h[:, _1o_start:_1o_end].clone()
                h_new_1o[frag_idx] = h_new_1o[frag_idx] + layer_rot_vecs.reshape(n_frag, vec_dim * 3)
                h = torch.cat([h[:, :_1o_start], h_new_1o, h[:, _1o_end:]], dim=-1)

        # --- Output ---
        if self.omega_mode in ("newton_euler", "hybrid"):
            # Extract ligand atom nodes → atom forces
            atom_mask = node_type == NTYPE_LIG_ATOM
            atom_idx = atom_mask.nonzero(as_tuple=True)[0]
            h_atoms = h[atom_idx]

            h_atom_head = self.atom_head_pre(h_atoms)
            h_atom_head = self.atom_head_act(h_atom_head)
            f_atom = self.f_atom_linear(h_atom_head)  # [N_lig_atom, 3]

            # Newton-Euler: atom forces → fragment (v_ne, omega_ne)
            atom_pos = coords[atom_idx]
            T_frag = batch["T_frag"]
            atom_frag_id = batch["frag_id_for_atoms"]

            v_ne, omega_ne = newton_euler_aggregate(
                f_atom, atom_pos, T_frag, atom_frag_id, n_frag, frag_sizes,
            )

        if self.omega_mode == "newton_euler":
            v_pred, omega_pred = v_ne, omega_ne
        elif self.omega_mode == "hybrid":
            # Direct v from fragment nodes, N-E omega from atom forces
            h_frag = h[frag_idx]
            h_head = self.head_pre(h_frag)
            h_head = self.head_act(h_head)
            v_pred = self.v_linear(h_head)
            omega_pred = omega_ne
        else:
            # Direct: both from fragment nodes
            h_frag = h[frag_idx]
            h_head = self.head_pre(h_frag)
            h_head = self.head_act(h_head)
            v_pred = self.v_linear(h_head)
            omega_pred = self.omega_linear(h_head)

        # Mask single-atom fragments
        omega_pred = omega_pred * (frag_sizes > 1).float().unsqueeze(-1)

        return {"v_pred": v_pred, "omega_pred": omega_pred}


__all__ = ["UnifiedFlowFrag"]
