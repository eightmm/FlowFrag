"""Unified equivariant model: single GNN over all node/edge types.

All nodes (ligand atoms, dummy atoms, fragments, protein atoms, CA virtuals)
live in the same irreps space and exchange messages through edge-type-specific
tensor product convolutions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

import cuequivariance as cue
import cuequivariance_torch as cuet

from .equivariant import EquivariantTPConv, EquivariantAdaLN, ScalarActivation, EquivariantDropout
from .layers import rbf_encode, sinusoidal_embedding, scatter_mean
from ..geometry.se3 import quaternion_to_matrix, quaternion_to_axis_angle


# ---------------------------------------------------------------------------
# Node type constants (must match build_static_complex_graph)
# ---------------------------------------------------------------------------
NTYPE_LIG_ATOM = 0
NTYPE_LIG_DUMMY = 1
NTYPE_FRAGMENT = 2
NTYPE_PROT_ATOM = 3
NTYPE_PROT_CA = 4
NUM_NODE_TYPES = 5

# Edge type names (for documentation; actual dispatch uses integer indices)
# 0=lig_bond, 1=lig_tri, 2=lig_cut, 3=lig_dummy_bond, 4=frag_adj,
# 5=prot_bond, 6=prot_lig_contact, 7=prot_ca_connect, 8=ca_ca
NUM_EDGE_TYPES = 9


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
        n = graph["num_nodes"]
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
    ) -> None:
        super().__init__()
        self.scalar_dim = scalar_dim
        self.n_rbf = n_rbf

        node_irreps = cue.Irreps("O3", f"{scalar_dim}x0e + {vec_dim}x1o + {vec_dim}x1e + {l2_dim}x2e")

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
        e_type = self.edge_type_emb(edge_type)
        e_bond = torch.cat([
            self.bond_type_emb(edge_bond_type),
            self.bond_conj_emb(edge_bond_conjugated),
            self.bond_ring_emb(edge_bond_in_ring),
            self.bond_stereo_emb(edge_bond_stereo),
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
        num_layers: int = 4,
        n_rbf: int = 16,
        t_emb_dim: int = 128,
        sh_lmax: int = 2,
        max_frag_size: int = 30,
        dropout: float = 0.0,
        # Compat kwargs for old config keys
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_vec_dim = hidden_vec_dim
        self.l2_dim = l2_dim
        self.n_rbf = n_rbf

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
            )
            for _ in range(num_layers)
        ])

        # Output head (operates on fragment nodes only)
        node_irreps = cue.Irreps("O3", f"{hidden_dim}x0e + {hidden_vec_dim}x1o + {hidden_vec_dim}x1e + {l2_dim}x2e")
        head_irreps = cue.Irreps("O3", f"{hidden_dim // 2}x0e + {hidden_vec_dim}x1o + {hidden_vec_dim}x1e + {l2_dim}x2e")

        self.head_pre = cuet.Linear(node_irreps, head_irreps, layout=cue.mul_ir)
        self.head_act = ScalarActivation(head_irreps)
        self.v_linear = cuet.Linear(head_irreps, cue.Irreps("O3", "1x1o"), layout=cue.mul_ir)
        self.omega_linear = cuet.Linear(head_irreps, cue.Irreps("O3", "1x1e"), layout=cue.mul_ir)

    def forward(self, batch: HeteroData) -> dict[str, Tensor]:
        graph = batch["graph"]  # unified graph from graph.pt
        device = graph["node_coords"].device

        n_nodes = graph["num_nodes"]
        coords = graph["node_coords"]
        node_type = graph["node_type"]
        edge_index = graph["edge_index"]
        edge_type = graph["edge_type"]

        # Edge features (bond-specific, 0/default for non-bond edges)
        edge_bond_type = graph.get("edge_bond_type", torch.zeros(edge_index.shape[1], dtype=torch.long, device=device))
        edge_bond_conjugated = graph.get("edge_bond_conjugated", torch.full((edge_index.shape[1],), 2, dtype=torch.long, device=device))
        edge_bond_in_ring = graph.get("edge_bond_in_ring", torch.full((edge_index.shape[1],), 2, dtype=torch.long, device=device))
        edge_bond_stereo = graph.get("edge_bond_stereo", torch.full((edge_index.shape[1],), 4, dtype=torch.long, device=device))
        edge_ref_dist = graph.get("edge_ref_dist", torch.zeros(edge_index.shape[1], dtype=torch.float32, device=device))

        frag_slice = graph["lig_frag_slice"]  # [start, end]
        frag_mask = node_type == NTYPE_FRAGMENT
        n_frag = int(frag_mask.sum())

        # Fragment data from batch
        T_frag = batch["fragment"].T_frag
        q_frag = batch["fragment"].q_frag
        frag_sizes = batch["fragment"].size
        t_val = batch.t

        # --- Time embedding ---
        t_sin = sinusoidal_embedding(t_val.view(-1), dim=32)
        t_emb = self.t_emb_mlp(t_sin)
        # Broadcast to all nodes
        t_emb_nodes = t_emb.expand(n_nodes, -1)

        # --- Node scalar embedding ---
        h_scalar = self.node_emb(graph)  # [N, hidden_dim]

        # Fragment-specific init: add size + time
        frag_idx = frag_mask.nonzero(as_tuple=True)[0]
        size_clamped = frag_sizes.clamp(max=30).long()
        size_feat = self.frag_size_emb(size_clamped)
        h_frag_init = self.frag_init_mlp(
            torch.cat([h_scalar[frag_idx], size_feat, t_emb.expand(n_frag, -1)], dim=-1)
        )
        h_scalar = h_scalar.clone()
        h_scalar[frag_idx] = h_frag_init

        # --- Build equivariant state: 0e + 1o + 1e + 2e ---
        vec_dim = self.hidden_vec_dim
        l2 = self.l2_dim

        # 1o init: gated displacement from center
        pocket_center = coords.mean(0, keepdim=True)
        r = coords - pocket_center
        gate = self.vec_gate(h_scalar)  # [N, vec_dim]
        h_1o = (gate.unsqueeze(-1) * r.unsqueeze(1)).reshape(n_nodes, vec_dim * 3)

        # Fragment rotation injection (R_t columns → 1o)
        R_frag = quaternion_to_matrix(q_frag)
        R_cols = R_frag.transpose(-1, -2)
        rot_gate = self.frag_rot_gate(h_scalar[frag_idx])
        rot_gate = rot_gate.view(n_frag, 3, vec_dim)
        rot_vecs = (rot_gate.unsqueeze(-1) * R_cols.unsqueeze(2)).sum(dim=1)
        h_1o_clone = h_1o.clone()
        h_1o_clone[frag_idx] = h_1o_clone[frag_idx] + rot_vecs.reshape(n_frag, vec_dim * 3)

        # 1e + 2e zero init
        h_1e = torch.zeros(n_nodes, vec_dim * 3, device=device, dtype=coords.dtype)
        h_2e = torch.zeros(n_nodes, l2 * 5, device=device, dtype=coords.dtype)

        h = torch.cat([h_scalar, h_1o_clone, h_1e, h_2e], dim=-1)

        # --- Update coordinates for current poses ---
        # Fragment nodes use T_frag, ligand atoms use pos_t
        # For now, use static coords from graph (crystal + pocket-centered)
        # This will be updated when we integrate flow matching state

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
            layer_rot_gate = self.frag_rot_gate(h[frag_idx, :S])
            layer_rot_gate = layer_rot_gate.view(n_frag, 3, vec_dim)
            layer_rot_vecs = (layer_rot_gate.unsqueeze(-1) * R_cols.unsqueeze(2)).sum(dim=1)
            h_new_1o = h[:, _1o_start:_1o_end].clone()
            h_new_1o[frag_idx] = h_new_1o[frag_idx] + layer_rot_vecs.reshape(n_frag, vec_dim * 3)
            h = torch.cat([h[:, :_1o_start], h_new_1o, h[:, _1o_end:]], dim=-1)

        # --- Output: extract fragment nodes ---
        h_frag = h[frag_idx]
        h_head = self.head_pre(h_frag)
        h_head = self.head_act(h_head)
        v_pred = self.v_linear(h_head)
        omega_pred = self.omega_linear(h_head)

        # Mask single-atom fragments
        omega_pred = omega_pred * (frag_sizes > 1).float().unsqueeze(-1)

        return {"v_pred": v_pred, "omega_pred": omega_pred}


__all__ = ["UnifiedFlowFrag"]
