"""SE(3)-equivariant, time-conditioned multi-resolution docking head."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

import cuequivariance as cue
import cuequivariance_torch as cuet

from .equivariant import EquivariantTPConv, EquivariantAdaLN, ScalarActivation, EquivariantDropout
from .layers import rbf_encode, scatter_mean, sinusoidal_embedding
from ..geometry.se3 import quaternion_to_axis_angle, quaternion_to_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cross_radius_edges(
    src_pos: Tensor,
    dst_pos: Tensor,
    r: float,
    max_neighbors: int,
    src_batch: Tensor | None,
    dst_batch: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Build radius graph between two point sets via cdist.

    Pure-torch implementation (no torch_cluster dependency).

    Returns:
        (src_idx, dst_idx) with shapes ``[E]``.
    """
    dists = torch.cdist(dst_pos, src_pos)  # [N_dst, N_src]
    mask = dists <= r

    if src_batch is not None and dst_batch is not None:
        mask = mask & (dst_batch.unsqueeze(1) == src_batch.unsqueeze(0))

    # Limit to max_neighbors closest per dst node
    dists_inf = dists.masked_fill(~mask, float("inf"))
    k = min(max_neighbors, dists_inf.shape[1])
    _, topk_src = dists_inf.topk(k, dim=1, largest=False)

    n_dst = dst_pos.shape[0]
    dst_idx = torch.arange(n_dst, device=dst_pos.device).unsqueeze(1).expand(-1, k)

    valid = torch.gather(dists_inf, 1, topk_src) < float("inf")
    return topk_src[valid], dst_idx[valid]


def _build_cross_all_to_all_edges(
    src_pos: Tensor,
    dst_pos: Tensor,
    src_batch: Tensor | None,
    dst_batch: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Build a complete bipartite graph, optionally restricted within batches."""
    n_src = src_pos.shape[0]
    n_dst = dst_pos.shape[0]
    device = dst_pos.device
    empty = torch.zeros(0, dtype=torch.long, device=device)

    if n_src == 0 or n_dst == 0:
        return empty, empty

    src_idx = torch.arange(n_src, device=device).unsqueeze(0).expand(n_dst, -1)
    dst_idx = torch.arange(n_dst, device=device).unsqueeze(1).expand(-1, n_src)

    if src_batch is not None and dst_batch is not None:
        valid = dst_batch.unsqueeze(1) == src_batch.unsqueeze(0)
    else:
        valid = torch.ones(n_dst, n_src, dtype=torch.bool, device=device)

    return src_idx[valid], dst_idx[valid]


def _build_cross_topology_edges(
    src_pos: Tensor,
    dst_pos: Tensor,
    *,
    topology: str,
    r: float,
    max_neighbors: int,
    src_batch: Tensor | None,
    dst_batch: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Build bipartite edges using either local radius or full connectivity."""
    if topology == "radius":
        return _build_cross_radius_edges(
            src_pos,
            dst_pos,
            r=r,
            max_neighbors=max_neighbors,
            src_batch=src_batch,
            dst_batch=dst_batch,
        )
    if topology == "full":
        return _build_cross_all_to_all_edges(
            src_pos,
            dst_pos,
            src_batch=src_batch,
            dst_batch=dst_batch,
        )
    raise ValueError(f"Unknown edge topology '{topology}'. Expected 'radius' or 'full'.")


def _compute_fragment_inertia(
    local_pos: Tensor,
    fragment_id: Tensor,
    n_frag: int,
    eps: float = 1e-4,
) -> Tensor:
    """Compute per-fragment inertia tensor from rest-frame coordinates.

    Assumes unit mass per atom.  Returns regularised ``I + eps * I_3``.

    Args:
        local_pos: Atom coordinates in fragment local frame ``[N_atom, 3]``.
        fragment_id: Fragment assignment per atom ``[N_atom]``.
        n_frag: Number of fragments.
        eps: Regularisation added to diagonal.

    Returns:
        Tensor of shape ``[N_frag, 3, 3]``.
    """
    r = local_pos  # [N, 3]
    r_sq = (r * r).sum(-1, keepdim=True)  # [N, 1]
    # I_atom = |r|^2 * I_3 - r ⊗ r   per atom
    I_atom = r_sq.unsqueeze(-1) * torch.eye(3, device=r.device, dtype=r.dtype).unsqueeze(0)
    I_atom = I_atom - r.unsqueeze(-1) * r.unsqueeze(-2)  # [N, 3, 3]
    # Scatter-add per fragment
    I_frag = torch.zeros(n_frag, 3, 3, device=r.device, dtype=r.dtype)
    I_frag.scatter_add_(0, fragment_id.view(-1, 1, 1).expand(-1, 3, 3), I_atom)
    # Regularise: eps proportional to trace for numerical stability
    trace = I_frag.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)  # [N, 1, 1]
    reg = (eps * trace.clamp(min=1.0)) * torch.eye(3, device=r.device, dtype=r.dtype).unsqueeze(0)
    I_frag = I_frag + reg
    return I_frag


# ---------------------------------------------------------------------------
# Multi-resolution interaction layer
# ---------------------------------------------------------------------------

class _MultiResInteractionLayer(nn.Module):
    """Two-phase equivariant interaction: atom update then fragment update.

    Phase 1 (Atom):  residue->atom TP + atom<->atom bond TP + frag->atom FiLM
    Phase 2 (Frag):  atom->frag re-lift TP + protein->frag TP + frag<->frag TP
    """

    def __init__(
        self,
        scalar_dim: int,
        vec_dim: int,
        t_emb_dim: int,
        n_rbf: int = 16,
        dropout: float = 0.0,
        use_cut_bond_edges: bool = False,
    ) -> None:
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vec_dim = vec_dim
        self.n_rbf = n_rbf
        self.use_cut_bond_edges = use_cut_bond_edges

        # Unified irreps for all node types (protein, atom, fragment)
        node_irreps = cue.Irreps("O3", f"{scalar_dim}x0e + {vec_dim}x1o + {vec_dim}x1e")

        # Edge scalar dims: rbf + src_scalar + dst_scalar + t_emb [+ ctx]
        ra_edge = n_rbf + scalar_dim + scalar_dim + t_emb_dim
        aa_edge = n_rbf + scalar_dim + scalar_dim + scalar_dim + t_emb_dim
        cut_edge = n_rbf + scalar_dim + scalar_dim + scalar_dim + scalar_dim + t_emb_dim
        # tri: rbf(dist) + rbf(|delta_d|) + delta_d(1) + src_s + dst_s + frag_s + t_emb
        tri_edge = n_rbf + n_rbf + 1 + scalar_dim + scalar_dim + scalar_dim + t_emb_dim
        af_edge = n_rbf + scalar_dim + scalar_dim + t_emb_dim
        pf_edge = n_rbf + scalar_dim + scalar_dim + t_emb_dim
        ff_edge = n_rbf + scalar_dim + scalar_dim + t_emb_dim

        # ---- Phase 1: Atom-level TPs ----
        self.ra_conv = EquivariantTPConv(
            node_irreps, node_irreps, sh_lmax=2, edge_scalar_dim=ra_edge, n_rbf=n_rbf,
        )
        self.aa_conv = EquivariantTPConv(
            node_irreps, node_irreps, sh_lmax=2, edge_scalar_dim=aa_edge, n_rbf=n_rbf,
        )
        if use_cut_bond_edges:
            self.cut_conv = EquivariantTPConv(
                node_irreps, node_irreps, sh_lmax=2, edge_scalar_dim=cut_edge, n_rbf=n_rbf,
            )
        else:
            self.cut_conv = None
        self.tri_conv = EquivariantTPConv(
            node_irreps, node_irreps, sh_lmax=2, edge_scalar_dim=tri_edge, n_rbf=n_rbf,
        )
        self.atom_proj = cuet.Linear(node_irreps, node_irreps, layout=cue.mul_ir)
        self.atom_act = ScalarActivation(node_irreps)
        self.atom_dropout = EquivariantDropout(node_irreps, p=dropout)
        self.atom_ada_ln = EquivariantAdaLN(node_irreps, t_emb_dim + scalar_dim)

        # ---- Phase 2: Fragment-level TPs ----
        self.af_conv = EquivariantTPConv(
            node_irreps, node_irreps, sh_lmax=2, edge_scalar_dim=af_edge, n_rbf=n_rbf,
        )
        self.pf_conv = EquivariantTPConv(
            node_irreps, node_irreps, sh_lmax=2, edge_scalar_dim=pf_edge, n_rbf=n_rbf,
        )
        self.ff_conv = EquivariantTPConv(
            node_irreps, node_irreps, sh_lmax=2, edge_scalar_dim=ff_edge, n_rbf=n_rbf,
        )
        self.frag_proj = cuet.Linear(node_irreps, node_irreps, layout=cue.mul_ir)
        self.frag_act = ScalarActivation(node_irreps)
        self.frag_dropout = EquivariantDropout(node_irreps, p=dropout)
        self.frag_ada_ln = EquivariantAdaLN(node_irreps, t_emb_dim)

    def forward(
        self,
        h_atom: Tensor,
        h_frag: Tensor,
        h_prot: Tensor,
        atom_pos_t: Tensor,
        T_frag: Tensor,
        prot_pos: Tensor,
        fragment_id: Tensor,
        ra_src: Tensor, ra_dst: Tensor,
        bond_src: Tensor, bond_dst: Tensor,
        cut_src: Tensor, cut_dst: Tensor,
        pf_src: Tensor, pf_dst: Tensor,
        ff_src: Tensor, ff_dst: Tensor,
        tri_src: Tensor, tri_dst: Tensor, tri_ref_dist: Tensor,
        t_emb_atom: Tensor,
        t_emb_frag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        n_atom = h_atom.shape[0]
        n_frag = h_frag.shape[0]
        S = self.scalar_dim
        h_atom_s = h_atom[:, :S]
        h_frag_s = h_frag[:, :S]
        h_prot_s = h_prot[:, :S]

        # ============== Phase 1: Atom update ==============

        # 1a. Residue -> atom
        if ra_src.shape[0] > 0:
            diff_ra = atom_pos_t[ra_dst] - prot_pos[ra_src]
            dist_ra = torch.linalg.vector_norm(diff_ra, dim=-1)
            ra_es = torch.cat([
                rbf_encode(dist_ra, self.n_rbf),
                h_prot_s[ra_src], h_atom_s[ra_dst], t_emb_atom[ra_dst],
            ], dim=-1)
            ra_msg = self.ra_conv(h_prot, diff_ra, ra_es, ra_src, ra_dst, n_atom)
        else:
            ra_msg = h_atom.new_zeros(n_atom, h_atom.shape[1])

        # 1b. Atom <-> atom (bond graph)
        if bond_src.shape[0] > 0:
            diff_aa = atom_pos_t[bond_dst] - atom_pos_t[bond_src]
            dist_aa = torch.linalg.vector_norm(diff_aa, dim=-1)
            aa_es = torch.cat([
                rbf_encode(dist_aa, self.n_rbf),
                h_atom_s[bond_src], h_atom_s[bond_dst],
                h_frag_s[fragment_id[bond_dst]],
                t_emb_atom[bond_dst],
            ], dim=-1)
            aa_msg = self.aa_conv(h_atom, diff_aa, aa_es, bond_src, bond_dst, n_atom)
        else:
            aa_msg = h_atom.new_zeros(n_atom, h_atom.shape[1])

        # 1c. Explicit cut-bond messages across broken rotatable bonds
        if self.cut_conv is not None and cut_src.shape[0] > 0:
            diff_cut = atom_pos_t[cut_dst] - atom_pos_t[cut_src]
            dist_cut = torch.linalg.vector_norm(diff_cut, dim=-1)
            cut_es = torch.cat([
                rbf_encode(dist_cut, self.n_rbf),
                h_atom_s[cut_src], h_atom_s[cut_dst],
                h_frag_s[fragment_id[cut_src]], h_frag_s[fragment_id[cut_dst]],
                t_emb_atom[cut_dst],
            ], dim=-1)
            cut_msg = self.cut_conv(h_atom, diff_cut, cut_es, cut_src, cut_dst, n_atom)
        else:
            cut_msg = h_atom.new_zeros(n_atom, h_atom.shape[1])

        # 1d. Triangulation: cross-fragment edges near cut bonds
        if tri_src.shape[0] > 0:
            diff_tri = atom_pos_t[tri_dst] - atom_pos_t[tri_src]
            dist_tri = torch.linalg.vector_norm(diff_tri, dim=-1)
            delta_d = dist_tri - tri_ref_dist
            tri_es = torch.cat([
                rbf_encode(dist_tri, self.n_rbf),
                rbf_encode(delta_d.abs(), self.n_rbf),
                delta_d.unsqueeze(-1),
                h_atom_s[tri_src], h_atom_s[tri_dst],
                h_frag_s[fragment_id[tri_dst]],
                t_emb_atom[tri_dst],
            ], dim=-1)
            tri_msg = self.tri_conv(h_atom, diff_tri, tri_es, tri_src, tri_dst, n_atom)
        else:
            tri_msg = h_atom.new_zeros(n_atom, h_atom.shape[1])

        # 1e. Residual + frag->atom FiLM via AdaLN
        atom_agg = ra_msg + aa_msg + cut_msg + tri_msg
        atom_update = self.atom_proj(atom_agg)
        atom_update = self.atom_act(atom_update)
        atom_update = self.atom_dropout(atom_update)
        h_atom = h_atom + atom_update
        atom_cond = torch.cat([t_emb_atom, h_frag_s[fragment_id]], dim=-1)
        h_atom = self.atom_ada_ln(h_atom, atom_cond)

        # Refresh scalar views
        h_atom_s = h_atom[:, :S]
        h_frag_s = h_frag[:, :S]

        # ============== Phase 2: Fragment update ==============

        # 2a. Atom -> frag re-lift
        r_af = atom_pos_t - T_frag[fragment_id]
        dist_af = torch.linalg.vector_norm(r_af, dim=-1)
        af_es = torch.cat([
            rbf_encode(dist_af, self.n_rbf),
            h_atom_s, h_frag_s[fragment_id], t_emb_atom,
        ], dim=-1)
        af_msg = self.af_conv(
            h_atom, r_af, af_es,
            src_idx=torch.arange(n_atom, device=h_atom.device),
            dst_idx=fragment_id, n_dst=n_frag,
        )

        # 2b. Protein -> frag
        if pf_src.shape[0] > 0:
            diff_pf = T_frag[pf_dst] - prot_pos[pf_src]
            dist_pf = torch.linalg.vector_norm(diff_pf, dim=-1)
            pf_es = torch.cat([
                rbf_encode(dist_pf, self.n_rbf),
                h_prot_s[pf_src], h_frag_s[pf_dst], t_emb_frag[pf_dst],
            ], dim=-1)
            pf_msg = self.pf_conv(h_prot, diff_pf, pf_es, pf_src, pf_dst, n_frag)
        else:
            pf_msg = h_frag.new_zeros(n_frag, h_frag.shape[1])

        # 2c. Frag <-> frag (FULL irreps source)
        if ff_src.shape[0] > 0:
            diff_ff = T_frag[ff_dst] - T_frag[ff_src]
            dist_ff = torch.linalg.vector_norm(diff_ff, dim=-1)
            ff_es = torch.cat([
                rbf_encode(dist_ff, self.n_rbf),
                h_frag_s[ff_src], h_frag_s[ff_dst], t_emb_frag[ff_dst],
            ], dim=-1)
            ff_msg = self.ff_conv(h_frag, diff_ff, ff_es, ff_src, ff_dst, n_frag)
        else:
            ff_msg = h_frag.new_zeros(n_frag, h_frag.shape[1])

        # 2d. Fragment residual update
        frag_agg = af_msg + pf_msg + ff_msg
        frag_update = self.frag_proj(frag_agg)
        frag_update = self.frag_act(frag_update)
        frag_update = self.frag_dropout(frag_update)
        h_frag = h_frag + frag_update
        h_frag = self.frag_ada_ln(h_frag, t_emb_frag)

        return h_atom, h_frag


# ---------------------------------------------------------------------------
# DockingHead
# ---------------------------------------------------------------------------

class DockingHead(nn.Module):
    """Multi-resolution SE(3)-equivariant docking head.

    Two-phase per layer:
      Phase 1 (Atom): residue->atom + atom<->atom bond + frag->atom FiLM
      Phase 2 (Frag): atom->frag re-lift + protein->frag + frag<->frag (full irreps)

    Predicts per-fragment translation velocity (v) and angular velocity (omega).

    Args:
        atom_dim: Dimension of atom features from LigandEncoder.
        prot_dim: Dimension of protein residue features from ProteinEncoder.
        hidden_scalar_dim: Scalar channels (0e) in hidden states.
        hidden_vec_dim: Vector channels (1o, 1e) in hidden states.
        num_layers: Number of multi-resolution interaction layers.
        max_frag_size: Maximum fragment size (for size embedding).
        n_rbf: RBF basis count for distance encoding.
        t_emb_dim: Time embedding dimension.
        pf_radius: Protein->fragment interaction radius (Angstroms).
        ff_radius: Fragment<->fragment interaction radius (Angstroms).
        ra_radius: Residue->atom interaction radius (Angstroms).
        pf_topology: Protein->fragment coarse topology ('radius' or 'full').
        ff_topology: Fragment<->fragment coarse topology ('radius' or 'full').
        use_cut_bond_edges: Whether to use explicit cut-bond atom edges.
        pf_max_neighbors: Max protein neighbors per fragment.
        ff_max_neighbors: Max fragment neighbors per fragment.
        ra_max_neighbors: Max residue neighbors per atom.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        atom_dim: int = 128,
        prot_dim: int = 128,
        hidden_scalar_dim: int = 128,
        hidden_vec_dim: int = 32,
        num_layers: int = 3,
        max_frag_size: int = 30,
        n_rbf: int = 16,
        t_emb_dim: int = 128,
        pf_radius: float = 10.0,
        ff_radius: float = 6.0,
        ra_radius: float = 8.0,
        pf_topology: str = "radius",
        ff_topology: str = "radius",
        use_cut_bond_edges: bool = False,
        pf_max_neighbors: int = 64,
        ff_max_neighbors: int = 8,
        ra_max_neighbors: int = 32,
        dropout: float = 0.0,
        omega_mode: str = "neural_1e",
    ) -> None:
        super().__init__()
        self.hidden_scalar_dim = hidden_scalar_dim
        self.hidden_vec_dim = hidden_vec_dim
        self.omega_mode = omega_mode
        self.pf_radius = pf_radius
        self.ff_radius = ff_radius
        self.ra_radius = ra_radius
        if pf_topology not in {"radius", "full"}:
            raise ValueError(f"Unsupported pf_topology '{pf_topology}'.")
        if ff_topology not in {"radius", "full"}:
            raise ValueError(f"Unsupported ff_topology '{ff_topology}'.")
        self.pf_topology = pf_topology
        self.ff_topology = ff_topology
        self.use_cut_bond_edges = use_cut_bond_edges
        self.pf_max_neighbors = pf_max_neighbors
        self.ff_max_neighbors = ff_max_neighbors
        self.ra_max_neighbors = ra_max_neighbors
        self.n_rbf = n_rbf

        # --- Time embedding ---
        self.t_emb_mlp = nn.Sequential(
            nn.Linear(32, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        # --- Protein projection + vector init ---
        self.prot_proj = nn.Linear(prot_dim, hidden_scalar_dim)
        self.prot_vec_gate = nn.Sequential(
            nn.Linear(hidden_scalar_dim, hidden_vec_dim),
            nn.Tanh(),
        )

        # --- Fragment scalar init ---
        self.frag_size_emb = nn.Embedding(max_frag_size + 1, 16)
        self.frag_init_mlp = nn.Sequential(
            nn.Linear(atom_dim + 16 + t_emb_dim, hidden_scalar_dim),
            nn.SiLU(),
            nn.Linear(hidden_scalar_dim, hidden_scalar_dim),
        )

        # --- Atom state init ---
        self.atom_init_mlp = nn.Sequential(
            nn.Linear(atom_dim + t_emb_dim, hidden_scalar_dim),
            nn.SiLU(),
            nn.Linear(hidden_scalar_dim, hidden_scalar_dim),
        )
        self.atom_vec_gate = nn.Sequential(
            nn.Linear(hidden_scalar_dim, hidden_vec_dim),
            nn.Tanh(),
        )
        # Gate for injecting rotation matrix columns as 1o features into fragments
        # 3 columns → 3 gates, each producing vec_dim channels
        self.frag_rot_gate = nn.Sequential(
            nn.Linear(hidden_scalar_dim, hidden_vec_dim * 3),
            nn.Tanh(),
        )

        # --- Equivariant lift: atom -> fragment (initial) ---
        in_scalar_irreps = cue.Irreps("O3", f"{atom_dim}x0e")
        lifted_irreps = cue.Irreps("O3", f"{hidden_scalar_dim}x0e + {hidden_vec_dim}x1o")
        lift_edge_dim = n_rbf + atom_dim + hidden_scalar_dim + t_emb_dim

        self.lift_conv = EquivariantTPConv(
            src_irreps=in_scalar_irreps,
            dst_irreps=lifted_irreps,
            sh_lmax=1,
            edge_scalar_dim=lift_edge_dim,
            n_rbf=n_rbf,
        )

        # --- Interaction layers ---
        self.interaction_layers = nn.ModuleList([
            _MultiResInteractionLayer(
                scalar_dim=hidden_scalar_dim,
                vec_dim=hidden_vec_dim,
                t_emb_dim=t_emb_dim,
                n_rbf=n_rbf,
                dropout=dropout,
                use_cut_bond_edges=use_cut_bond_edges,
            )
            for _ in range(num_layers)
        ])

        # --- Output head ---
        frag_irreps = cue.Irreps("O3", f"{hidden_scalar_dim}x0e + {hidden_vec_dim}x1o + {hidden_vec_dim}x1e")
        head_irreps = cue.Irreps("O3", f"{hidden_scalar_dim // 2}x0e + {hidden_vec_dim}x1o + {hidden_vec_dim}x1e")

        self.head_pre = cuet.Linear(frag_irreps, head_irreps, layout=cue.mul_ir)
        self.head_act = ScalarActivation(head_irreps)
        self.v_linear = cuet.Linear(head_irreps, cue.Irreps("O3", "1x1o"), layout=cue.mul_ir)

        atom_irreps = cue.Irreps("O3", f"{hidden_scalar_dim}x0e + {hidden_vec_dim}x1o + {hidden_vec_dim}x1e")
        atom_head_irreps = cue.Irreps("O3", f"{hidden_scalar_dim // 2}x0e + {hidden_vec_dim}x1o")

        if omega_mode == "analytic":
            self.omega_scale = nn.Sequential(
                nn.Linear(hidden_scalar_dim, hidden_scalar_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_scalar_dim // 2, 1),
            )
        elif omega_mode == "local_frame":
            # Predict omega in fragment body frame from invariant features
            # Input: fragment scalar + invariant protein-fragment features
            self.omega_body_mlp = nn.Sequential(
                nn.Linear(hidden_scalar_dim + n_rbf + t_emb_dim, hidden_scalar_dim),
                nn.SiLU(),
                nn.Linear(hidden_scalar_dim, hidden_scalar_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_scalar_dim // 2, 3),
            )
        elif omega_mode == "neural_1e":
            self.omega_linear = cuet.Linear(head_irreps, cue.Irreps("O3", "1x1e"), layout=cue.mul_ir)
        elif omega_mode == "neural_1e_separate":
            # Dedicated omega branch: own pre-projection + activation + time conditioning
            # Does NOT share h_head with translation, avoiding gradient interference
            self.omega_head_pre = cuet.Linear(frag_irreps, head_irreps, layout=cue.mul_ir)
            self.omega_head_act = ScalarActivation(head_irreps)
            self.omega_head_ada = EquivariantAdaLN(head_irreps, t_emb_dim)
            self.omega_head_proj = cuet.Linear(head_irreps, head_irreps, layout=cue.mul_ir)
            self.omega_head_act2 = ScalarActivation(head_irreps)
            self.omega_head_out = cuet.Linear(head_irreps, cue.Irreps("O3", "1x1e"), layout=cue.mul_ir)
        elif omega_mode in ("newton_euler", "atom_velocity"):
            # Per-atom head: predict atom-level 1o vectors
            self.atom_head_pre = cuet.Linear(atom_irreps, atom_head_irreps, layout=cue.mul_ir)
            self.atom_head_act = ScalarActivation(atom_head_irreps)
            self.atom_force_linear = cuet.Linear(atom_head_irreps, cue.Irreps("O3", "1x1o"), layout=cue.mul_ir)

    def forward(
        self,
        h_prot: Tensor,
        h_atom: Tensor,
        batch: HeteroData,
    ) -> tuple[Tensor, Tensor]:
        """Predict fragment velocities.

        Args:
            h_prot: Protein residue features ``[N_res, prot_dim]``.
            h_atom: Atom features ``[N_atom, atom_dim]``.
            batch: PyG HeteroData batch.

        Returns:
            Tuple ``(v_pred, omega_pred)`` each of shape ``[N_frag, 3]``.
        """
        fragment_id: Tensor = batch["atom"].fragment_id
        T_frag: Tensor = batch["fragment"].T_frag
        frag_sizes: Tensor = batch["fragment"].size
        atom_pos_t: Tensor = batch["atom"].pos_t
        prot_pos: Tensor = batch["protein"].pos
        t_val: Tensor = batch.t

        n_frag = T_frag.shape[0]
        n_atom = h_atom.shape[0]
        scalar_dim = self.hidden_scalar_dim
        vec_dim = self.hidden_vec_dim
        device = T_frag.device

        # Batch vectors
        frag_batch: Tensor | None = getattr(batch["fragment"], "batch", None)
        prot_batch: Tensor | None = getattr(batch["protein"], "batch", None)
        atom_batch: Tensor | None = getattr(batch["atom"], "batch", None)

        # --- Time embedding ---
        t_scalar = t_val.view(-1)
        t_sin = sinusoidal_embedding(t_scalar, dim=32)
        t_emb = self.t_emb_mlp(t_sin)

        if frag_batch is not None and t_emb.shape[0] > 1:
            t_emb_frag = t_emb[frag_batch]
            t_emb_atom = t_emb[atom_batch] if atom_batch is not None else t_emb.expand(n_atom, -1)
        else:
            t_emb_frag = t_emb.expand(n_frag, -1)
            t_emb_atom = t_emb.expand(n_atom, -1)

        # --- Fragment scalar init ---
        h_frag_atom = scatter_mean(h_atom, fragment_id, n_frag)
        size_clamped = frag_sizes.clamp(max=30).long()
        size_feat = self.frag_size_emb(size_clamped)
        h_frag_scalar = self.frag_init_mlp(
            torch.cat([h_frag_atom, size_feat, t_emb_frag], dim=-1)
        )

        # --- Equivariant lift: atom -> fragment ---
        r_af = atom_pos_t - T_frag[fragment_id]
        dist_af = torch.linalg.vector_norm(r_af, dim=-1)
        lift_edge_scalars = torch.cat([
            rbf_encode(dist_af, n_rbf=self.n_rbf),
            h_atom, h_frag_scalar[fragment_id], t_emb_atom,
        ], dim=-1)
        h_lifted = self.lift_conv(
            h_atom, r_af, lift_edge_scalars,
            src_idx=torch.arange(n_atom, device=device),
            dst_idx=fragment_id, n_dst=n_frag,
        )

        # Build fragment hidden: scalar + 1o + 1e
        feat_dim = scalar_dim + vec_dim * 3 + vec_dim * 3
        h_frag = torch.zeros(n_frag, feat_dim, device=device, dtype=T_frag.dtype)
        h_frag[:, :scalar_dim] = h_frag_scalar + h_lifted[:, :scalar_dim]
        h_frag[:, scalar_dim:scalar_dim + vec_dim * 3] = h_lifted[:, scalar_dim:]

        # Inject rotation matrix columns as additional 1o features (equivariant)
        # R_frag[:, :, i] = i-th column → transforms as R_g @ col_i under rotation
        q_frag: Tensor = batch["fragment"].q_frag
        R_frag = quaternion_to_matrix(q_frag)  # [N_frag, 3, 3]
        R_cols = R_frag.transpose(-1, -2)  # [N_frag, 3, 3] — R_cols[:, i, :] = i-th column
        rot_gate = self.frag_rot_gate(h_frag_scalar)  # [N_frag, vec_dim * 3]
        rot_gate = rot_gate.view(n_frag, 3, vec_dim)  # [N_frag, 3_cols, vec_dim]
        rot_vecs = rot_gate.unsqueeze(-1) * R_cols.unsqueeze(2)  # [N_frag, 3, vec_dim, 3]
        rot_vecs = rot_vecs.sum(dim=1)  # [N_frag, vec_dim, 3] — weighted sum of columns
        h_frag[:, scalar_dim:scalar_dim + vec_dim * 3] += rot_vecs.reshape(n_frag, vec_dim * 3)

        # --- Atom state init: 0e + 1o + 1e ---
        h_atom_scalar = self.atom_init_mlp(torch.cat([h_atom, t_emb_atom], dim=-1))
        gate = self.atom_vec_gate(h_atom_scalar)          # [N_atom, vec_dim]
        h_atom_1o = gate.unsqueeze(-1) * r_af.unsqueeze(1)  # [N_atom, vec_dim, 3]
        h_atom_1o = h_atom_1o.reshape(n_atom, vec_dim * 3)
        h_atom_1e = torch.zeros(n_atom, vec_dim * 3, device=device, dtype=T_frag.dtype)
        h_atom_state = torch.cat([h_atom_scalar, h_atom_1o, h_atom_1e], dim=-1)

        # --- Build edge graphs ---
        empty = torch.zeros(0, dtype=torch.long, device=device)

        # Residue -> atom (radius 8A)
        if prot_pos.shape[0] > 0 and n_atom > 0:
            ra_src, ra_dst = _build_cross_radius_edges(
                prot_pos, atom_pos_t, r=self.ra_radius,
                max_neighbors=self.ra_max_neighbors,
                src_batch=prot_batch, dst_batch=atom_batch,
            )
        else:
            ra_src, ra_dst = empty, empty

        # Atom <-> atom (bond graph, symmetrized)
        bond_idx = batch["atom", "bond", "atom"].edge_index
        bond_src = torch.cat([bond_idx[0], bond_idx[1]])
        bond_dst = torch.cat([bond_idx[1], bond_idx[0]])

        # Explicit cut-bond edges (rotatable-bond anchors across fragments)
        if self.use_cut_bond_edges:
            try:
                cut_edge = batch["atom", "cut", "atom"]
                if hasattr(cut_edge, "edge_index") and cut_edge.edge_index.shape[1] > 0:
                    cut_src = cut_edge.edge_index[0]
                    cut_dst = cut_edge.edge_index[1]
                else:
                    cut_src, cut_dst = empty, empty
            except (KeyError, AttributeError):
                cut_src, cut_dst = empty, empty
        else:
            cut_src, cut_dst = empty, empty

        # Triangulation edges (cross-fragment, from preprocessing)
        try:
            tri_edge = batch["atom", "tri", "atom"]
            if hasattr(tri_edge, "edge_index") and tri_edge.edge_index.shape[1] > 0:
                tri_src = tri_edge.edge_index[0]
                tri_dst = tri_edge.edge_index[1]
                tri_ref_dist = tri_edge.ref_dist
            else:
                tri_src, tri_dst = empty, empty
                tri_ref_dist = torch.zeros(0, device=device, dtype=T_frag.dtype)
        except (KeyError, AttributeError):
            tri_src, tri_dst = empty, empty
            tri_ref_dist = torch.zeros(0, device=device, dtype=T_frag.dtype)

        # Protein -> frag (local radius or global coarse graph)
        if prot_pos.shape[0] > 0 and n_frag > 0:
            pf_src, pf_dst = _build_cross_topology_edges(
                prot_pos, T_frag,
                topology=self.pf_topology,
                r=self.pf_radius,
                max_neighbors=self.pf_max_neighbors,
                src_batch=prot_batch, dst_batch=frag_batch,
            )
        else:
            pf_src, pf_dst = empty, empty

        # Frag <-> frag (local radius or global coarse graph)
        if n_frag > 1:
            ff_src, ff_dst = _build_cross_topology_edges(
                T_frag, T_frag,
                topology=self.ff_topology,
                r=self.ff_radius,
                max_neighbors=self.ff_max_neighbors,
                src_batch=frag_batch, dst_batch=frag_batch,
            )
            mask = ff_src != ff_dst
            ff_src, ff_dst = ff_src[mask], ff_dst[mask]
            if self.ff_topology == "radius":
                # Add topological adjacency from cut bonds only in the local graph mode.
                try:
                    frag_adj = batch["fragment", "adj", "fragment"]
                    has_adj = hasattr(frag_adj, "edge_index") and frag_adj.edge_index.shape[1] > 0
                except (KeyError, AttributeError):
                    has_adj = False
                if has_adj:
                    ff_src = torch.cat([ff_src, frag_adj.edge_index[0]])
                    ff_dst = torch.cat([ff_dst, frag_adj.edge_index[1]])
        else:
            ff_src, ff_dst = empty, empty

        # --- Protein equivariant init: 0e + 1o + 1e ---
        n_prot = prot_pos.shape[0]
        h_prot_s = self.prot_proj(h_prot)
        if n_prot > 0:
            if prot_batch is not None:
                prot_center = scatter_mean(prot_pos, prot_batch, int(prot_batch.max()) + 1)
                r_prot = prot_pos - prot_center[prot_batch]
            else:
                r_prot = prot_pos - prot_pos.mean(0, keepdim=True)
            prot_gate = self.prot_vec_gate(h_prot_s)
            h_prot_1o = (prot_gate.unsqueeze(-1) * r_prot.unsqueeze(1)).reshape(n_prot, vec_dim * 3)
        else:
            h_prot_1o = torch.zeros(0, vec_dim * 3, device=device, dtype=T_frag.dtype)
        h_prot_1e = torch.zeros(n_prot, vec_dim * 3, device=device, dtype=T_frag.dtype)
        h_prot_full = torch.cat([h_prot_s, h_prot_1o, h_prot_1e], dim=-1)

        # --- Interaction layers ---
        for layer in self.interaction_layers:
            h_atom_state, h_frag = layer(
                h_atom_state, h_frag, h_prot_full,
                atom_pos_t, T_frag, prot_pos, fragment_id,
                ra_src, ra_dst, bond_src, bond_dst, cut_src, cut_dst,
                pf_src, pf_dst, ff_src, ff_dst,
                tri_src, tri_dst, tri_ref_dist,
                t_emb_atom, t_emb_frag,
            )

        # --- Output head ---
        h_head = self.head_pre(h_frag)
        h_head = self.head_act(h_head)
        v_pred = self.v_linear(h_head)

        if self.omega_mode == "analytic":
            axis_angle = quaternion_to_axis_angle(q_frag)
            scale = self.omega_scale(h_frag[:, :self.hidden_scalar_dim])
            omega_pred = scale * axis_angle

        elif self.omega_mode == "local_frame":
            # Transform protein coords to fragment body frame → invariant features
            R_frag = quaternion_to_matrix(q_frag)  # [N_frag, 3, 3]
            h_frag_scalar = h_frag[:, :self.hidden_scalar_dim]

            # For each fragment, compute mean distance to nearby protein residues (invariant)
            # Use the pf edges already built
            pf_inv_feats = torch.zeros(n_frag, self.n_rbf, device=device, dtype=T_frag.dtype)
            if pf_src.shape[0] > 0:
                # Transform protein positions to each fragment's body frame
                diff_pf_global = prot_pos[pf_src] - T_frag[pf_dst]  # [E_pf, 3]
                # Rotate to body frame: R^T @ diff
                diff_pf_body = torch.einsum(
                    "nij,nj->ni",
                    R_frag[pf_dst].transpose(-1, -2),
                    diff_pf_global,
                )  # [E_pf, 3] — invariant under global rotation
                dist_pf_body = diff_pf_body.norm(dim=-1)
                rbf_pf = rbf_encode(dist_pf_body, self.n_rbf)  # [E_pf, n_rbf]
                # Aggregate per fragment (mean)
                count = torch.zeros(n_frag, 1, device=device, dtype=T_frag.dtype)
                pf_inv_feats.scatter_add_(0, pf_dst.unsqueeze(-1).expand_as(rbf_pf), rbf_pf)
                count.scatter_add_(0, pf_dst.unsqueeze(-1), torch.ones(pf_dst.shape[0], 1, device=device, dtype=T_frag.dtype))
                pf_inv_feats = pf_inv_feats / count.clamp(min=1)

            # MLP input: fragment scalar + protein invariant features + time
            omega_input = torch.cat([h_frag_scalar, pf_inv_feats, t_emb_frag], dim=-1)
            omega_body = self.omega_body_mlp(omega_input)  # [N_frag, 3] in body frame

            # Rotate back to global frame
            omega_pred = torch.einsum("nij,nj->ni", R_frag, omega_body)

        elif self.omega_mode == "neural_1e":
            omega_pred = self.omega_linear(h_head)

        elif self.omega_mode == "neural_1e_separate":
            # Dedicated omega branch from h_frag (not h_head)
            h_omega = self.omega_head_pre(h_frag)
            h_omega = self.omega_head_act(h_omega)
            h_omega = self.omega_head_ada(h_omega, t_emb_frag)
            h_omega = h_omega + self.omega_head_proj(h_omega)
            h_omega = self.omega_head_act2(h_omega)
            omega_pred = self.omega_head_out(h_omega)

        elif self.omega_mode == "newton_euler":
            # Predict per-atom forces → torque → inertia solve → omega
            h_atom_head = self.atom_head_pre(h_atom_state)
            h_atom_head = self.atom_head_act(h_atom_head)
            f_atom = self.atom_force_linear(h_atom_head)  # [N_atom, 3]
            r_arm = atom_pos_t - T_frag[fragment_id]  # [N_atom, 3]
            torque = torch.cross(r_arm, f_atom, dim=-1)  # [N_atom, 3]
            tau_global = torch.zeros(n_frag, 3, device=device, dtype=torque.dtype)
            tau_global.scatter_add_(0, fragment_id.unsqueeze(-1).expand_as(torque), torque)

            # Inertia tensor from rest-frame coordinates + proper solve
            local_pos = batch["atom"].local_pos
            I_body = _compute_fragment_inertia(local_pos, fragment_id, n_frag)
            R_frag = quaternion_to_matrix(q_frag)
            tau_body = torch.einsum("nij,nj->ni", R_frag.transpose(-1, -2), tau_global)
            omega_body = torch.linalg.solve(I_body, tau_body.unsqueeze(-1)).squeeze(-1)
            omega_norm = omega_body.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            omega_body = omega_body * (omega_norm.clamp(max=10.0) / omega_norm)
            omega_pred = torch.einsum("nij,nj->ni", R_frag, omega_body)

        elif self.omega_mode == "atom_velocity":
            # Predict per-atom velocity → derive v_frag and omega from atoms
            h_atom_head = self.atom_head_pre(h_atom_state)
            h_atom_head = self.atom_head_act(h_atom_head)
            v_atom_pred = self.atom_force_linear(h_atom_head)  # [N_atom, 3]

            # Override v_pred: scatter_mean of atom velocities
            v_pred = scatter_mean(v_atom_pred, fragment_id, n_frag)

            # Derive omega from residual atom velocity: v_rot = v_atom - v_frag
            v_rot = v_atom_pred - v_pred[fragment_id]
            r_arm = atom_pos_t - T_frag[fragment_id]
            torque = torch.cross(r_arm, v_rot, dim=-1)
            tau_global = torch.zeros(n_frag, 3, device=device, dtype=v_rot.dtype)
            tau_global.scatter_add_(0, fragment_id.unsqueeze(-1).expand_as(torque), torque)

            local_pos = batch["atom"].local_pos
            I_body = _compute_fragment_inertia(local_pos, fragment_id, n_frag)
            R_frag = quaternion_to_matrix(q_frag)
            tau_body = torch.einsum("nij,nj->ni", R_frag.transpose(-1, -2), tau_global)
            omega_body = torch.linalg.solve(I_body, tau_body.unsqueeze(-1)).squeeze(-1)
            omega_norm = omega_body.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            omega_body = omega_body * (omega_norm.clamp(max=10.0) / omega_norm)
            omega_pred = torch.einsum("nij,nj->ni", R_frag, omega_body)

        # Mask single-atom fragments
        omega_pred = omega_pred * (frag_sizes > 1).float().unsqueeze(-1)

        result = (v_pred, omega_pred)
        if self.omega_mode == "atom_velocity":
            return result, v_atom_pred  # type: ignore[possibly-undefined]
        return result



__all__ = ["DockingHead"]
