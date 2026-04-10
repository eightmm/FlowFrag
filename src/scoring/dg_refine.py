"""Distance geometry-based refinement for predicted poses.

Optimizes fragment-level (T, R) to minimize cut bond length violations.
Each fragment moves as a rigid body — only inter-fragment bonds are corrected.
"""

from __future__ import annotations

import torch
from rdkit import Chem


def _small_angle_rotation(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle [N, 3] to rotation matrix [N, 3, 3].

    Uses Rodrigues formula with safe gradient at aa=0.
    """
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [N, 1]
    axis = aa / theta  # [N, 3]

    c = torch.cos(theta).unsqueeze(-1)   # [N, 1, 1]
    s = torch.sin(theta).unsqueeze(-1)   # [N, 1, 1]

    # Skew-symmetric matrix K
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    O = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([O, -z, y], dim=-1),
        torch.stack([z, O, -x], dim=-1),
        torch.stack([-y, x, O], dim=-1),
    ], dim=-2)  # [N, 3, 3]

    I = torch.eye(3, device=aa.device).unsqueeze(0)
    R = I + s * K + (1 - c) * (K @ K)
    return R


def get_cut_bond_targets(
    mol: Chem.Mol, crystal_pos: torch.Tensor, fragment_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract inter-fragment (cut) bond pairs and target distances.

    Returns:
        cut_pairs: [2, E_cut] atom indices of cut bonds.
        target_dists: [E_cut] target bond lengths from crystal.
    """
    pairs_i, pairs_j, dists = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if fragment_id[i] != fragment_id[j]:
            pairs_i.append(i)
            pairs_j.append(j)
            dists.append((crystal_pos[i] - crystal_pos[j]).norm().item())

    if not pairs_i:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

    cut_pairs = torch.tensor([pairs_i, pairs_j], dtype=torch.long)
    target_dists = torch.tensor(dists, dtype=torch.float32)
    return cut_pairs, target_dists


def reconstruct_atoms(
    delta_T: torch.Tensor,
    delta_aa: torch.Tensor,
    T_init: torch.Tensor,
    R_init: torch.Tensor,
    local_coords: torch.Tensor,
    fragment_id: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct atom positions from fragment (T, R) with differentiable deltas.

    Args:
        delta_T: [N_frag, 3] translation correction (optimized).
        delta_aa: [N_frag, 3] axis-angle rotation correction (optimized).
        T_init: [N_frag, 3] initial fragment translations.
        R_init: [N_frag, 3, 3] initial fragment rotation matrices.
        local_coords: [N_atom, 3] atom coords in fragment local frame.
        fragment_id: [N_atom] fragment assignment.

    Returns:
        pos: [N_atom, 3] reconstructed atom positions.
    """
    T = T_init + delta_T
    dR = _small_angle_rotation(delta_aa)
    R = dR @ R_init

    R_per_atom = R[fragment_id]
    T_per_atom = T[fragment_id]
    pos = torch.einsum("nij,nj->ni", R_per_atom, local_coords) + T_per_atom
    return pos


def dg_refine(
    mol: Chem.Mol,
    pred_pos: torch.Tensor,
    pocket_center: torch.Tensor,
    crystal_pos: torch.Tensor,
    n_steps: int = 100,
    lr: float = 0.01,
    k_bond: float = 100.0,
    fragment_id: torch.Tensor | None = None,
    local_coords: torch.Tensor | None = None,
) -> torch.Tensor:
    """Refine predicted pose by optimizing fragment rigid body transforms.

    Only cut bonds (inter-fragment) contribute to the loss.
    Intra-fragment geometry is preserved exactly.

    Args:
        mol: RDKit Mol (heavy atoms).
        pred_pos: [N, 3] predicted positions (pocket-centered).
        pocket_center: [3].
        crystal_pos: [N, 3] crystal positions (absolute) for bond targets.
        n_steps: optimization steps.
        lr: learning rate.
        k_bond: bond spring constant.
        fragment_id: [N_atom] fragment assignment per atom.
        local_coords: [N_atom, 3] atom coords in fragment local frame.
    Returns:
        refined positions [N, 3] (pocket-centered).
    """
    if fragment_id is None or local_coords is None:
        return pred_pos

    n_frags = fragment_id.max().item() + 1
    if n_frags <= 1:
        return pred_pos

    cut_pairs, target_dists = get_cut_bond_targets(mol, crystal_pos, fragment_id)
    if cut_pairs.shape[1] == 0:
        return pred_pos

    # Derive initial T per fragment from predicted positions
    pos_abs = pred_pos + pocket_center
    T_init = torch.zeros(n_frags, 3)
    for f in range(n_frags):
        mask = fragment_id == f
        T_init[f] = pos_abs[mask].mean(dim=0)

    # Derive initial R per fragment: solve R_f such that R_f @ local + T_f ≈ pos_abs
    # Use Kabsch / SVD alignment per fragment
    R_init = torch.eye(3).unsqueeze(0).expand(n_frags, -1, -1).clone()
    for f in range(n_frags):
        mask = fragment_id == f
        local_f = local_coords[mask]  # [n_atoms_f, 3]
        pos_f = pos_abs[mask] - T_init[f]  # [n_atoms_f, 3]
        if local_f.shape[0] >= 3:
            H = local_f.T @ pos_f  # [3, 3]
            U, _, Vt = torch.linalg.svd(H)
            d = torch.det(Vt.T @ U.T)
            S = torch.diag(torch.tensor([1.0, 1.0, d.sign()]))
            R_init[f] = Vt.T @ S @ U.T

    # Optimization variables: small corrections to T and R
    delta_T = torch.zeros(n_frags, 3, requires_grad=True)
    delta_aa = torch.zeros(n_frags, 3, requires_grad=True)

    optimizer = torch.optim.Adam([delta_T, delta_aa], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        pos = reconstruct_atoms(delta_T, delta_aa, T_init, R_init, local_coords, fragment_id)
        diff = pos[cut_pairs[0]] - pos[cut_pairs[1]]
        d_cut = diff.norm(dim=-1).clamp(min=1e-6)
        loss = k_bond * ((d_cut - target_dists) ** 2).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_T, delta_aa], max_norm=1.0)
        optimizer.step()
        # Clamp rotation correction to small angles
        with torch.no_grad():
            delta_aa.clamp_(-0.5, 0.5)

    # Final reconstruction
    with torch.no_grad():
        pos_final = reconstruct_atoms(delta_T, delta_aa, T_init, R_init, local_coords, fragment_id)

    return pos_final - pocket_center
