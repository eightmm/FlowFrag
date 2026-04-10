"""Pose ranking: Vina score + PoseBusters-style physicochemical checks.

Implements SigmaDock's ranking formula: s_i = -b_i * p_i^beta
where b_i is Vina binding energy and p_i is average PoseBusters validity.

PB checks (5 items, all distance-geometry based):
  1. Bond lengths — within [0.75, 2.2]A
  2. Bond angles — within ±15° of hybridization ideal
  3. Internal steric clash — non-1,2/1,3 pairs > vdw_sum * 0.75
  4. Tetrahedral chirality — signed volume sign matches reference
  5. Ring planarity — aromatic ring deviation from plane < 0.25A
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .vina_features import compute_pocket_features_from_pdb, compute_vina_features
from .vina_scoring import precompute_interaction_matrices, vina_scoring

# Expected angles per hybridization
_IDEAL_ANGLES = {
    Chem.rdchem.HybridizationType.SP: 180.0,
    Chem.rdchem.HybridizationType.SP2: 120.0,
    Chem.rdchem.HybridizationType.SP3: 109.5,
    Chem.rdchem.HybridizationType.SP3D: 90.0,
    Chem.rdchem.HybridizationType.SP3D2: 90.0,
}

# VdW radii for steric clash check (Bondi radii, common elements)
_VDW_RADII = {1: 1.20, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
              14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 34: 1.90, 35: 1.85, 53: 1.98}


def _get_angle(pos: torch.Tensor, i: int, j: int, k: int) -> float:
    """Compute angle i-j-k in degrees from coordinates."""
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    cos_a = (v1 * v2).sum() / (v1.norm() * v2.norm() + 1e-8)
    cos_a = cos_a.clamp(-1.0, 1.0)
    return math.degrees(math.acos(cos_a.item()))


def _signed_volume(pos: torch.Tensor, center: int, nbrs: list[int]) -> float:
    """Signed volume of tetrahedron formed by center + 3 neighbors."""
    v0 = pos[nbrs[0]] - pos[center]
    v1 = pos[nbrs[1]] - pos[center]
    v2 = pos[nbrs[2]] - pos[center]
    return torch.dot(v0, torch.cross(v1, v2)).item()


def check_physicochemical_validity(
    mol: Chem.Mol, pred_pos: torch.Tensor,
) -> dict[str, float]:
    """PoseBusters-style internal geometry validity checks (5 items).

    Returns dict with per-check pass rates and averaged validity_score in [0, 1].
    PL interaction quality is handled by Vina scoring, not here.
    """
    n_atoms = mol.GetNumAtoms()

    # ---- 1. Bond lengths ----
    n_bonds = mol.GetNumBonds()
    n_bad_bonds = 0
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        d = (pred_pos[i] - pred_pos[j]).norm().item()
        if d < 0.75 or d > 2.2:
            n_bad_bonds += 1
    bond_pass = 1.0 - n_bad_bonds / n_bonds if n_bonds > 0 else 1.0

    # ---- 2. Bond angles ----
    n_angles = 0
    n_bad_angles = 0
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) < 2:
            continue
        ideal = _IDEAL_ANGLES.get(atom.GetHybridization(), 109.5)
        for ii in range(len(nbrs)):
            for jj in range(ii + 1, len(nbrs)):
                angle = _get_angle(pred_pos, nbrs[ii], idx, nbrs[jj])
                n_angles += 1
                if abs(angle - ideal) > 15.0:
                    n_bad_angles += 1
    angle_pass = 1.0 - n_bad_angles / n_angles if n_angles > 0 else 1.0

    # ---- 3. Internal steric clash (non-1,2 and non-1,3 pairs) ----
    # Build 1,2 (bonded) and 1,3 (angle) exclusion sets
    excluded = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        excluded.add((min(i, j), max(i, j)))
    for atom in mol.GetAtoms():
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        for ii in range(len(nbrs)):
            for jj in range(ii + 1, len(nbrs)):
                pair = (min(nbrs[ii], nbrs[jj]), max(nbrs[ii], nbrs[jj]))
                excluded.add(pair)

    n_clashes = 0
    n_checked = 0
    if n_atoms > 1:
        dists = torch.cdist(pred_pos.unsqueeze(0), pred_pos.unsqueeze(0)).squeeze(0)
        for i in range(n_atoms):
            r_i = _VDW_RADII.get(mol.GetAtomWithIdx(i).GetAtomicNum(), 1.70)
            for j in range(i + 1, n_atoms):
                if (i, j) in excluded:
                    continue
                n_checked += 1
                r_j = _VDW_RADII.get(mol.GetAtomWithIdx(j).GetAtomicNum(), 1.70)
                if dists[i, j].item() < (r_i + r_j) * 0.75:
                    n_clashes += 1
    clash_pass = 1.0 - n_clashes / n_checked if n_checked > 0 else 1.0

    # ---- 4. Tetrahedral chirality ----
    # Reference: use mol's existing conformer (crystal) for sign comparison
    ref_conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    chiral_atoms = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
    n_chiral = 0
    n_chiral_wrong = 0
    if ref_conf is not None and chiral_atoms:
        ref_pos = torch.zeros(n_atoms, 3)
        for k in range(n_atoms):
            p = ref_conf.GetAtomPosition(k)
            ref_pos[k] = torch.tensor([p.x, p.y, p.z])

        for center_idx, _ in chiral_atoms:
            atom = mol.GetAtomWithIdx(center_idx)
            nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(nbrs) < 3:
                continue
            n_chiral += 1
            ref_vol = _signed_volume(ref_pos, center_idx, nbrs[:3])
            pred_vol = _signed_volume(pred_pos, center_idx, nbrs[:3])
            if ref_vol * pred_vol < 0:  # sign flip = chirality inversion
                n_chiral_wrong += 1
    chiral_pass = 1.0 - n_chiral_wrong / n_chiral if n_chiral > 0 else 1.0

    # ---- 5. Ring planarity (aromatic rings) ----
    ri = mol.GetRingInfo()
    n_rings = 0
    n_nonplanar = 0
    for ring in ri.AtomRings():
        # Only check aromatic rings
        if not all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            continue
        n_rings += 1
        coords = torch.stack([pred_pos[idx] for idx in ring])
        centered = coords - coords.mean(dim=0, keepdim=True)
        _, s, _ = torch.linalg.svd(centered)
        # Max deviation from plane: smallest singular value / sqrt(n_ring_atoms)
        max_dev = s[-1].item() / math.sqrt(len(ring))
        if max_dev > 0.25:
            n_nonplanar += 1
    planar_pass = 1.0 - n_nonplanar / n_rings if n_rings > 0 else 1.0

    # ---- Aggregate ----
    checks = [bond_pass, angle_pass, clash_pass, chiral_pass, planar_pass]
    validity_score = sum(checks) / len(checks)

    # Binary valid (all checks perfect)
    valid = 1.0 if all(c == 1.0 for c in checks) else 0.0

    return {
        "bond_pass": bond_pass,
        "angle_pass": angle_pass,
        "clash_pass": clash_pass,
        "chiral_pass": chiral_pass,
        "planar_pass": planar_pass,
        "n_bad_bonds": float(n_bad_bonds),
        "n_bad_angles": float(n_bad_angles),
        "n_clashes": float(n_clashes),
        "n_chiral_wrong": float(n_chiral_wrong),
        "n_nonplanar": float(n_nonplanar),
        "valid": valid,
        "validity_score": validity_score,
    }


def rank_poses(
    mol: Chem.Mol,
    poses: list[torch.Tensor],
    pocket_pdb: str | Path,
    pocket_center: torch.Tensor,
    weight_preset: str = "vina",
    validity_beta: float = 4.0,
    device: torch.device = torch.device("cpu"),
    pocket_cutoff: float | None = 8.0,
) -> list[dict]:
    """Rank poses by combined score: s_i = -vina_i * validity_i^beta.

    SigmaDock-style: continuous validity score multiplied with Vina energy.
    Higher s_i = better pose (more negative Vina * high validity).

    If pocket_cutoff is set, protein atoms further than cutoff Å from
    pocket_center are filtered out (speeds up full-protein PDBs).
    """
    pocket_features, pocket_coords = compute_pocket_features_from_pdb(
        str(pocket_pdb), device,
        center=pocket_center if pocket_cutoff else None,
        cutoff=pocket_cutoff,
    )
    lig_features = compute_vina_features(mol, device)
    precomputed = precompute_interaction_matrices(lig_features, pocket_features, device)

    num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

    results = []
    for i, pose in enumerate(poses):
        pos_abs = pose + pocket_center

        v_score = vina_scoring(
            pos_abs.to(device), pocket_coords, precomputed,
            weight_preset=weight_preset,
            num_rotatable_bonds=num_rot_bonds,
        ).item()

        physchem = check_physicochemical_validity(mol, pos_abs)

        # SigmaDock formula: s = -b * p^beta (higher = better)
        p = physchem["validity_score"]
        combined = -v_score * (p ** validity_beta)

        results.append({
            "idx": i,
            "vina_score": v_score,
            "combined_score": combined,
            **physchem,
        })

    # Rank by combined score (higher = better)
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results
