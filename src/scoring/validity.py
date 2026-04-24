"""PoseBusters-style physicochemical validity checks for ligand poses.

Runs 5 ligand-only checks (bond lengths, bond angles, internal steric clash,
tetrahedral chirality, aromatic ring planarity) plus 2 protein-ligand checks
(steric clash, pocket proximity) when protein coordinates are supplied.

Output ``validity_score`` is the mean of per-check pass rates ∈ [0, 1]; used
by :func:`src.scoring.ranking.rank_poses` to down-weight physically
implausible poses via ``s_i = -vina_i · validity_i^β``.
"""
from __future__ import annotations

import math

import torch
from rdkit import Chem


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
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    cos_a = (v1 * v2).sum() / (v1.norm() * v2.norm() + 1e-8)
    cos_a = cos_a.clamp(-1.0, 1.0)
    return math.degrees(math.acos(cos_a.item()))


def _signed_volume(pos: torch.Tensor, center: int, nbrs: list[int]) -> float:
    v0 = pos[nbrs[0]] - pos[center]
    v1 = pos[nbrs[1]] - pos[center]
    v2 = pos[nbrs[2]] - pos[center]
    return torch.dot(v0, torch.cross(v1, v2)).item()


def check_physicochemical_validity(
    mol: Chem.Mol,
    pred_pos: torch.Tensor,
    protein_coords: torch.Tensor | None = None,
    protein_elements: torch.Tensor | None = None,
) -> dict[str, float]:
    """PoseBusters-style validity checks.

    Ligand-only (always): bond lengths, bond angles, internal clash, tetrahedral
    chirality, aromatic-ring planarity.

    With ``protein_coords``/``protein_elements``: adds (i) protein-ligand
    steric clash per ``d < 0.5·(r_lig + r_prot)`` and (ii) pocket proximity
    (min lig-prot distance ≤ 5 Å). Matches SigmaDock's validity composite.
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
            if ref_vol * pred_vol < 0:
                n_chiral_wrong += 1
    chiral_pass = 1.0 - n_chiral_wrong / n_chiral if n_chiral > 0 else 1.0

    # ---- 5. Ring planarity (aromatic rings) ----
    ri = mol.GetRingInfo()
    n_rings = 0
    n_nonplanar = 0
    for ring in ri.AtomRings():
        if not all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            continue
        n_rings += 1
        coords = torch.stack([pred_pos[idx] for idx in ring])
        centered = coords - coords.mean(dim=0, keepdim=True)
        _, s, _ = torch.linalg.svd(centered)
        max_dev = s[-1].item() / math.sqrt(len(ring))
        if max_dev > 0.25:
            n_nonplanar += 1
    planar_pass = 1.0 - n_nonplanar / n_rings if n_rings > 0 else 1.0

    # ---- 6. Protein-ligand steric clash & 7. Pocket proximity ----
    pl_clash_pass = 1.0
    pl_proximity_pass = 1.0
    n_pl_clashes = 0
    if protein_coords is not None and protein_coords.numel() > 0:
        lig_abs = pred_pos.to(protein_coords.device)
        dmat = torch.cdist(lig_abs, protein_coords)  # [N_lig, N_prot]

        r_lig = torch.tensor(
            [_VDW_RADII.get(mol.GetAtomWithIdx(i).GetAtomicNum(), 1.70)
             for i in range(n_atoms)],
            device=protein_coords.device, dtype=protein_coords.dtype,
        )
        if protein_elements is not None:
            r_prot = torch.stack([
                torch.tensor(_VDW_RADII.get(int(z), 1.75),
                             device=protein_coords.device, dtype=protein_coords.dtype)
                for z in protein_elements.tolist()
            ])
        else:
            r_prot = torch.full((protein_coords.shape[0],), 1.75,
                                device=protein_coords.device, dtype=protein_coords.dtype)
        vdw_sum = r_lig.unsqueeze(1) + r_prot.unsqueeze(0)
        n_pl_clashes = int((dmat < 0.5 * vdw_sum).sum().item())
        total_pairs = dmat.numel()
        pl_clash_pass = 1.0 - n_pl_clashes / total_pairs

        min_lig_to_prot = dmat.min(dim=1).values  # [N_lig]
        if min_lig_to_prot.min().item() > 5.0:
            pl_proximity_pass = 0.0

    checks = [bond_pass, angle_pass, clash_pass, chiral_pass, planar_pass]
    if protein_coords is not None and protein_coords.numel() > 0:
        checks.append(pl_clash_pass)
        checks.append(pl_proximity_pass)
    validity_score = sum(checks) / len(checks)
    valid = 1.0 if all(c == 1.0 for c in checks) else 0.0

    return {
        "bond_pass": bond_pass,
        "angle_pass": angle_pass,
        "clash_pass": clash_pass,
        "chiral_pass": chiral_pass,
        "planar_pass": planar_pass,
        "pl_clash_pass": pl_clash_pass,
        "pl_proximity_pass": pl_proximity_pass,
        "n_bad_bonds": float(n_bad_bonds),
        "n_bad_angles": float(n_bad_angles),
        "n_clashes": float(n_clashes),
        "n_chiral_wrong": float(n_chiral_wrong),
        "n_nonplanar": float(n_nonplanar),
        "n_pl_clashes": float(n_pl_clashes),
        "valid": valid,
        "validity_score": validity_score,
    }


__all__ = ["check_physicochemical_validity"]
