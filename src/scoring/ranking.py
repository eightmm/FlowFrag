"""Pose ranking: Vina score combined with PoseBusters-style validity.

Ranking formula (SigmaDock): ``s_i = -b_i · p_i^β`` where ``b_i`` is Vina
binding energy and ``p_i`` is the averaged PoseBusters validity (see
:mod:`validity`).  Higher ``s_i`` = better pose (more negative Vina × high
validity).
"""
from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .validity import check_physicochemical_validity
from .vina import (
    compute_pocket_features_from_pdb,
    compute_vina_features,
    precompute_interaction_matrices,
    vina_scoring,
)


def rank_poses(
    mol: Chem.Mol,
    poses: list[torch.Tensor],
    pocket_pdb: str | Path,
    pocket_center: torch.Tensor,
    weight_preset: str = "vina",
    validity_beta: float = 4.0,
    device: torch.device = torch.device("cpu"),
    pocket_cutoff: float | None = 15.0,
) -> list[dict]:
    """Rank poses by combined score ``s_i = -vina_i · validity_i^β``.

    If ``pocket_cutoff`` is set, protein atoms further than cutoff Å from
    ``pocket_center`` are filtered out (speeds up full-protein PDBs).
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

        physchem = check_physicochemical_validity(
            mol, pos_abs, protein_coords=pocket_coords,
        )

        p = physchem["validity_score"]
        combined = -v_score * (p ** validity_beta)

        results.append({
            "idx": i,
            "vina_score": v_score,
            "combined_score": combined,
            **physchem,
        })

    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results


__all__ = ["rank_poses"]
