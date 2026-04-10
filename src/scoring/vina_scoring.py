"""Differentiable AutoDock Vina scoring function.

Adapted from https://github.com/eightmm/lig-mcs-align
"""

import torch

from .vina_params import VINA_WEIGHTS


def precompute_interaction_matrices(
    query_features: dict, pocket_features: dict, device: torch.device,
) -> dict:
    """Precompute feature interaction matrices (independent of coordinates)."""
    num_q = query_features["vdw"].shape[0]
    num_p = pocket_features["vdw"].shape[0]

    q_hydro = query_features["hydro"].unsqueeze(1).expand(-1, num_p)
    p_hydro = pocket_features["hydro"].unsqueeze(0).expand(num_q, -1)
    is_hydrophobic = q_hydro * p_hydro

    q_hbd = query_features["hbd"].unsqueeze(1).expand(-1, num_p)
    p_hba = pocket_features["hba"].unsqueeze(0).expand(num_q, -1)
    q_hba = query_features["hba"].unsqueeze(1).expand(-1, num_p)
    p_hbd = pocket_features["hbd"].unsqueeze(0).expand(num_q, -1)
    is_hbond = ((q_hbd * p_hba) + (q_hba * p_hbd) > 0).float()

    R_ij = query_features["vdw"].unsqueeze(1) + pocket_features["vdw"].unsqueeze(0)

    return {"is_hydrophobic": is_hydrophobic, "is_hbond": is_hbond, "R_ij": R_ij}


def vina_scoring(
    ligand_coords: torch.Tensor,
    pocket_coords: torch.Tensor,
    precomputed: dict,
    weight_preset: str = "vina",
    num_rotatable_bonds: int | None = None,
) -> torch.Tensor:
    """Compute Vina intermolecular energy.

    Args:
        ligand_coords: [B, N_lig, 3] or [N_lig, 3] ligand atom positions.
        pocket_coords: [N_pock, 3] pocket atom positions.
        precomputed: from precompute_interaction_matrices.
        weight_preset: 'vina' or 'vinardo'.
        num_rotatable_bonds: optional torsional entropy penalty.

    Returns:
        energy: [B] or scalar.
    """
    squeeze = False
    if ligand_coords.ndim == 2:
        ligand_coords = ligand_coords.unsqueeze(0)
        squeeze = True

    batch_size = ligand_coords.shape[0]
    pocket_expanded = pocket_coords.unsqueeze(0).expand(batch_size, -1, -1)

    d_ij = torch.cdist(ligand_coords, pocket_expanded)

    R_ij = precomputed["R_ij"].unsqueeze(0).expand(batch_size, -1, -1)
    is_hydrophobic = precomputed["is_hydrophobic"].unsqueeze(0).expand(batch_size, -1, -1)
    is_hbond = precomputed["is_hbond"].unsqueeze(0).expand(batch_size, -1, -1)

    delta_d = d_ij - R_ij

    w = VINA_WEIGHTS[weight_preset]

    if weight_preset == "vinardo":
        gauss1 = torch.exp(-((delta_d / 0.8) ** 2))
        gauss2 = torch.zeros_like(delta_d)
        repulsion = torch.where(delta_d < 0, delta_d ** 2, torch.zeros_like(delta_d))
        hydro_1 = is_hydrophobic * (delta_d <= 0.0).float()
        hydro_2 = is_hydrophobic * (delta_d > 0.0).float() * (delta_d < 2.5).float() * (1.0 - delta_d / 2.5)
        hydrophobic_term = hydro_1 + hydro_2
        hbond_1 = is_hbond * (delta_d <= -0.6).float()
        hbond_2 = is_hbond * (delta_d < 0).float() * (delta_d > -0.6).float() * (-delta_d / 0.6)
        hbond_term = hbond_1 + hbond_2
    else:
        gauss1 = torch.exp(-((delta_d / 0.5) ** 2))
        gauss2 = torch.exp(-(((delta_d - 3.0) / 2.0) ** 2))
        repulsion = torch.where(delta_d < 0, delta_d ** 2, torch.zeros_like(delta_d))
        hydro_1 = is_hydrophobic * (delta_d <= 0.5).float()
        hydro_2 = is_hydrophobic * (delta_d > 0.5).float() * (delta_d < 1.5).float() * (1.5 - delta_d)
        hydrophobic_term = hydro_1 + hydro_2
        hbond_1 = is_hbond * (delta_d <= -0.7).float()
        hbond_2 = is_hbond * (delta_d < 0).float() * (delta_d > -0.7).float() * (-delta_d / 0.7)
        hbond_term = hbond_1 + hbond_2

    energy_matrix = (
        w["gauss1"] * gauss1
        + w["gauss2"] * gauss2
        + w["repulsion"] * repulsion
        + w["hydrophobic"] * hydrophobic_term
        + w["hbond"] * hbond_term
    )

    energy = energy_matrix.sum(dim=(1, 2))

    if num_rotatable_bonds is not None:
        energy = energy / (1.0 + w["rot"] * num_rotatable_bonds)

    if squeeze:
        energy = energy.squeeze(0)
    return energy
