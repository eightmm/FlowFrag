"""AutoDock Vina scoring: weights, molecular features, and energy computation.

Adapted from https://github.com/eightmm/lig-mcs-align
"""

import os

import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

# ---------------------------------------------------------------------------
# Weight parameters
# ---------------------------------------------------------------------------

VINA_WEIGHTS = {
    "vina": {
        "gauss1": -0.035579,
        "gauss2": -0.005156,
        "repulsion": 0.840245,
        "hydrophobic": -0.035069,
        "hbond": -0.587439,
        "rot": 0.05846,
    },
    "vinardo": {
        "gauss1": -0.0356,
        "gauss2": 0.0,
        "repulsion": 0.840,
        "hydrophobic": -0.0351,
        "hbond": -0.587,
        "rot": 0.05846,
    },
}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_FDEF_PATH = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
_FACTORY = ChemicalFeatures.BuildFeatureFactory(_FDEF_PATH)


def compute_vina_features(mol: Chem.Mol, device: torch.device) -> dict:
    """Extract atomic features for Vina scoring: vdW radii, hydrophobic, HBD, HBA."""
    num_atoms = mol.GetNumAtoms()
    ptable = Chem.GetPeriodicTable()

    radii = torch.zeros(num_atoms, dtype=torch.float32, device=device)
    hydro = torch.zeros(num_atoms, dtype=torch.float32, device=device)
    hbd = torch.zeros(num_atoms, dtype=torch.float32, device=device)
    hba = torch.zeros(num_atoms, dtype=torch.float32, device=device)

    for i, atom in enumerate(mol.GetAtoms()):
        radii[i] = ptable.GetRvdw(atom.GetAtomicNum())

    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
    except Exception:
        pass

    feats = _FACTORY.GetFeaturesForMol(mol)
    for feat in feats:
        f_type = feat.GetFamily()
        atom_ids = feat.GetAtomIds()

        if f_type == "Hydrophobe":
            for idx in atom_ids:
                hydro[idx] = 1.0
        elif f_type == "Donor":
            for idx in atom_ids:
                hbd[idx] = 1.0
        elif f_type == "Acceptor":
            for idx in atom_ids:
                hba[idx] = 1.0

    return {"vdw": radii, "hydro": hydro, "hbd": hbd, "hba": hba}


def compute_pocket_features_from_pdb(
    pdb_path: str,
    device: torch.device,
    center: torch.Tensor | None = None,
    cutoff: float | None = None,
) -> tuple[dict, torch.Tensor]:
    """Extract Vina features + coords from pocket PDB.

    If center + cutoff are provided, atoms further than `cutoff` Å from `center`
    are removed BEFORE feature extraction (speeds up full-protein PDBs).

    Returns (features_dict, coords [N, 3]).
    """
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=True, sanitize=False)
    assert mol is not None, f"Failed to parse pocket PDB: {pdb_path}"
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)

    if center is not None and cutoff is not None:
        conf = mol.GetConformer()
        n = mol.GetNumAtoms()
        all_coords = torch.zeros(n, 3, dtype=torch.float32)
        for i in range(n):
            p = conf.GetAtomPosition(i)
            all_coords[i] = torch.tensor([p.x, p.y, p.z])
        c = center.to(all_coords.device) if center.device != all_coords.device else center
        dists = (all_coords - c.unsqueeze(0)).norm(dim=-1)
        keep_mask = dists < cutoff
        if keep_mask.sum() > 0:
            rw = Chem.RWMol(mol)
            remove_indices = sorted(
                (i for i in range(n) if not keep_mask[i].item()),
                reverse=True,
            )
            for i in remove_indices:
                rw.RemoveAtom(i)
            mol = rw.GetMol()
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)

    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    coords = torch.zeros(n, 3, dtype=torch.float32, device=device)
    for i in range(n):
        p = conf.GetAtomPosition(i)
        coords[i] = torch.tensor([p.x, p.y, p.z], device=device)

    features = compute_vina_features(mol, device)
    return features, coords


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

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
