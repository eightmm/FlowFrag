"""AutoDock Vina scoring: weights, molecular features, and energy computation.

Adapted from https://github.com/eightmm/lig-mcs-align and refined to match the
AutoDock4 atom-type conventions the reference Vina uses.
"""

import torch
from rdkit import Chem

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
        "gauss1": -0.045,
        "gauss2": 0.0,
        "repulsion": 0.800,
        "hydrophobic": -0.035,
        "hbond": -0.600,
        "rot": 0.05846,
    },
}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# AutoDock4 van der Waals radii (Å). Source: Vina `atom_constants.h` — the
# same table the reference AutoDock Vina uses when typing AD4 atoms.
# RDKit's Bondi radii (GetRvdw) are ~0.2-0.3Å smaller for C/S/halogens and
# would flatten the gauss1/repulsion terms. Keyed by atomic number.
AD4_VDW_RADII = {
    1: 1.00,    # H  (HD)
    6: 2.00,    # C  (aliphatic and aromatic A share R)
    7: 1.75,    # N  (N and NA share R)
    8: 1.60,    # O  (O and OA share R)
    9: 1.545,   # F
    15: 2.10,   # P
    16: 2.00,   # S  (S and SA share R)
    17: 2.045,  # Cl
    34: 1.90,   # Se
    35: 2.165,  # Br
    53: 2.36,   # I
}
_FALLBACK_VDW = 2.00  # generic heavy atom


def _classify_atom_ad4(atom: Chem.Atom) -> tuple[float, float, float]:
    """Return (hydrophobic, hbond-donor, hbond-acceptor) flags for one atom.

    Follows AutoDock4 typing conventions as implemented in Meeko/Vina:
    - Carbon is hydrophobic unless bonded to a polar heteroatom (N/O/S/P).
    - Halogens (F/Cl/Br/I) are hydrophobic only.
    - Oxygen is always an acceptor; also a donor if it carries H (hydroxyl).
    - Nitrogen is a donor if it carries H (amine, amide, indole), an acceptor
      if it has no H and is neutral (pyridine, tertiary amine lone pair).
    - Sulfur is hydrophobic; thiol S is also a donor.
    """
    z = atom.GetAtomicNum()
    # Count both implicit and explicit H (PDB input sometimes leaves a polar H
    # in the graph even after removeHs=True); default includeNeighbors=False
    # would miss those and mis-classify the atom as an acceptor-only.
    n_h = atom.GetTotalNumHs(includeNeighbors=True)
    charge = atom.GetFormalCharge()

    if z in (9, 17, 35, 53):  # halogens
        return (1.0, 0.0, 0.0)
    if z == 6:  # carbon
        has_polar_nbr = any(
            nbr.GetAtomicNum() in (7, 8, 15, 16) for nbr in atom.GetNeighbors()
        )
        return (0.0 if has_polar_nbr else 1.0, 0.0, 0.0)
    if z == 16:  # sulfur
        return (1.0, 1.0 if n_h > 0 else 0.0, 0.0)
    if z == 8:  # oxygen
        return (0.0, 1.0 if n_h > 0 else 0.0, 1.0)
    if z == 7:  # nitrogen
        is_donor = n_h > 0 and charge >= 0
        # Acceptor: sp2/sp3 N with a lone pair (no H, not protonated).
        is_acceptor = n_h == 0 and charge <= 0
        return (0.0, float(is_donor), float(is_acceptor))
    return (0.0, 0.0, 0.0)


def compute_vina_features(mol: Chem.Mol, device: torch.device) -> dict:
    """Extract per-atom AD4 features for Vina: vdW radii, hydrophobic, HBD, HBA.

    Uses AutoDock4 atom typing rules rather than RDKit's generic feature
    factory; sanitizes lightly first so implicit-H counts are available.
    """
    n = mol.GetNumAtoms()
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    radii = torch.zeros(n, dtype=torch.float32, device=device)
    hydro = torch.zeros(n, dtype=torch.float32, device=device)
    hbd = torch.zeros(n, dtype=torch.float32, device=device)
    hba = torch.zeros(n, dtype=torch.float32, device=device)

    for i, atom in enumerate(mol.GetAtoms()):
        radii[i] = AD4_VDW_RADII.get(atom.GetAtomicNum(), _FALLBACK_VDW)
        h, d, a = _classify_atom_ad4(atom)
        hydro[i] = h
        hbd[i] = d
        hba[i] = a

    return {"vdw": radii, "hydro": hydro, "hbd": hbd, "hba": hba}


def _prefilter_pdb_to_pocket(pdb_path: str, center: torch.Tensor, cutoff: float) -> str:
    """Return a temp PDB containing only ATOM/HETATM rows within cutoff of center.

    Pre-filtering is done via string parsing of the 30:54 coordinate columns so
    that exotic records in full-protein PDBs (altloc variants, nonstandard atom
    names like "A") don't trip RDKit before we get a chance to strip them.
    """
    import tempfile
    cx, cy, cz = [float(v) for v in center.detach().cpu().tolist()]
    r2 = cutoff * cutoff
    keep: list[str] = []
    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6]
            if rec in ("ATOM  ", "HETATM"):
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                dx, dy, dz = x - cx, y - cy, z - cz
                if dx * dx + dy * dy + dz * dz > r2:
                    continue
                # Keep only altloc ' ' or 'A' to avoid duplicate parse states.
                if line[16] not in (" ", "A"):
                    continue
                keep.append(line)
            elif rec in ("TER   ", "END   ", "ENDMDL"):
                keep.append(line)
    keep.append("END\n")
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.writelines(keep)
    tmp.close()
    return tmp.name


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
    parse_path = pdb_path
    tmp_path: str | None = None
    if center is not None and cutoff is not None:
        tmp_path = _prefilter_pdb_to_pocket(pdb_path, center, cutoff)
        parse_path = tmp_path
    mol = Chem.MolFromPDBFile(parse_path, removeHs=True, sanitize=False)
    if tmp_path is not None:
        import os as _os
        _os.unlink(tmp_path)
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
