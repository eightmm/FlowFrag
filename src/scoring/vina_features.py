"""Molecular feature extraction for Vina scoring."""

import os

import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


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

    feats = factory.GetFeaturesForMol(mol)
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
    are removed from the mol BEFORE feature extraction. This is critical for
    speed on full-protein PDBs since RDKit's GetFeaturesForMol is O(N) over
    atoms and can take 20+ seconds on full proteins.

    Returns (features_dict, coords [N, 3]).
    """
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=True, sanitize=False)
    assert mol is not None, f"Failed to parse pocket PDB: {pdb_path}"
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)

    # Filter atoms by distance BEFORE feature extraction (for speed)
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
            # Build a new mol with only the kept atoms
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
