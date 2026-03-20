"""Protein preprocessing: PDB → residue-level and atom-level tensors."""

from pathlib import Path

import torch
from rdkit import Chem

from .ligand import (
    ELEMENT_VOCAB,
    OTHER_ELEMENT_IDX,
    HYBRIDIZATION_MAP,
    OTHER_HYBRID_IDX,
)


# Standard amino acid 3-letter → index mapping (0-19), unknown = 20
AA3_TO_IDX: dict[str, int] = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}
# Non-standard → standard mappings
NONSTANDARD_MAP: dict[str, str] = {
    "MSE": "MET",  # selenomethionine
    "HIP": "HIS", "HIE": "HIS", "HID": "HIS",
    "CYX": "CYS",
    "ASH": "ASP",
    "GLH": "GLU",
}
UNK_IDX = 20
NUM_RES_TYPES = 21


def parse_pocket_pdb(pdb_path: Path) -> dict[str, torch.Tensor] | None:
    """Parse pocket PDB file, extract CA coords and residue types.

    Returns dict with:
        res_coords: [N_res, 3] float32 — CA coordinates
        res_type: [N_res] int64 — amino acid index (0-19, 20=UNK)

    Returns None if no valid residues found.
    """
    ca_coords: list[list[float]] = []
    res_types: list[int] = []
    seen_residues: set[tuple[str, int, str]] = set()  # (chain, resnum, icode)

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            res_name = line[17:20].strip()
            chain = line[21]
            res_num = int(line[22:26].strip())
            icode = line[26].strip()
            # Skip non-standard residues (waters, ligands, etc.)
            mapped = NONSTANDARD_MAP.get(res_name, res_name)
            if mapped not in AA3_TO_IDX and res_name not in NONSTANDARD_MAP:
                continue

            # Deterministic altloc: take first encountered (usually A or blank)
            res_key = (chain, res_num, icode)
            if res_key in seen_residues:
                continue
            seen_residues.add(res_key)

            # Parse coordinates
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            ca_coords.append([x, y, z])
            res_types.append(AA3_TO_IDX.get(mapped, UNK_IDX))

    if len(ca_coords) == 0:
        return None

    return {
        "res_coords": torch.tensor(ca_coords, dtype=torch.float32),
        "res_type": torch.tensor(res_types, dtype=torch.int64),
    }


def parse_pocket_atoms(
    pdb_path: Path,
    ligand_coords: torch.Tensor | None = None,
    cutoff: float = 8.0,
) -> dict[str, torch.Tensor] | None:
    """Parse pocket heavy atoms from PDB using RDKit, with ligand-like features.

    Args:
        pdb_path: Path to pocket PDB file.
        ligand_coords: ``[N_lig, 3]`` ligand coordinates for distance cutoff.
            If None, all protein heavy atoms are returned.
        cutoff: Distance cutoff in Angstroms from any ligand atom.

    Returns dict with:
        patom_coords: ``[N_patom, 3]`` float32 — heavy atom coordinates
        patom_element: ``[N_patom]`` int64 — element index (same vocab as ligand)
        patom_charge: ``[N_patom]`` int8 — formal charge (0=neutral, 1=positive, 2=negative)
        patom_aromatic: ``[N_patom]`` bool
        patom_hybridization: ``[N_patom]`` int8
        patom_in_ring: ``[N_patom]`` bool
        patom_is_backbone: ``[N_patom]`` bool — backbone vs sidechain

    Returns None if parsing fails or no atoms within cutoff.
    """
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=False)
    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Try partial sanitization
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                             | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                             | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                             | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION)
        except Exception:
            pass

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    # Backbone atom names
    backbone_names = {"N", "CA", "C", "O"}

    coords = []
    elements = []
    charges = []
    aromatics = []
    hybridizations = []
    in_rings = []
    is_backbone = []

    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
        elements.append(ELEMENT_VOCAB.get(atom.GetAtomicNum(), OTHER_ELEMENT_IDX))
        fc = atom.GetFormalCharge()
        charges.append(0 if fc == 0 else (1 if fc > 0 else 2))
        aromatics.append(atom.GetIsAromatic())
        hybridizations.append(HYBRIDIZATION_MAP.get(atom.GetHybridization(), OTHER_HYBRID_IDX))
        try:
            ri = atom.IsInRing()
        except Exception:
            ri = False
        in_rings.append(ri)
        pdb_info = atom.GetPDBResidueInfo()
        aname = pdb_info.GetName().strip() if pdb_info is not None else ""
        is_backbone.append(aname in backbone_names)

    if len(coords) == 0:
        return None

    coords_t = torch.tensor(coords, dtype=torch.float32)

    # Distance cutoff from ligand — if ANY atom in a residue is within cutoff,
    # include ALL atoms of that residue.
    if ligand_coords is not None and ligand_coords.shape[0] > 0:
        dists = torch.cdist(coords_t, ligand_coords)  # [N_patom, N_lig]
        min_dists = dists.min(dim=1).values  # [N_patom]
        atom_in_range = min_dists <= cutoff

        # Group atoms by residue: if any atom passes, keep the whole residue
        res_keys = []
        for i in range(n_atoms):
            pdb_info = mol.GetAtomWithIdx(i).GetPDBResidueInfo()
            if pdb_info is not None:
                res_keys.append((pdb_info.GetChainId(), pdb_info.GetResidueNumber(), pdb_info.GetInsertionCode()))
            else:
                res_keys.append(("?", i, " "))

        # Find residues with at least one atom in range
        active_residues: set[tuple] = set()
        for i, in_range in enumerate(atom_in_range.tolist()):
            if in_range:
                active_residues.add(res_keys[i])

        mask = torch.tensor([res_keys[i] in active_residues for i in range(n_atoms)], dtype=torch.bool)
        if not mask.any():
            return None
        indices = mask.nonzero(as_tuple=True)[0]
    else:
        indices = torch.arange(len(coords))

    return {
        "patom_coords": coords_t[indices],
        "patom_element": torch.tensor(elements, dtype=torch.int64)[indices],
        "patom_charge": torch.tensor(charges, dtype=torch.int8)[indices],
        "patom_aromatic": torch.tensor(aromatics, dtype=torch.bool)[indices],
        "patom_hybridization": torch.tensor(hybridizations, dtype=torch.int8)[indices],
        "patom_in_ring": torch.tensor(in_rings, dtype=torch.bool)[indices],
        "patom_is_backbone": torch.tensor(is_backbone, dtype=torch.bool)[indices],
    }
