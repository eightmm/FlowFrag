"""Protein preprocessing: PDB → residue-level tensors."""

from pathlib import Path

import torch


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
