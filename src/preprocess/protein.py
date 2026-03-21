"""Protein preprocessing: PDB → residue-level and atom-level tensors."""

from pathlib import Path

import torch
from rdkit import Chem

from .ligand import (
    BOND_STEREO_MAP,
    BOND_TYPE_MAP,
    compute_atom_smarts_features,
    OTHER_BOND_IDX,
    CHIRALITY_MAP,
    ELEMENT_VOCAB,
    OTHER_ELEMENT_IDX,
    HYBRIDIZATION_MAP,
    OTHER_HYBRID_IDX,
    _get_atom_valence,
)


# Standard amino acid 3-letter → index mapping (0-19), unknown = 20
AA3_TO_IDX: dict[str, int] = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}
# Non-standard → standard mappings
NONSTANDARD_MAP: dict[str, str] = {
    "MSE": "MET",  # selenomethionine
    "HIP": "HIS",
    "HIE": "HIS",
    "HID": "HIS",
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
        patom_residue_id: ``[N_patom]`` int64 — local residue index
        pbond_index: ``[2, E_dir]`` int64 — directed covalent bonds
        pbond_type: ``[E_dir]`` int8
        pbond_conjugated: ``[E_dir]`` bool
        pbond_in_ring: ``[E_dir]`` bool
        pbond_stereo: ``[E_dir]`` int8
        pca_coords: ``[N_res, 3]`` float32 — CA virtual-node coordinates
        pca_res_type: ``[N_res]`` int64 — amino acid index per residue
        pca_atom_index: ``[N_res]`` int64 — CA atom index in patom arrays

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
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
            )
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
    amino_acids = []
    is_ca = []
    depth_from_ca = []
    degrees = []
    implicit_valences = []
    explicit_valences = []
    num_rings_list = []
    chiralities = []

    # depth_from_ca uses 3D Euclidean distance (not graph distance — 12s+ per complex)

    ring_info = mol.GetRingInfo()
    smarts_features = compute_atom_smarts_features(mol)

    ca_idx_by_res = {}
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        pdb_info = atom.GetPDBResidueInfo()
        if pdb_info is not None:
            aname = pdb_info.GetName().strip()
            if aname == "CA":
                res_key = (
                    pdb_info.GetChainId(),
                    pdb_info.GetResidueNumber(),
                    pdb_info.GetInsertionCode(),
                )
                ca_idx_by_res[res_key] = i

    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
        elements.append(ELEMENT_VOCAB.get(atom.GetAtomicNum(), OTHER_ELEMENT_IDX))
        fc = atom.GetFormalCharge()
        charges.append(fc)
        aromatics.append(atom.GetIsAromatic())
        hybridizations.append(HYBRIDIZATION_MAP.get(atom.GetHybridization(), OTHER_HYBRID_IDX))
        try:
            ri = atom.IsInRing()
        except Exception:
            ri = False
        in_rings.append(ri)

        degrees.append(atom.GetDegree())
        implicit_valences.append(_get_atom_valence(atom, explicit=False))
        explicit_valences.append(_get_atom_valence(atom, explicit=True))
        try:
            nr = ring_info.NumAtomRings(i)
        except Exception:
            nr = 0
        num_rings_list.append(nr)
        chiralities.append(CHIRALITY_MAP.get(atom.GetChiralTag(), 0))

        pdb_info = atom.GetPDBResidueInfo()
        aname = pdb_info.GetName().strip() if pdb_info is not None else ""
        res_name = pdb_info.GetResidueName().strip() if pdb_info is not None else ""
        is_backbone.append(aname in backbone_names)

        mapped = NONSTANDARD_MAP.get(res_name, res_name)
        amino_acids.append(AA3_TO_IDX.get(mapped, UNK_IDX))
        is_ca.append(aname == "CA")

        # 3D Euclidean distance to same-residue CA (Angstroms, rounded)
        ca_dist = 0.0
        if pdb_info is not None:
            res_key = (
                pdb_info.GetChainId(),
                pdb_info.GetResidueNumber(),
                pdb_info.GetInsertionCode(),
            )
            if res_key in ca_idx_by_res:
                ca_idx = ca_idx_by_res[res_key]
                ca_pos = conf.GetAtomPosition(ca_idx)
                ca_dist = ((pos.x - ca_pos.x)**2 + (pos.y - ca_pos.y)**2 + (pos.z - ca_pos.z)**2)**0.5
        depth_from_ca.append(round(ca_dist, 1))

    if len(coords) == 0:
        return None

    coords_t = torch.tensor(coords, dtype=torch.float32)
    res_keys = []

    # Distance cutoff from ligand — if ANY atom in a residue is within cutoff,
    # include ALL atoms of that residue.
    if ligand_coords is not None and ligand_coords.shape[0] > 0:
        dists = torch.cdist(coords_t, ligand_coords)  # [N_patom, N_lig]
        min_dists = dists.min(dim=1).values  # [N_patom]
        atom_in_range = min_dists <= cutoff

        # Group atoms by residue: if any atom passes, keep the whole residue
        for i in range(n_atoms):
            pdb_info = mol.GetAtomWithIdx(i).GetPDBResidueInfo()
            if pdb_info is not None:
                res_keys.append(
                    (
                        pdb_info.GetChainId(),
                        pdb_info.GetResidueNumber(),
                        pdb_info.GetInsertionCode(),
                    )
                )
            else:
                res_keys.append(("?", i, " "))

        # Find residues with at least one atom in range
        active_residues: set[tuple] = set()
        for i, in_range in enumerate(atom_in_range.tolist()):
            if in_range:
                active_residues.add(res_keys[i])

        mask = torch.tensor(
            [res_keys[i] in active_residues for i in range(n_atoms)], dtype=torch.bool
        )
        if not mask.any():
            return None
        indices = mask.nonzero(as_tuple=True)[0]
    else:
        indices = torch.arange(len(coords))
        for i in range(n_atoms):
            pdb_info = mol.GetAtomWithIdx(i).GetPDBResidueInfo()
            if pdb_info is not None:
                res_keys.append(
                    (
                        pdb_info.GetChainId(),
                        pdb_info.GetResidueNumber(),
                        pdb_info.GetInsertionCode(),
                    )
                )
            else:
                res_keys.append(("?", i, " "))

    selected_global = indices.tolist()
    global_to_local = {g: i for i, g in enumerate(selected_global)}

    residue_keys = [res_keys[g] for g in selected_global]
    residue_to_local: dict[tuple, int] = {}
    ordered_residue_keys: list[tuple] = []
    for key in residue_keys:
        if key not in residue_to_local:
            residue_to_local[key] = len(ordered_residue_keys)
            ordered_residue_keys.append(key)

    patom_residue_id = torch.tensor(
        [residue_to_local[key] for key in residue_keys], dtype=torch.int64
    )

    pbond_src: list[int] = []
    pbond_dst: list[int] = []
    pbond_type: list[int] = []
    pbond_conjugated: list[bool] = []
    pbond_in_ring: list[bool] = []
    pbond_stereo: list[int] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in global_to_local or end not in global_to_local:
            continue
        bt = BOND_TYPE_MAP.get(bond.GetBondType(), OTHER_BOND_IDX)
        conj = bond.GetIsConjugated()
        in_ring = bond.IsInRing()
        stereo = BOND_STEREO_MAP.get(bond.GetStereo(), 0)
        lb = global_to_local[begin]
        le = global_to_local[end]
        pbond_src.extend([lb, le])
        pbond_dst.extend([le, lb])
        pbond_type.extend([bt, bt])
        pbond_conjugated.extend([conj, conj])
        pbond_in_ring.extend([in_ring, in_ring])
        pbond_stereo.extend([stereo, stereo])

    pca_coords = []
    pca_res_type = []
    pca_atom_index = []
    for residue_key in ordered_residue_keys:
        global_ca = ca_idx_by_res.get(residue_key)
        if global_ca is None or global_ca not in global_to_local:
            continue
        local_ca = global_to_local[global_ca]
        pca_coords.append(coords[global_ca])
        pca_res_type.append(amino_acids[global_ca])
        pca_atom_index.append(local_ca)

    if not pca_coords:
        return None

    return {
        "patom_coords": coords_t[indices],
        "patom_element": torch.tensor(elements, dtype=torch.int64)[indices],
        "patom_charge": torch.tensor(charges, dtype=torch.int8)[indices],
        "patom_aromatic": torch.tensor(aromatics, dtype=torch.bool)[indices],
        "patom_hybridization": torch.tensor(hybridizations, dtype=torch.int8)[indices],
        "patom_in_ring": torch.tensor(in_rings, dtype=torch.bool)[indices],
        "patom_is_backbone": torch.tensor(is_backbone, dtype=torch.bool)[indices],
        "patom_amino_acid": torch.tensor(amino_acids, dtype=torch.int64)[indices],
        "patom_is_ca": torch.tensor(is_ca, dtype=torch.bool)[indices],
        "patom_ca_dist": torch.tensor(depth_from_ca, dtype=torch.float32)[indices],
        "patom_degree": torch.tensor(degrees, dtype=torch.int8)[indices],
        "patom_implicit_valence": torch.tensor(implicit_valences, dtype=torch.int8)[indices],
        "patom_explicit_valence": torch.tensor(explicit_valences, dtype=torch.int8)[indices],
        "patom_num_rings": torch.tensor(num_rings_list, dtype=torch.int8)[indices],
        "patom_chirality": torch.tensor(chiralities, dtype=torch.int8)[indices],
        "patom_is_donor": smarts_features["atom_is_donor"][indices],
        "patom_is_acceptor": smarts_features["atom_is_acceptor"][indices],
        "patom_is_positive": smarts_features["atom_is_positive"][indices],
        "patom_is_negative": smarts_features["atom_is_negative"][indices],
        "patom_is_hydrophobe": smarts_features["atom_is_hydrophobe"][indices],
        "patom_is_halogen": smarts_features["atom_is_halogen"][indices],
        "patom_residue_id": patom_residue_id,
        "pbond_index": torch.tensor([pbond_src, pbond_dst], dtype=torch.int64),
        "pbond_type": torch.tensor(pbond_type, dtype=torch.int8),
        "pbond_conjugated": torch.tensor(pbond_conjugated, dtype=torch.bool),
        "pbond_in_ring": torch.tensor(pbond_in_ring, dtype=torch.bool),
        "pbond_stereo": torch.tensor(pbond_stereo, dtype=torch.int8),
        "pca_coords": torch.tensor(pca_coords, dtype=torch.float32),
        "pca_res_type": torch.tensor(pca_res_type, dtype=torch.int64),
        "pca_atom_index": torch.tensor(pca_atom_index, dtype=torch.int64),
    }
