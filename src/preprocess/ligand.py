"""Ligand preprocessing: SDF/MOL2 → atom/bond feature tensors."""

from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import rdmolops


# Element vocab: common drug-like elements + OTHER
ELEMENT_VOCAB: dict[int, int] = {
    6: 0,  # C
    7: 1,  # N
    8: 2,  # O
    16: 3,  # S
    15: 4,  # P
    9: 5,  # F
    17: 6,  # Cl
    35: 7,  # Br
    53: 8,  # I
    5: 9,  # B
    14: 10,  # Si
    34: 11,  # Se
}
OTHER_ELEMENT_IDX = 12
NUM_ELEMENTS = 13

# Hybridization vocab
HYBRIDIZATION_MAP: dict[Chem.rdchem.HybridizationType, int] = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
}
OTHER_HYBRID_IDX = 5
NUM_HYBRIDIZATIONS = 6

# Bond type vocab
BOND_TYPE_MAP: dict[Chem.rdchem.BondType, int] = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
OTHER_BOND_IDX = 4
NUM_BOND_TYPES = 5

# Chirality vocab
CHIRALITY_MAP: dict[Chem.rdchem.ChiralType, int] = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER: 3,
}

# Bond stereo vocab
BOND_STEREO_MAP: dict[Chem.rdchem.BondStereo, int] = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3,
    Chem.rdchem.BondStereo.STEREOCIS: 4,
    Chem.rdchem.BondStereo.STEREOTRANS: 5,
}

ATOM_SMARTS = {
    "donor": [
        "[N,O,S;!H0]",
        "[nH]",
    ],
    "acceptor": [
        "[O,S;H0;v2]",
        "[O,S;-]",
        "[nH0,o,s;+0]",
        "[N;H0;!$(N-[O,S,P]=O);!$(N=*);!$(N#*)]",
    ],
    "positive": [
        "[+,+2,+3,+4]",
    ],
    "negative": [
        "[-,-2,-3,-4]",
    ],
    "hydrophobe": [
        "[#6,#16;A;!$(*=[O,N,P,S]);!$(*[#7,#8,#9])]",
        "[c,s]",
    ],
    "halogen": [
        "[#9,#17,#35,#53]",
    ],
}

COMPILED_ATOM_SMARTS = {
    key: [Chem.MolFromSmarts(pattern) for pattern in patterns]
    for key, patterns in ATOM_SMARTS.items()
}


def _get_atom_valence(atom: Chem.Atom, explicit: bool) -> int:
    """Return atom valence while supporting old/new RDKit APIs."""
    try:
        which = Chem.rdchem.ValenceType.EXPLICIT if explicit else Chem.rdchem.ValenceType.IMPLICIT
        return int(atom.GetValence(which))
    except Exception:
        if explicit:
            return int(atom.GetExplicitValence())
        return int(atom.GetImplicitValence())


def compute_atom_smarts_features(mol: Chem.Mol) -> dict[str, torch.Tensor]:
    """Compute boolean atom role features from SMARTS patterns."""
    num_atoms = mol.GetNumAtoms()
    matches = {key: torch.zeros(num_atoms, dtype=torch.bool) for key in COMPILED_ATOM_SMARTS}

    for key, patterns in COMPILED_ATOM_SMARTS.items():
        for pattern in patterns:
            if pattern is None:
                continue
            for match in mol.GetSubstructMatches(pattern, uniquify=True):
                for atom_idx in match:
                    matches[key][atom_idx] = True

    return {
        "atom_is_donor": matches["donor"],
        "atom_is_acceptor": matches["acceptor"],
        "atom_is_positive": matches["positive"],
        "atom_is_negative": matches["negative"],
        "atom_is_hydrophobe": matches["hydrophobe"],
        "atom_is_halogen": matches["halogen"],
    }


def load_molecule(sdf_path: Path, mol2_path: Path | None = None) -> tuple[Chem.Mol | None, bool]:
    """Load molecule from SDF, fallback to MOL2.

    Returns (mol, used_mol2_fallback).
    mol is sanitized with H removed, or None on failure.
    """
    mol = _try_load_sdf(sdf_path)
    used_fallback = False

    if mol is None and mol2_path is not None and mol2_path.exists():
        mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False)
        if mol is not None:
            mol = _try_sanitize(mol)
            used_fallback = True

    if mol is None:
        return None, used_fallback

    # Remove hydrogens, keep only heavy atoms
    mol = Chem.RemoveHs(mol)

    # Keep largest connected component
    mol = _keep_largest_component(mol)

    # Verify 3D conformer
    if mol.GetNumConformers() == 0:
        return None, used_fallback
    conf = mol.GetConformer()
    if not conf.Is3D():
        return None, used_fallback

    return mol, used_fallback


def featurize_ligand(mol: Chem.Mol) -> dict[str, torch.Tensor] | None:
    """Extract minimal atom/bond features from RDKit mol.

    Returns dict with:
        atom_coords: [N_atom, 3] float32
        atom_element: [N_atom] int64
        atom_charge: [N_atom] int8
        atom_aromatic: [N_atom] bool
        atom_hybridization: [N_atom] int8
        atom_in_ring: [N_atom] bool
        bond_index: [2, E_dir] int64 (directed, E_dir = 2 * N_bonds)
        bond_type: [E_dir] int8
        bond_conjugated: [E_dir] bool
        bond_in_ring: [E_dir] bool

    Returns None if molecule has < 2 atoms.
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms < 2:
        return None

    conf = mol.GetConformer()
    smarts_features = compute_atom_smarts_features(mol)

    # Atom features
    coords = []
    elements = []
    charges = []
    aromatics = []
    hybridizations = []
    in_rings = []
    degrees = []
    implicit_valences = []
    explicit_valences = []
    num_rings_list = []
    chiralities = []

    ring_info = mol.GetRingInfo()

    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)

        coords.append([pos.x, pos.y, pos.z])
        elements.append(ELEMENT_VOCAB.get(atom.GetAtomicNum(), OTHER_ELEMENT_IDX))
        charges.append(atom.GetFormalCharge())
        aromatics.append(atom.GetIsAromatic())
        hybridizations.append(HYBRIDIZATION_MAP.get(atom.GetHybridization(), OTHER_HYBRID_IDX))
        in_rings.append(ring_info.NumAtomRings(i) > 0)
        degrees.append(atom.GetDegree())
        implicit_valences.append(_get_atom_valence(atom, explicit=False))
        explicit_valences.append(_get_atom_valence(atom, explicit=True))
        num_rings_list.append(ring_info.NumAtomRings(i))
        chiralities.append(CHIRALITY_MAP.get(atom.GetChiralTag(), 0))

    # Bond features (directed: each bond stored twice)
    src_list = []
    dst_list = []
    bond_types = []
    bond_conj = []
    bond_rings = []
    bond_stereos = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = BOND_TYPE_MAP.get(bond.GetBondType(), OTHER_BOND_IDX)
        conj = bond.GetIsConjugated()
        in_ring = bond.IsInRing()
        stereo = BOND_STEREO_MAP.get(bond.GetStereo(), 0)

        # Forward
        src_list.append(i)
        dst_list.append(j)
        bond_types.append(bt)
        bond_conj.append(conj)
        bond_rings.append(in_ring)
        bond_stereos.append(stereo)

        # Backward
        src_list.append(j)
        dst_list.append(i)
        bond_types.append(bt)
        bond_conj.append(conj)
        bond_rings.append(in_ring)
        bond_stereos.append(stereo)

    coords_t = torch.tensor(coords, dtype=torch.float32)

    # Check for NaN/Inf
    if not torch.isfinite(coords_t).all():
        return None

    return {
        "atom_coords": coords_t,
        "atom_element": torch.tensor(elements, dtype=torch.int64),
        "atom_charge": torch.tensor(charges, dtype=torch.int8),
        "atom_aromatic": torch.tensor(aromatics, dtype=torch.bool),
        "atom_hybridization": torch.tensor(hybridizations, dtype=torch.int8),
        "atom_in_ring": torch.tensor(in_rings, dtype=torch.bool),
        "atom_degree": torch.tensor(degrees, dtype=torch.int8),
        "atom_implicit_valence": torch.tensor(implicit_valences, dtype=torch.int8),
        "atom_explicit_valence": torch.tensor(explicit_valences, dtype=torch.int8),
        "atom_num_rings": torch.tensor(num_rings_list, dtype=torch.int8),
        "atom_chirality": torch.tensor(chiralities, dtype=torch.int8),
        **smarts_features,
        "bond_index": torch.tensor([src_list, dst_list], dtype=torch.int64),
        "bond_type": torch.tensor(bond_types, dtype=torch.int8),
        "bond_conjugated": torch.tensor(bond_conj, dtype=torch.bool),
        "bond_in_ring": torch.tensor(bond_rings, dtype=torch.bool),
        "bond_stereo": torch.tensor(bond_stereos, dtype=torch.int8),
    }


def _try_load_sdf(path: Path) -> Chem.Mol | None:
    """Try loading SDF with sanitization."""
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    for mol in supplier:
        if mol is not None:
            return _try_sanitize(mol)
    return None


def _try_sanitize(mol: Chem.Mol) -> Chem.Mol | None:
    """Try sanitizing mol, retry with relaxed settings on failure."""
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        pass

    # Retry: skip problematic sanitization steps
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
        return mol
    except Exception:
        return None


def _keep_largest_component(mol: Chem.Mol) -> Chem.Mol:
    """Keep only the largest connected component."""
    frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) <= 1:
        return mol
    return max(frags, key=lambda m: m.GetNumAtoms())
