"""Fragment decomposition: rotatable bond cuts → rigid fragments.

Each fragment gets:
- T_frag (centroid) and R_frag = I (identity, by construction)
- local_coords relative to centroid
- Single-atom fragments: valid for translation, omega fixed to 0
"""

import torch
from rdkit import Chem


def decompose_fragments(mol: Chem.Mol, atom_coords: torch.Tensor) -> dict | None:
    """Decompose molecule into rigid fragments via rotatable bond cuts.

    Args:
        mol: RDKit molecule (heavy atoms only, sanitized)
        atom_coords: [N_atom, 3] crystal coordinates

    Returns dict with:
        fragment_id: [N_atom] int64 — fragment assignment per atom
        frag_centers: [N_frag, 3] float32 — geometric centroids
        frag_local_coords: [N_atom, 3] float32 — coords in fragment frame
        frag_sizes: [N_frag] int64 — number of atoms per fragment
        n_frags: int

    Returns None on failure.
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms < 2:
        return None

    # Find rotatable bonds to cut
    rot_bonds = _get_rotatable_bonds(mol)

    # Assign atoms to fragments via connected components after cutting
    fragment_id = _assign_fragments(mol, rot_bonds, n_atoms)

    # Reindex fragments to be contiguous 0..N_frag-1
    unique_frags, fragment_id = torch.unique(fragment_id, return_inverse=True)
    n_frags = len(unique_frags)

    # Compute per-fragment geometric centroids and local coords
    frag_centers = torch.zeros(n_frags, 3, dtype=torch.float32)
    frag_sizes = torch.zeros(n_frags, dtype=torch.int64)

    for f in range(n_frags):
        mask = fragment_id == f
        frag_coords = atom_coords[mask]
        frag_centers[f] = frag_coords.mean(dim=0)
        frag_sizes[f] = mask.sum()

    # Local coords: relative to fragment centroid
    # Reconstruction: x_global = R_frag @ x_local + T_frag
    # At crystal pose: R_frag = I, T_frag = centroid
    frag_local_coords = atom_coords - frag_centers[fragment_id]

    # Validate
    if not torch.isfinite(frag_centers).all():
        return None
    if not torch.isfinite(frag_local_coords).all():
        return None

    # Triangulation: cross-fragment edges near cut bonds
    tri_data = _build_triangulation_edges(mol, rot_bonds, fragment_id, atom_coords)

    return {
        "fragment_id": fragment_id,
        "frag_centers": frag_centers,
        "frag_local_coords": frag_local_coords,
        "frag_sizes": frag_sizes,
        "n_frags": n_frags,
        "rot_bonds": rot_bonds,
        **tri_data,
    }


def _build_triangulation_edges(
    mol: Chem.Mol,
    rot_bonds: list[tuple[int, int]],
    fragment_id: torch.Tensor,
    atom_coords: torch.Tensor,
) -> dict:
    """Build cross-fragment triangulation edges around cut bonds.

    For each cut bond (a, b) with a in frag_A, b in frag_B:
    - Collect 1-hop neighbors of a within frag_A (including a)
    - Collect 1-hop neighbors of b within frag_B (including b)
    - Add all cross-fragment pairs as triangulation edges
    - Store crystal reference distances for each pair

    Also builds fragment adjacency from cut bond topology.
    """
    tri_src, tri_dst = [], []
    ref_dists = []
    frag_adj_src, frag_adj_dst = [], []

    for a, b in rot_bonds:
        fa = fragment_id[a].item()
        fb = fragment_id[b].item()
        if fa == fb:
            continue  # shouldn't happen, but safety check

        # Fragment adjacency (bidirectional)
        frag_adj_src.extend([fa, fb])
        frag_adj_dst.extend([fb, fa])

        # 1-hop neighbors of a within frag_A (+ a itself)
        left = {a}
        for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
            ni = nbr.GetIdx()
            if fragment_id[ni].item() == fa:
                left.add(ni)

        # 1-hop neighbors of b within frag_B (+ b itself)
        right = {b}
        for nbr in mol.GetAtomWithIdx(b).GetNeighbors():
            ni = nbr.GetIdx()
            if fragment_id[ni].item() == fb:
                right.add(ni)

        # Cross-fragment pairs (bidirectional)
        for i in left:
            for j in right:
                tri_src.extend([i, j])
                tri_dst.extend([j, i])
                d = torch.linalg.vector_norm(atom_coords[i] - atom_coords[j]).item()
                ref_dists.extend([d, d])

    # Build tensors
    if tri_src:
        tri_edge_index = torch.tensor([tri_src, tri_dst], dtype=torch.int64)
        tri_edge_ref_dist = torch.tensor(ref_dists, dtype=torch.float32)
    else:
        tri_edge_index = torch.zeros(2, 0, dtype=torch.int64)
        tri_edge_ref_dist = torch.zeros(0, dtype=torch.float32)

    if frag_adj_src:
        # Deduplicate fragment adjacency
        adj_pairs = set()
        dedup_src, dedup_dst = [], []
        for s, d in zip(frag_adj_src, frag_adj_dst):
            if (s, d) not in adj_pairs:
                adj_pairs.add((s, d))
                dedup_src.append(s)
                dedup_dst.append(d)
        fragment_adj_index = torch.tensor([dedup_src, dedup_dst], dtype=torch.int64)
    else:
        fragment_adj_index = torch.zeros(2, 0, dtype=torch.int64)

    return {
        "cut_bond_index": torch.tensor(rot_bonds, dtype=torch.int64).T if rot_bonds else torch.zeros(2, 0, dtype=torch.int64),
        "tri_edge_index": tri_edge_index,
        "tri_edge_ref_dist": tri_edge_ref_dist,
        "fragment_adj_index": fragment_adj_index,
    }


def add_dummy_atoms(
    frag_data: dict,
    atom_coords: torch.Tensor,
    atom_feats: dict,
    rot_bonds: list[tuple[int, int]],
) -> dict:
    """Add dummy atoms at cut-bond boundaries (SigmaDock-style).

    For each cut bond (a in frag_A, b in frag_B):
    - Add a copy of b into frag_A (dummy)
    - Add a copy of a into frag_B (dummy)

    Dummy atoms have identical features to their real counterparts but belong
    to the adjacent fragment.  Centroids are NOT recomputed (stay from real
    atoms only).

    Returns updated frag_data with extra keys:
    - ``is_dummy``: ``[N_atom_ext]`` bool — True for dummy atoms
    - ``dummy_to_real``: ``[N_dummy, 2]`` — (dummy_idx, real_idx) pairs
    - All existing atom-level tensors extended with dummy entries
    """
    fragment_id = frag_data["fragment_id"]
    frag_centers = frag_data["frag_centers"]
    n_real = atom_coords.shape[0]

    dummy_coords = []
    dummy_frag_ids = []
    dummy_to_real_pairs = []  # (dummy_idx, real_idx)
    dummy_feat_indices = []  # index into real atoms to copy features from

    idx = n_real  # next available atom index
    for a, b in rot_bonds:
        fa = fragment_id[a].item()
        fb = fragment_id[b].item()
        if fa == fb:
            continue

        # Add b as dummy in frag_A
        dummy_coords.append(atom_coords[b])
        dummy_frag_ids.append(fa)
        dummy_to_real_pairs.append((idx, b))
        dummy_feat_indices.append(b)
        idx += 1

        # Add a as dummy in frag_B
        dummy_coords.append(atom_coords[a])
        dummy_frag_ids.append(fb)
        dummy_to_real_pairs.append((idx, a))
        dummy_feat_indices.append(a)
        idx += 1

    n_dummy = len(dummy_coords)
    if n_dummy == 0:
        # No cut bonds → no dummies needed
        frag_data["is_dummy"] = torch.zeros(n_real, dtype=torch.bool)
        frag_data["dummy_to_real"] = torch.zeros(0, 2, dtype=torch.int64)
        return frag_data

    # Extended coordinates
    dummy_coords_t = torch.stack(dummy_coords)  # [N_dummy, 3]
    all_coords = torch.cat([atom_coords, dummy_coords_t])  # [N_ext, 3]

    # Extended fragment IDs
    dummy_frag_t = torch.tensor(dummy_frag_ids, dtype=torch.int64)
    ext_frag_id = torch.cat([fragment_id, dummy_frag_t])

    # Local coords for dummies (relative to their assigned fragment centroid)
    dummy_local = dummy_coords_t - frag_centers[dummy_frag_t]
    ext_local = torch.cat([frag_data["frag_local_coords"], dummy_local])

    # Update frag_sizes to include dummies
    ext_sizes = frag_data["frag_sizes"].clone()
    for fid in dummy_frag_ids:
        ext_sizes[fid] += 1

    # is_dummy mask
    is_dummy = torch.zeros(n_real + n_dummy, dtype=torch.bool)
    is_dummy[n_real:] = True

    # dummy_to_real mapping
    dummy_to_real = torch.tensor(dummy_to_real_pairs, dtype=torch.int64)

    # Extend atom features by copying from real counterparts
    feat_idx = torch.tensor(dummy_feat_indices, dtype=torch.int64)
    for key in ("atom_element", "atom_charge", "atom_aromatic", "atom_hybridization", "atom_in_ring"):
        if key in atom_feats:
            orig = atom_feats[key]
            frag_data[key] = torch.cat([orig, orig[feat_idx]])

    frag_data["atom_coords"] = all_coords
    frag_data["fragment_id"] = ext_frag_id
    frag_data["frag_local_coords"] = ext_local
    frag_data["frag_sizes"] = ext_sizes
    frag_data["is_dummy"] = is_dummy
    frag_data["dummy_to_real"] = dummy_to_real
    # frag_centers unchanged — computed from real atoms only

    return frag_data


def _get_rotatable_bonds(mol: Chem.Mol) -> list[tuple[int, int]]:
    """Get rotatable bonds to cut for fragment decomposition.

    Rules:
    - Non-ring single bonds that are rotatable
    - Exclude bonds inside ring systems
    - Exclude bonds to terminal atoms (degree 1) — they stay with parent
    """
    rot_bonds = []
    for bond in mol.GetBonds():
        if not _is_cuttable_bond(bond):
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rot_bonds.append((i, j))

    return rot_bonds


def _is_cuttable_bond(bond: Chem.Bond) -> bool:
    """Determine if a bond should be cut for fragmentation.

    A bond is cuttable if:
    1. It's a single bond
    2. It's not in a ring
    3. It's not to a terminal heavy atom (degree 1)
    4. Neither end is in an aromatic system connected only by this bond
    """
    # Must be single bond
    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
        return False

    # Must not be in a ring
    if bond.IsInRing():
        return False

    # Must not be conjugated (keeps amide bonds intact)
    if bond.GetIsConjugated():
        return False

    # Neither atom should be terminal (degree 1 in heavy-atom graph)
    begin = bond.GetBeginAtom()
    end = bond.GetEndAtom()
    if begin.GetDegree() <= 1 or end.GetDegree() <= 1:
        return False

    return True


def _assign_fragments(
    mol: Chem.Mol,
    rot_bonds: list[tuple[int, int]],
    n_atoms: int,
) -> torch.Tensor:
    """Assign atoms to fragments using connected components after cutting bonds.

    Uses union-find for efficient component assignment.
    """
    # Build bond set to exclude
    cut_set = set()
    for i, j in rot_bonds:
        cut_set.add((min(i, j), max(i, j)))

    # Union-Find
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Union atoms connected by non-cut bonds
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        key = (min(i, j), max(i, j))
        if key not in cut_set:
            union(i, j)

    # Assign fragment IDs
    fragment_id = torch.tensor([find(i) for i in range(n_atoms)], dtype=torch.int64)
    return fragment_id
