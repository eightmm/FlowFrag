"""Static unified graph construction for protein-ligand complexes."""

import torch

from .ligand import OTHER_BOND_IDX, OTHER_ELEMENT_IDX, OTHER_HYBRID_IDX
from .protein import UNK_IDX


EDGE_TYPES = {
    "ligand_bond": 0,
    "ligand_tri": 1,
    "ligand_cut": 2,
    "ligand_atom_frag": 3,
    "ligand_frag_frag": 4,
    "protein_bond": 5,
    "protein_atom_ca": 6,
    "protein_ca_ca": 7,
    "protein_ca_frag": 8,
}

NODE_TYPES = {
    "ligand_atom": 0,
    "ligand_dummy": 1,
    "ligand_fragment": 2,
    "protein_atom": 3,
    "protein_ca": 4,
}


def build_static_complex_graph(
    lig_data: dict[str, torch.Tensor],
    patom_data: dict[str, torch.Tensor],
    pca_cutoff: float = 18.0,
) -> dict[str, torch.Tensor]:
    """Build a unified static graph for one complex.

    Only protein-ligand interaction edges are intentionally left out so they can
    be rebuilt dynamically from the current diffusion state.
    """
    lig_atom_coords = lig_data["atom_coords"]
    frag_coords = lig_data["frag_centers"]
    prot_atom_coords = patom_data["patom_coords"]
    pca_coords = patom_data["pca_coords"]

    n_lig_atom = lig_atom_coords.shape[0]
    n_frag = frag_coords.shape[0]
    n_prot_atom = prot_atom_coords.shape[0]
    n_pca = pca_coords.shape[0]
    pca_residue_id = patom_data["patom_residue_id"][patom_data["pca_atom_index"]]

    frag_offset = n_lig_atom
    prot_atom_offset = frag_offset + n_frag
    pca_offset = prot_atom_offset + n_prot_atom

    node_coords = torch.cat([lig_atom_coords, frag_coords, prot_atom_coords, pca_coords], dim=0)
    total_nodes = node_coords.shape[0]

    ligand_is_dummy = lig_data.get("is_dummy", torch.zeros(n_lig_atom, dtype=torch.bool))
    ligand_node_type = torch.full((n_lig_atom,), NODE_TYPES["ligand_atom"], dtype=torch.int64)
    ligand_node_type[ligand_is_dummy] = NODE_TYPES["ligand_dummy"]

    frag_node_type = torch.full((n_frag,), NODE_TYPES["ligand_fragment"], dtype=torch.int64)
    prot_atom_node_type = torch.full((n_prot_atom,), NODE_TYPES["protein_atom"], dtype=torch.int64)
    pca_node_type = torch.full((n_pca,), NODE_TYPES["protein_ca"], dtype=torch.int64)
    node_type = torch.cat(
        [ligand_node_type, frag_node_type, prot_atom_node_type, pca_node_type], dim=0
    )

    node_element = torch.cat(
        [
            lig_data["atom_element"],
            torch.full((n_frag,), OTHER_ELEMENT_IDX, dtype=torch.int64),
            patom_data["patom_element"],
            torch.full((n_pca,), OTHER_ELEMENT_IDX, dtype=torch.int64),
        ]
    )
    node_charge = torch.cat(
        [
            lig_data["atom_charge"],
            torch.zeros(n_frag, dtype=torch.int8),
            patom_data["patom_charge"],
            torch.zeros(n_pca, dtype=torch.int8),
        ]
    )
    node_aromatic = torch.cat(
        [
            lig_data["atom_aromatic"],
            torch.zeros(n_frag, dtype=torch.bool),
            patom_data["patom_aromatic"],
            torch.zeros(n_pca, dtype=torch.bool),
        ]
    )
    node_hybridization = torch.cat(
        [
            lig_data["atom_hybridization"],
            torch.full((n_frag,), OTHER_HYBRID_IDX, dtype=torch.int8),
            patom_data["patom_hybridization"],
            torch.full((n_pca,), OTHER_HYBRID_IDX, dtype=torch.int8),
        ]
    )
    node_in_ring = torch.cat(
        [
            lig_data["atom_in_ring"],
            torch.zeros(n_frag, dtype=torch.bool),
            patom_data["patom_in_ring"],
            torch.zeros(n_pca, dtype=torch.bool),
        ]
    )
    node_degree = torch.cat(
        [
            lig_data["atom_degree"],
            torch.zeros(n_frag, dtype=torch.int8),
            patom_data["patom_degree"],
            torch.zeros(n_pca, dtype=torch.int8),
        ]
    )
    node_implicit_valence = torch.cat(
        [
            lig_data["atom_implicit_valence"],
            torch.zeros(n_frag, dtype=torch.int8),
            patom_data["patom_implicit_valence"],
            torch.zeros(n_pca, dtype=torch.int8),
        ]
    )
    node_explicit_valence = torch.cat(
        [
            lig_data["atom_explicit_valence"],
            torch.zeros(n_frag, dtype=torch.int8),
            patom_data["patom_explicit_valence"],
            torch.zeros(n_pca, dtype=torch.int8),
        ]
    )
    node_num_rings = torch.cat(
        [
            lig_data["atom_num_rings"],
            torch.zeros(n_frag, dtype=torch.int8),
            patom_data["patom_num_rings"],
            torch.zeros(n_pca, dtype=torch.int8),
        ]
    )
    node_chirality = torch.cat(
        [
            lig_data["atom_chirality"],
            torch.zeros(n_frag, dtype=torch.int8),
            patom_data["patom_chirality"],
            torch.zeros(n_pca, dtype=torch.int8),
        ]
    )
    node_is_donor = torch.cat(
        [
            lig_data["atom_is_donor"],
            torch.zeros(n_frag, dtype=torch.bool),
            patom_data["patom_is_donor"],
            torch.zeros(n_pca, dtype=torch.bool),
        ]
    )
    node_is_acceptor = torch.cat(
        [
            lig_data["atom_is_acceptor"],
            torch.zeros(n_frag, dtype=torch.bool),
            patom_data["patom_is_acceptor"],
            torch.zeros(n_pca, dtype=torch.bool),
        ]
    )
    node_is_positive = torch.cat(
        [
            lig_data["atom_is_positive"],
            torch.zeros(n_frag, dtype=torch.bool),
            patom_data["patom_is_positive"],
            torch.zeros(n_pca, dtype=torch.bool),
        ]
    )
    node_is_negative = torch.cat(
        [
            lig_data["atom_is_negative"],
            torch.zeros(n_frag, dtype=torch.bool),
            patom_data["patom_is_negative"],
            torch.zeros(n_pca, dtype=torch.bool),
        ]
    )
    node_is_hydrophobe = torch.cat(
        [
            lig_data["atom_is_hydrophobe"],
            torch.zeros(n_frag, dtype=torch.bool),
            patom_data["patom_is_hydrophobe"],
            torch.zeros(n_pca, dtype=torch.bool),
        ]
    )
    node_is_halogen = torch.cat(
        [
            lig_data["atom_is_halogen"],
            torch.zeros(n_frag, dtype=torch.bool),
            patom_data["patom_is_halogen"],
            torch.zeros(n_pca, dtype=torch.bool),
        ]
    )
    node_amino_acid = torch.cat(
        [
            torch.full((n_lig_atom,), UNK_IDX, dtype=torch.int64),
            torch.full((n_frag,), UNK_IDX, dtype=torch.int64),
            patom_data["patom_amino_acid"],
            patom_data["pca_res_type"],
        ]
    )
    node_is_backbone = torch.cat(
        [
            torch.zeros(n_lig_atom, dtype=torch.bool),
            torch.zeros(n_frag, dtype=torch.bool),
            patom_data["patom_is_backbone"],
            torch.zeros(n_pca, dtype=torch.bool),
        ]
    )
    node_is_ca = torch.cat(
        [
            torch.zeros(n_lig_atom, dtype=torch.bool),
            torch.zeros(n_frag, dtype=torch.bool),
            patom_data["patom_is_ca"],
            torch.ones(n_pca, dtype=torch.bool),
        ]
    )
    node_ca_dist = torch.cat(
        [
            torch.zeros(n_lig_atom, dtype=torch.int64),
            torch.zeros(n_frag, dtype=torch.int64),
            patom_data["patom_ca_dist"],
            torch.zeros(n_pca, dtype=torch.int64),
        ]
    )
    node_fragment_id = torch.cat(
        [
            lig_data["fragment_id"],
            torch.arange(n_frag, dtype=torch.int64),
            torch.full((n_prot_atom,), -1, dtype=torch.int64),
            torch.full((n_pca,), -1, dtype=torch.int64),
        ]
    )
    node_residue_id = torch.cat(
        [
            torch.full((n_lig_atom,), -1, dtype=torch.int64),
            torch.full((n_frag,), -1, dtype=torch.int64),
            patom_data["patom_residue_id"],
            pca_residue_id,
        ]
    )
    node_is_dummy = torch.cat(
        [
            ligand_is_dummy,
            torch.zeros(n_frag, dtype=torch.bool),
            torch.zeros(n_prot_atom, dtype=torch.bool),
            torch.zeros(n_pca, dtype=torch.bool),
        ]
    )
    node_is_virtual = torch.cat(
        [
            torch.zeros(n_lig_atom, dtype=torch.bool),
            torch.ones(n_frag, dtype=torch.bool),
            torch.zeros(n_prot_atom, dtype=torch.bool),
            torch.ones(n_pca, dtype=torch.bool),
        ]
    )

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_type: list[int] = []
    edge_ref_dist: list[float] = []
    edge_bond_type: list[int] = []
    edge_bond_conjugated: list[bool] = []
    edge_bond_in_ring: list[bool] = []
    edge_bond_stereo: list[int] = []

    def add_edges(
        src: torch.Tensor,
        dst: torch.Tensor,
        etype: int,
        ref_dist: torch.Tensor | None = None,
        bond_type: torch.Tensor | None = None,
        bond_conj: torch.Tensor | None = None,
        bond_ring: torch.Tensor | None = None,
        bond_stereo: torch.Tensor | None = None,
    ) -> None:
        if src.numel() == 0:
            return
        if ref_dist is None:
            ref_dist = torch.linalg.vector_norm(node_coords[src] - node_coords[dst], dim=-1)
        if bond_type is None:
            bond_type = torch.full((src.shape[0],), -1, dtype=torch.int8)
        if bond_conj is None:
            bond_conj = torch.zeros(src.shape[0], dtype=torch.bool)
        if bond_ring is None:
            bond_ring = torch.zeros(src.shape[0], dtype=torch.bool)
        if bond_stereo is None:
            bond_stereo = torch.full((src.shape[0],), -1, dtype=torch.int8)
        assert ref_dist is not None
        assert bond_type is not None
        assert bond_conj is not None
        assert bond_ring is not None
        assert bond_stereo is not None

        edge_src.extend(src.tolist())
        edge_dst.extend(dst.tolist())
        edge_type.extend([etype] * src.shape[0])
        edge_ref_dist.extend(ref_dist.tolist())
        edge_bond_type.extend(bond_type.tolist())
        edge_bond_conjugated.extend(bond_conj.tolist())
        edge_bond_in_ring.extend(bond_ring.tolist())
        edge_bond_stereo.extend(bond_stereo.tolist())

    # Ligand chemical graph.
    add_edges(
        lig_data["bond_index"][0],
        lig_data["bond_index"][1],
        EDGE_TYPES["ligand_bond"],
        bond_type=lig_data["bond_type"],
        bond_conj=lig_data["bond_conjugated"],
        bond_ring=lig_data["bond_in_ring"],
        bond_stereo=lig_data["bond_stereo"],
    )

    # Ligand triangulation edges.
    add_edges(
        lig_data["tri_edge_index"][0],
        lig_data["tri_edge_index"][1],
        EDGE_TYPES["ligand_tri"],
        ref_dist=lig_data["tri_edge_ref_dist"],
    )

    # Explicit cut-bond edges.
    cut_index = lig_data["cut_bond_index"]
    if cut_index.numel() > 0:
        cut_src = torch.cat([cut_index[0], cut_index[1]])
        cut_dst = torch.cat([cut_index[1], cut_index[0]])
        add_edges(cut_src, cut_dst, EDGE_TYPES["ligand_cut"])

    # Atom-fragment ownership edges.
    atom_idx = torch.arange(n_lig_atom, dtype=torch.int64)
    frag_idx = frag_offset + lig_data["fragment_id"].to(torch.int64)
    add_edges(atom_idx, frag_idx, EDGE_TYPES["ligand_atom_frag"])
    add_edges(frag_idx, atom_idx, EDGE_TYPES["ligand_atom_frag"])

    # Fragment-fragment full graph.
    if n_frag > 1:
        frag_src, frag_dst = [], []
        for i in range(n_frag):
            for j in range(n_frag):
                if i != j:
                    frag_src.append(frag_offset + i)
                    frag_dst.append(frag_offset + j)
        add_edges(
            torch.tensor(frag_src, dtype=torch.int64),
            torch.tensor(frag_dst, dtype=torch.int64),
            EDGE_TYPES["ligand_frag_frag"],
        )

    # Protein covalent graph.
    add_edges(
        prot_atom_offset + patom_data["pbond_index"][0],
        prot_atom_offset + patom_data["pbond_index"][1],
        EDGE_TYPES["protein_bond"],
        bond_type=patom_data["pbond_type"],
        bond_conj=patom_data["pbond_conjugated"],
        bond_ring=patom_data["pbond_in_ring"],
        bond_stereo=patom_data["pbond_stereo"],
    )

    # Protein atom-CA hierarchy edges.
    residue_lookup = {
        residue_id: local_pca for local_pca, residue_id in enumerate(pca_residue_id.tolist())
    }
    atom_src = []
    atom_dst = []
    for local_atom, residue_id in enumerate(patom_data["patom_residue_id"].tolist()):
        local_pca = residue_lookup.get(residue_id)
        if local_pca is None:
            continue
        atom_src.append(prot_atom_offset + local_atom)
        atom_dst.append(pca_offset + local_pca)
    if atom_src:
        prot_src = torch.tensor(atom_src, dtype=torch.int64)
        prot_dst = torch.tensor(atom_dst, dtype=torch.int64)
        add_edges(prot_src, prot_dst, EDGE_TYPES["protein_atom_ca"])
        add_edges(prot_dst, prot_src, EDGE_TYPES["protein_atom_ca"])

    # Protein CA-CA graph.
    if n_pca > 1:
        ca_dmat = torch.cdist(pca_coords, pca_coords)
        ca_mask = (ca_dmat <= pca_cutoff) & (~torch.eye(n_pca, dtype=torch.bool))
        ca_src, ca_dst = ca_mask.nonzero(as_tuple=True)
        add_edges(
            pca_offset + ca_src,
            pca_offset + ca_dst,
            EDGE_TYPES["protein_ca_ca"],
            ref_dist=ca_dmat[ca_src, ca_dst],
        )

    # Protein CA-fragment global edges.
    if n_pca > 0 and n_frag > 0:
        cf_src, cf_dst = [], []
        for i in range(n_pca):
            for j in range(n_frag):
                cf_src.append(pca_offset + i)
                cf_dst.append(frag_offset + j)
        cf_src_t = torch.tensor(cf_src, dtype=torch.int64)
        cf_dst_t = torch.tensor(cf_dst, dtype=torch.int64)
        add_edges(cf_src_t, cf_dst_t, EDGE_TYPES["protein_ca_frag"])
        add_edges(cf_dst_t, cf_src_t, EDGE_TYPES["protein_ca_frag"])

    return {
        "node_coords": node_coords,
        "node_type": node_type,
        "node_element": node_element,
        "node_charge": node_charge,
        "node_aromatic": node_aromatic,
        "node_hybridization": node_hybridization,
        "node_in_ring": node_in_ring,
        "node_degree": node_degree,
        "node_implicit_valence": node_implicit_valence,
        "node_explicit_valence": node_explicit_valence,
        "node_num_rings": node_num_rings,
        "node_chirality": node_chirality,
        "node_is_donor": node_is_donor,
        "node_is_acceptor": node_is_acceptor,
        "node_is_positive": node_is_positive,
        "node_is_negative": node_is_negative,
        "node_is_hydrophobe": node_is_hydrophobe,
        "node_is_halogen": node_is_halogen,
        "node_amino_acid": node_amino_acid,
        "node_is_backbone": node_is_backbone,
        "node_is_ca": node_is_ca,
        "node_ca_dist": node_ca_dist,
        "node_fragment_id": node_fragment_id,
        "node_residue_id": node_residue_id,
        "node_is_dummy": node_is_dummy,
        "node_is_virtual": node_is_virtual,
        "edge_index": torch.tensor([edge_src, edge_dst], dtype=torch.int64),
        "edge_type": torch.tensor(edge_type, dtype=torch.int8),
        "edge_ref_dist": torch.tensor(edge_ref_dist, dtype=torch.float32),
        "edge_bond_type": torch.tensor(edge_bond_type, dtype=torch.int8),
        "edge_bond_conjugated": torch.tensor(edge_bond_conjugated, dtype=torch.bool),
        "edge_bond_in_ring": torch.tensor(edge_bond_in_ring, dtype=torch.bool),
        "edge_bond_stereo": torch.tensor(edge_bond_stereo, dtype=torch.int8),
        "num_nodes": torch.tensor(total_nodes, dtype=torch.int64),
        "num_lig_atom": torch.tensor(n_lig_atom, dtype=torch.int64),
        "num_lig_frag": torch.tensor(n_frag, dtype=torch.int64),
        "num_prot_atom": torch.tensor(n_prot_atom, dtype=torch.int64),
        "num_prot_ca": torch.tensor(n_pca, dtype=torch.int64),
        "lig_atom_slice": torch.tensor([0, n_lig_atom], dtype=torch.int64),
        "lig_frag_slice": torch.tensor([frag_offset, prot_atom_offset], dtype=torch.int64),
        "prot_atom_slice": torch.tensor([prot_atom_offset, pca_offset], dtype=torch.int64),
        "prot_ca_slice": torch.tensor([pca_offset, total_nodes], dtype=torch.int64),
    }
