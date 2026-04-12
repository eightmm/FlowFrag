# Dataset Format

## Raw Data

Each protein-ligand complex is stored as a pair of files:

```
raw_data/<pdb_id>/
├── <pdb_id>_pocket.pdb    # Binding pocket residues
└── <pdb_id>_ligand.sdf    # Ligand structure (or .mol2)
```

## Building the Dataset

```bash
uv run python scripts/build_fragment_flow_dataset.py \
    --raw_dir /path/to/raw_data \
    --out_dir data/processed \
    --workers 8 \
    --dummy              # Add dummy atoms at cut bonds (optional)
```

## Processed Output

Per complex, preprocessing produces four files:

| File | Description |
|---|---|
| `ligand.pt` | Atom features, bonds, fragment decomposition, dummy atoms |
| `protein_atoms.pt` | Protein atom graph and C&alpha; virtual node metadata |
| `graph.pt` | Unified static graph (all non-contact edges) |
| `meta.pt` | Counts, pocket center, schema version |

### `meta.pt`

| Key | Description |
|---|---|
| `pdb_id` | Complex identifier |
| `pocket_center` | Reference center for coordinate centering |
| `num_res` | Number of C&alpha; virtual nodes in pocket |
| `num_atom` | Number of ligand atoms (incl. optional dummies) |
| `num_frag` | Number of rigid fragments |
| `num_prot_atom` | Number of protein heavy atoms in pocket |
| `schema_version` | Preprocessing schema version |

### `ligand.pt`

**Atom features:** `atom_coords`, `atom_element`, `atom_charge`, `atom_aromatic`, `atom_hybridization`, `atom_in_ring`, `atom_degree`, `atom_implicit_valence`, `atom_explicit_valence`, `atom_num_rings`, `atom_chirality`, `atom_is_donor`, `atom_is_acceptor`, `atom_is_positive`, `atom_is_negative`, `atom_is_hydrophobe`, `atom_is_halogen`

**Bond and fragment data:** `bond_index`, `bond_type`, `fragment_id`, `frag_centers`, `frag_local_coords`, `frag_sizes`, `cut_bond_index`, `tri_edge_index`, `fragment_adj_index`

### `protein_atoms.pt`

**Atom features:** Same chemistry fields as ligand, plus `patom_is_backbone`, `patom_amino_acid`, `patom_is_ca`, `patom_ca_dist`, `patom_residue_id`

**Topology:** `pbond_index`, `pca_coords`, `pca_res_type`, `pca_atom_index`

## Graph Topology

The unified graph (`graph.pt`) concatenates all nodes and precomputes static edges. Dynamic protein-ligand contact edges are rebuilt at runtime from the current diffusion state.

### Node Types

| ID | Type | Description |
|---|---|---|
| 0 | `ligand_atom` | Ligand heavy atoms |
| 1 | `ligand_dummy` | Dummy atoms at cut bonds (optional) |
| 2 | `ligand_fragment` | Fragment centroid virtual nodes |
| 3 | `protein_atom` | Pocket heavy atoms (8 &Aring; cutoff) |
| 4 | `protein_ca` | C&alpha; virtual node per residue |

### Edge Types (Static)

| ID | Type | Description |
|---|---|---|
| 0 | `ligand_bond` | Covalent ligand bonds |
| 1 | `ligand_tri` | Cross-fragment triangulation edges |
| 2 | `ligand_cut` | Explicit cut-bond boundary edges |
| 3 | `ligand_atom_frag` | Atom &harr; owning fragment |
| 4 | `ligand_frag_frag` | Full graph over fragment nodes |
| 5 | `protein_bond` | Protein covalent bonds |
| 6 | `protein_atom_ca` | Protein atom &harr; residue C&alpha; |
| 7 | `protein_ca_ca` | C&alpha; &harr; C&alpha; (&le; 18 &Aring;) |
| 8 | `protein_ca_frag` | C&alpha; &harr; fragment (full bipartite) |

### Dynamic Edges (Runtime)

| ID | Type | Description |
|---|---|---|
| 9 | `dynamic_contact` | Protein atom &harr; ligand atom (distance-based, rebuilt each step) |

### Node Features

All node types share a unified feature schema in `graph.pt`:

| Feature | Type | Description |
|---|---|---|
| `node_coords` | float32 [N, 3] | Crystal coordinates |
| `node_type` | int64 [N] | Node category (0-4) |
| `node_element` | int64 [N] | Shared element vocabulary |
| `node_charge` | int8 [N] | Formal charge |
| `node_aromatic` | bool [N] | Aromatic flag |
| `node_hybridization` | int8 [N] | Hybridization type |
| `node_in_ring` | bool [N] | Ring membership |
| `node_amino_acid` | int64 [N] | Residue type (protein nodes only) |
| `node_fragment_id` | int64 [N] | Fragment assignment (ligand nodes only) |
| `node_is_donor/acceptor/...` | bool [N] | Pharmacophore flags |

### Edge Features

| Feature | Type | Description |
|---|---|---|
| `edge_index` | int64 [2, E] | Directed edge list |
| `edge_type` | int8 [E] | Edge category (0-9) |
| `edge_ref_dist` | float32 [E] | Reference distance in crystal frame |
| `edge_bond_type` | int8 [E] | Bond type for covalent edges (&minus;1 otherwise) |

### Design Principle

- **Static** edges (intra-ligand topology, intra-protein topology, protein-fragment hierarchy) are precomputed once.
- **Dynamic** edges (protein-ligand contacts) depend on the current noised ligand coordinates and are rebuilt every forward pass.

## Fragment Decomposition

Ligands are decomposed into rigid fragments by cutting non-ring rotatable single bonds:

1. Identify rotatable single bonds (excluding ring bonds and terminal hydrogens)
2. Keep aromatic/ring systems as single fragments
3. Merge very small terminal groups (&le;2 atoms) into adjacent fragments
4. Compute fragment centroids and local-frame coordinates

The decomposition is deterministic and computed once during preprocessing.
