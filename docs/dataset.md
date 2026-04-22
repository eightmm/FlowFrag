# Dataset Format

## Raw Data

PDBbind-style per-complex directory:

```
raw_data/<pdb_id>/
├── <pdb_id>_protein.pdb   # Full protein (preferred)
├── <pdb_id>_pocket.pdb    # Fallback if full protein missing
├── <pdb_id>_ligand.sdf    # Ligand structure
└── <pdb_id>_ligand.mol2   # Optional MOL2 fallback
```

## Building the Dataset

```bash
uv run python scripts/build_fragment_flow_dataset.py \
    --raw_dir /mnt/data/PLI/P-L \
    --out_dir data/processed \
    --workers 8
```

## Processed Output

Preprocessing saves **full protein** + **ligand** tensors separately. The unified graph is **constructed at runtime** in `Dataset.__getitem__` after cropping the protein to a pocket region. This lets training jitter the pocket boundary and inference use a predicted pocket center without recomputing anything offline.

| File | Description |
|------|-------------|
| `protein.pt` | Full protein heavy atoms + residue virtual nodes (no cutoff) |
| `ligand.pt` | Atom features + fragment decomposition |
| `meta.pt` | Pocket center + counts + flags |

### `meta.pt`

| Key | Description |
|-----|-------------|
| `pdb_id` | Complex identifier |
| `pocket_center` | Centroid of pocket residue virtual nodes within 8 Å of crystal ligand (protein-derived) |
| `num_pocket_res` | Residues in 8 Å pocket (reference only) |
| `num_res` | Residues in the full protein |
| `num_atom` | Ligand atoms |
| `num_frag` | Rigid fragments |
| `num_prot_atom` | Full-protein heavy atoms |
| `used_full_protein` | True if `_protein.pdb` used (else fallback to `_pocket.pdb`) |
| `used_mol2_fallback` | True if SDF failed and MOL2 was used |
| `ligand_sanitize_ok` | RDKit sanitize status |
| `schema_version` | Preprocessing schema version (currently 1) |

### `protein.pt`

Full protein heavy atoms + canonical bond topology, no spatial cutoff.

| Key | Shape | Description |
|-----|-------|-------------|
| `patom_coords` | [N_patom, 3] float32 | Heavy atom coordinates |
| `patom_token` | [N_patom] int64 | `(residue, atom_name)` identity token (covers all 20 AAs + metals) |
| `patom_residue_id` | [N_patom] int64 | Local residue index |
| `patom_is_backbone` | [N_patom] bool | Backbone (N/CA/C/O) flag |
| `patom_is_metal` | [N_patom] bool | Metal ion flag |
| `pbond_index` | [2, E_prot] int64 | Canonical intra-AA + peptide + disulfide bonds (bidirectional) |
| `pres_coords` | [N_res, 3] float32 | Residue virtual node position (CB / pseudo-CB / metal atom) |
| `pres_residue_type` | [N_res] int64 | AA type (0-19 standard, 20=UNK, 21=METAL) |
| `pres_atom_index` | [N_res] int64 | Anchor atom in patom space |
| `pres_is_pseudo` | [N_res] bool | True for GLY pseudo-CB or CA fallback |

### `ligand.pt`

| Key | Shape | Description |
|-----|-------|-------------|
| `atom_coords` | [N_atom, 3] float32 | Crystal coordinates |
| `atom_element` | [N_atom] int64 | Element index (C=0..Se=11, OTHER=12) |
| `atom_charge` | [N_atom] int8 | Formal charge |
| `atom_aromatic` | [N_atom] bool | Aromatic flag |
| `atom_hybridization` | [N_atom] int8 | SP..SP3D2=0-4, UNSPECIFIED=5, OTHER=6 |
| `atom_degree, atom_implicit_valence, atom_explicit_valence` | [N_atom] int8 | |
| `atom_num_rings, atom_chirality` | [N_atom] int8 | |
| `atom_is_donor / acceptor / positive / negative / hydrophobe / halogen` | [N_atom] bool | Pharmacophore flags (RDKit BaseFeatures.fdef) |
| `bond_index` | [2, E_dir] int64 | Directed bond list (E_dir = 2 × N_bonds) |
| `bond_type, bond_conjugated, bond_in_ring, bond_stereo` | [E_dir] | Bond chemistry |
| `fragment_id` | [N_atom] int64 | Atom → fragment assignment |
| `frag_centers` | [N_frag, 3] float32 | Fragment centroids |
| `frag_local_coords` | [N_atom, 3] float32 | Atom coords in fragment local frame (centroid-subtracted) |
| `frag_sizes` | [N_frag] int64 | Atoms per fragment |
| `tri_edge_index, tri_edge_ref_dist` | | Cross-fragment triangulation edges |
| `fragment_adj_index, cut_bond_index` | | Fragment adjacency + cut bonds |
| `dg_bounds` | [N_atom, N_atom] float32 | Distance geometry bounds (optional) |

## Runtime Graph Construction

In `Dataset.__getitem__`:

1. Load `protein.pt`, `ligand.pt`, `meta.pt`
2. **Crop** protein to pocket: `crop_to_pocket(protein, pocket_center ± jitter, cutoff ± noise)`
3. **Build** unified graph: `build_static_complex_graph(ligand, cropped_protein)`
4. Sample flow matching state (prior + t) and update dynamic coordinates

### Node Types (in runtime graph)

| ID | Type | Description |
|----|------|-------------|
| 0 | `ligand_atom` | Ligand heavy atoms |
| 1 | `ligand_fragment` | Fragment centroid virtual nodes |
| 2 | `protein_atom` | Pocket heavy atoms |
| 3 | `protein_res` | Residue virtual nodes (CB/pseudo-CB/metal) |

### Edge Types (Static)

| ID | Type | Description |
|----|------|-------------|
| 0 | `ligand_bond` | Covalent ligand bonds (keeps bond-type features) |
| 1 | `ligand_tri` | Cross-fragment triangulation edges (with ref_dist) |
| 2 | `ligand_cut` | Explicit cut-bond boundary edges |
| 3 | `ligand_atom_frag` | Atom ↔ owning fragment |
| 4 | `ligand_frag_frag` | All-pairs between fragment nodes |
| 5 | `protein_bond` | Canonical intra-AA + peptide + disulfide bonds |
| 6 | `protein_atom_res` | Protein atom ↔ residue virtual node |
| 7 | `protein_res_res` | Residue ↔ residue (10 Å cutoff) |
| 8 | `protein_res_frag` | Residue ↔ fragment (all-pairs bipartite) |

### Dynamic Edges (per forward)

| ID | Type | Description |
|----|------|-------------|
| 9 | `dynamic_contact` | Protein atom ↔ ligand atom within `contact_cutoff` Å, rebuilt every forward |

### Node Features in Graph

All node types share one flattened feature tensor. Non-applicable slots are padded with sentinels.

| Feature | Description |
|---------|-------------|
| `node_coords` | [N, 3] — positions (dynamic for ligand atoms/fragments during flow matching) |
| `node_type` | 0-3 (see above) |
| Ligand chemistry | `node_element, node_charge, node_aromatic, node_hybridization, node_num_rings, node_chirality, atom_is_*` — sentinels for non-ligand nodes |
| Protein identity | `node_patom_token` (covers all (AA, atom_name) pairs), `node_patom_is_metal` |
| Residue-level | `node_pres_residue_type`, `node_pres_is_pseudo` |
| Structural | `node_fragment_id`, `node_residue_id` |

### Edge Features

| Feature | Description |
|---------|-------------|
| `edge_index` | [2, E] directed edges |
| `edge_type` | 0-9 |
| `edge_ref_dist` | Reference distance (for tri edges and inter-node distances) |
| `edge_bond_type, edge_bond_conjugated, edge_bond_in_ring, edge_bond_stereo` | Bond chemistry (-1 sentinel for non-bond edges) |
| `edge_frag_hop` | Topological hop distance for `ligand_frag_frag` edges |

## Pocket Augmentation

Training jitters the pocket definition to simulate imperfect binding site prediction at inference:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `pocket_cutoff` | 8.0 Å | Base residue-aware distance cutoff |
| `pocket_jitter_sigma` | 2.0 Å | Gaussian jitter on pocket center |
| `pocket_cutoff_noise` | 2.0 Å | Uniform noise on cutoff (6-10 Å range) |

Validation sets both jitter params to 0 for deterministic evaluation.

## Fragment Decomposition

Ligands are decomposed into rigid fragments by cutting non-ring rotatable single bonds:

1. Identify cuttable bonds: single, not in ring, not amide/ester/sulfonamide, both endpoints with degree > 1
2. Greedy selection with "no singleton fragments" constraint
3. Keep aromatic/ring systems as single fragments
4. Merge size-1 fragments into adjacent fragments
5. Compute fragment centroids and local-frame coordinates

Deterministic, computed once in preprocessing.

## Dataset Filtering

Applied in `UnifiedDataset.__init__`:

| Parameter | Default | Reason |
|-----------|---------|--------|
| `min_atoms` | 5 | Exclude trivial ligands |
| `max_atoms` | 80 | p95 of ligand size; drop outliers |
| `max_frags` | 20 | Frag-frag edges are O(N²); 93.1% retained |
| `min_protein_res` | 50 | Exclude peptides/artifacts (~22 complexes) |

**PDBbind 2020 build stats** (schema_version=1):
- Raw: 19,037 complexes. Success: 18,157 (95.4%). Covalent skip: 877.
- After default filters: ~16,700 complexes (92%)
- Ligand atoms: median 28, p95 = 83
- Fragments: median 5, p95 = 24
- Full-protein residues: median 382, max 5,292
- Pocket residues (8 Å): median 29, p95 = 44

## Splits

`data/splits/pdbbind2020.json`:
- **Train**: 16,463 complexes
- **Val**: 284 complexes (CASF-2016 core set)
- **Filter criteria** stored in the JSON for reproducibility

CASF-2016 core is PDBbind's official benchmark set — small, diverse (57 targets), and established for docking/scoring evaluation.
