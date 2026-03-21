# Data Graph Topology

This note describes the intended atom-level data topology after the SigmaDock-
style preprocessing extension.

Important status note:

- preprocessing now writes the new unified static graph to `graph.pt`
- the current runtime dataset in `src/data/dataset.py` still reads the legacy
  `protein.pt` path and does not yet consume `graph.pt`
- until the loader is migrated, `data/processed/` may contain a mixture of old
  and new outputs if it was rebuilt in place

For operator-facing rebuild instructions, see `data/README.md`.

## Design Split

The graph is intentionally split into:

- static topology, precomputed once in preprocessing
- dynamic topology, rebuilt from the current diffusion state at runtime

Only protein-ligand interaction edges are intended to remain dynamic.

## Stored Files

Per complex, the new schema aims to store:

- `ligand.pt`
- `protein_atoms.pt`
- `graph.pt`
- `meta.pt`

Deprecated legacy file:

- `protein.pt`
  - residue-level pocket representation from the old loader path

## Node Sets In `graph.pt`

`graph.pt` concatenates four node blocks into one unified static graph.

| Node block | `node_type` | Meaning |
|---|---:|---|
| ligand atom | `0` | real ligand heavy atom |
| ligand dummy | `1` | dummy atom copied across a cut bond |
| ligand fragment virtual | `2` | fragment centroid virtual node |
| protein atom | `3` | heavy atom from the pocket residues |
| protein CA virtual | `4` | per-residue `CA` virtual node |

The block boundaries are stored in:

- `lig_atom_slice`
- `lig_frag_slice`
- `prot_atom_slice`
- `prot_ca_slice`

## Unified Node Features In `graph.pt`

The following features are aligned across ligand and protein atoms so the model
can treat them in one graph.

| Key | Type | Meaning |
|---|---|---|
| `node_coords` | `float32 [N, 3]` | static coordinates in crystal gauge |
| `node_type` | `int64 [N]` | node category id |
| `node_element` | `int64 [N]` | shared element vocab |
| `node_charge` | `int8 [N]` | raw formal charge |
| `node_aromatic` | `bool [N]` | aromatic flag |
| `node_hybridization` | `int8 [N]` | shared hybridization vocab |
| `node_in_ring` | `bool [N]` | ring membership flag |
| `node_degree` | `int8 [N]` | heavy-atom degree |
| `node_implicit_valence` | `int8 [N]` | implicit valence |
| `node_explicit_valence` | `int8 [N]` | explicit valence |
| `node_num_rings` | `int8 [N]` | number of rings containing the atom |
| `node_chirality` | `int8 [N]` | RDKit chiral tag index |
| `node_is_donor` | `bool [N]` | SMARTS donor role |
| `node_is_acceptor` | `bool [N]` | SMARTS acceptor role |
| `node_is_positive` | `bool [N]` | SMARTS positive role |
| `node_is_negative` | `bool [N]` | SMARTS negative role |
| `node_is_hydrophobe` | `bool [N]` | SMARTS hydrophobe role |
| `node_is_halogen` | `bool [N]` | SMARTS halogen role |
| `node_amino_acid` | `int64 [N]` | residue type for protein-side nodes, `UNK` elsewhere |
| `node_is_backbone` | `bool [N]` | protein backbone atom flag |
| `node_is_ca` | `bool [N]` | `CA` atom flag or CA virtual flag |
| `node_ca_dist` | `int64 [N]` | graph distance to residue `CA` for protein atoms |
| `node_fragment_id` | `int64 [N]` | ligand fragment ownership, `-1` elsewhere |
| `node_residue_id` | `int64 [N]` | protein residue ownership, `-1` elsewhere |
| `node_is_dummy` | `bool [N]` | ligand dummy-atom flag |
| `node_is_virtual` | `bool [N]` | fragment/CA virtual-node flag |

Notes:

- fragment virtual and CA virtual nodes are padded with neutral/default values
  for chemistry fields they do not naturally own
- `node_charge` is now consistent across ligand and protein: both use raw formal
  charge rather than a sign-only class

## Static Edge Types In `graph.pt`

All non-protein-ligand edges are precomputed and stored in one edge list.

| Edge type | `edge_type` | Meaning |
|---|---:|---|
| ligand bond | `0` | covalent ligand bond |
| ligand triangulation | `1` | cross-fragment soft geometry edge |
| ligand cut | `2` | explicit cut-bond boundary edge |
| ligand atom-fragment | `3` | atom to owning fragment virtual |
| ligand fragment-fragment | `4` | full graph over fragment virtual nodes |
| protein bond | `5` | covalent protein bond |
| protein atom-CA | `6` | protein atom to residue CA virtual |
| protein CA-CA | `7` | CA virtual graph within `18A` |
| protein CA-fragment | `8` | full bipartite graph between CA virtuals and fragment virtuals |

Shared edge tensors:

| Key | Type | Meaning |
|---|---|---|
| `edge_index` | `int64 [2, E]` | unified directed edge list |
| `edge_type` | `int8 [E]` | edge category id |
| `edge_ref_dist` | `float32 [E]` | reference distance in the stored crystal gauge |
| `edge_bond_type` | `int8 [E]` | bond type for covalent edges, `-1` otherwise |
| `edge_bond_conjugated` | `bool [E]` | conjugation flag for covalent edges |
| `edge_bond_in_ring` | `bool [E]` | ring-bond flag for covalent edges |
| `edge_bond_stereo` | `int8 [E]` | bond stereo for covalent edges, `-1` otherwise |

## Dynamic Topology

These edges are intentionally not stored in `graph.pt` and should be rebuilt
from the current diffusion state:

- protein atom <-> ligand atom interaction edges
- any radius graph that depends on the current noised ligand coordinates
- optional time-dependent distance encodings for dynamic contacts

The intended rule is:

- static chemistry and hierarchy: preprocess once
- protein-ligand contacts: rebuild every step

## `ligand.pt`

`ligand.pt` keeps ligand-specific tensors that are still useful outside the
unified graph.

### Atom features

- `atom_coords`
- `atom_element`
- `atom_charge`
- `atom_aromatic`
- `atom_hybridization`
- `atom_in_ring`
- `atom_degree`
- `atom_implicit_valence`
- `atom_explicit_valence`
- `atom_num_rings`
- `atom_chirality`
- `atom_is_donor`
- `atom_is_acceptor`
- `atom_is_positive`
- `atom_is_negative`
- `atom_is_hydrophobe`
- `atom_is_halogen`

### Bond and fragment data

- `bond_index`, `bond_type`, `bond_conjugated`, `bond_in_ring`, `bond_stereo`
- `fragment_id`
- `frag_centers`
- `frag_local_coords`
- `frag_sizes`
- `cut_bond_index`
- `tri_edge_index`
- `tri_edge_ref_dist`
- `fragment_adj_index`
- optional dummy metadata:
  - `is_dummy`
  - `dummy_to_real`

## `protein_atoms.pt`

`protein_atoms.pt` stores the protein-side atom graph before unification.

### Atom features

- `patom_coords`
- `patom_element`
- `patom_charge`
- `patom_aromatic`
- `patom_hybridization`
- `patom_in_ring`
- `patom_degree`
- `patom_implicit_valence`
- `patom_explicit_valence`
- `patom_num_rings`
- `patom_chirality`
- `patom_is_donor`
- `patom_is_acceptor`
- `patom_is_positive`
- `patom_is_negative`
- `patom_is_hydrophobe`
- `patom_is_halogen`
- `patom_is_backbone`
- `patom_amino_acid`
- `patom_is_ca`
- `patom_ca_dist`
- `patom_residue_id`

### Protein topology metadata

- `pbond_index`, `pbond_type`, `pbond_conjugated`, `pbond_in_ring`, `pbond_stereo`
- `pca_coords`
- `pca_res_type`
- `pca_atom_index`

## Current Runtime Gap

The new schema is documented here, but the default runtime still uses the old
loader path:

- `src/data/dataset.py` reads `protein.pt`, `ligand.pt`, and `meta.pt`
- `protein_atoms.pt` is only attached as optional side information
- `graph.pt` is not consumed yet

So today:

- preprocessing supports the atom-level unified graph
- training code does not yet use it by default

## Recommended Next Step

Before new experiments that depend on the unified atom-level graph:

1. build a fresh dataset directory, e.g. `data/processed_v2/`
2. migrate `src/data/dataset.py` to read `graph.pt`
3. stop depending on legacy `protein.pt`
