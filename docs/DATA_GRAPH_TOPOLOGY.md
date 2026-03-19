# Data Graph Topology

This note summarizes the graph topology that `FlowFrag` actually builds today,
how it maps to the processed dataset, and which SigmaDock-style extensions are
most relevant for this codebase.

## Node Types

- `protein`
  - Residue-level pocket nodes.
  - Coordinates are pocket-centered `CA` positions.
  - Features are residue-type ids.
- `atom`
  - Ligand heavy-atom nodes.
  - Features include element, charge, aromaticity, hybridization, ring flag.
- `fragment`
  - One node per rigid ligand fragment.
  - Stores the current SE(3) state `(T_frag, q_frag)` plus fragment size.

Code:
- [`src/data/dataset.py`](/home/jaemin/project/protein-ligand/flowfrag/src/data/dataset.py)
- [`src/preprocess/protein.py`](/home/jaemin/project/protein-ligand/flowfrag/src/preprocess/protein.py)
- [`src/preprocess/ligand.py`](/home/jaemin/project/protein-ligand/flowfrag/src/preprocess/ligand.py)
- [`src/preprocess/fragments.py`](/home/jaemin/project/protein-ligand/flowfrag/src/preprocess/fragments.py)

## Static Topology

These edges are prepared in preprocessing / dataset loading and do not depend on
the current diffusion state.

- `atom --bond--> atom`
  - Directed ligand bond graph from the 2D chemistry graph.
  - Edge features: bond type, conjugation, ring membership.
- `atom --tri--> atom`
  - Cross-fragment triangulation edges around cut rotatable bonds.
  - Built from 1-hop neighborhoods on each side of a cut bond.
  - Each edge stores a crystal reference distance `ref_dist`.
- `atom --cut--> atom`
  - Explicit directed edges for the rotatable bonds used to split fragments.
  - These are the exact atom pairs from `cut_bond_index`, symmetrized at load time.
- `fragment --adj--> fragment`
  - Fragment adjacency induced by rotatable-bond cuts.

Important detail:
- `tri` and `cut` are complementary, not redundant.
- `cut` says which atom pair is the actual rotatable bond boundary.
- `tri` adds a wider soft local geometry stencil around that boundary.

Code:
- [`src/preprocess/fragments.py`](/home/jaemin/project/protein-ligand/flowfrag/src/preprocess/fragments.py)
- [`src/data/dataset.py`](/home/jaemin/project/protein-ligand/flowfrag/src/data/dataset.py)

## Dynamic Topology

These edges are rebuilt inside the docking head from the current `t`-state.

- `protein -> protein`
  - Residue radius graph in the protein encoder.
  - Default: `8A`, max `32` neighbors.
- `protein -> atom`
  - Residue-to-ligand-atom contact graph.
  - Default: `8A`, max `32` neighbors per atom.
- `protein -> fragment`
  - Coarse pocket-to-fragment graph.
  - Default: `10A`, max `64` neighbors per fragment.
- `fragment <-> fragment`
  - Coarse fragment interaction graph on fragment centroids.
  - Default: `6A`, max `8` neighbors per fragment.
  - In local mode, fragment topological adjacency is added on top.
- `atom -> fragment`
  - Not stored as an explicit edge type.
  - Implemented implicitly via ownership (`fragment_id`) and atom-to-fragment
    re-lifting inside the docking head.

Code:
- [`src/models/protein_encoder.py`](/home/jaemin/project/protein-ligand/flowfrag/src/models/protein_encoder.py)
- [`src/models/docking_head.py`](/home/jaemin/project/protein-ligand/flowfrag/src/models/docking_head.py)

## Distance Features

Distance encoding is currently simpler than SigmaDock-style Fourier/Bessel
blocks.

- All radius-based distances use Gaussian RBF features.
- `tri` edges additionally encode:
  - current distance,
  - `|d - d_ref|`,
  - signed `d - d_ref`.

Code:
- [`src/models/layers.py`](/home/jaemin/project/protein-ligand/flowfrag/src/models/layers.py)
- [`src/models/docking_head.py`](/home/jaemin/project/protein-ligand/flowfrag/src/models/docking_head.py)

## SigmaDock Comparison

Closest conceptual matches:

- SigmaDock `VF` node ~= our `fragment` node.
- SigmaDock protein virtual / `C_alpha` node ~= our `protein` residue node.
- SigmaDock triangulation / soft local geometry constraints ~= our `tri` edges.

Main differences:

- We do not use protein heavy atoms; protein context is residue-level only.
- We do not have explicit protein-heavy-atom / ligand-heavy-atom transient edges.
- We do not use a dedicated torsional-bond message-passing edge type.
- Our coarse graphs were originally local radius graphs, not all-to-all virtual
  graphs.

## Patch Roadmap

Most useful topology upgrades for this codebase, in priority order:

1. Stronger coarse global communication
   - Densify `protein -> fragment` and `fragment <-> fragment` graphs.
   - Rationale: closest match to SigmaDock's virtual-node message flow.
2. Explicit short-range protein-ligand contact edges
   - Preferably at heavier than `CA` resolution if the dataset is extended.
   - Rationale: current residue-only contacts are likely too coarse for fine
     docking.
3. Richer radial encoders
   - Optional replacement of the plain Gaussian RBF stack with smoother cutoff
     radial blocks.

## Implemented Patch: Optional Global Coarse Edges

This repo now supports switching the coarse graphs from local radius mode to
global full-connectivity mode.

- `pf_topology: radius | full`
  - Controls the `protein -> fragment` coarse graph.
- `ff_topology: radius | full`
  - Controls the `fragment <-> fragment` coarse graph.

Behavior:

- `radius`
  - Keeps the original sparse radius graph.
- `full`
  - Connects all nodes within each batch item.
  - For `ff_topology=full`, explicit topological adjacency is not re-added,
    because the dense graph already contains those pairs.

Current limits of the coarse patch:

- It does not introduce new node types.
- It does not add protein heavy-atom contacts.

Example config starting point:
- [`configs/overfit_neural1e_q1neqI_stochastic_priortime_effbs16_globalcoarse_wandb.yaml`](/home/jaemin/project/protein-ligand/flowfrag/configs/overfit_neural1e_q1neqI_stochastic_priortime_effbs16_globalcoarse_wandb.yaml)

## Implemented Patch: Explicit Cut-Bond Edges

This repo now also supports promoting `cut_bond_index` into a dedicated
atom-level message-passing edge type.

- dataset edge type: `atom --cut--> atom`
- model flag: `use_cut_bond_edges: true`

Behavior:

- Uses the exact rotatable-bond atom pairs that separate fragments.
- Adds a direct atom-level message path across the fragment boundary.
- Keeps `tri` edges in place, so the model sees both:
  - the exact hinge bond (`cut`)
  - the nearby geometry stencil (`tri`)

Intended effect:

- Improve local coordination across adjacent fragments.
- Give the model a cleaner signal for torsional coupling than triangulation
  alone.

## Observed Impact of the Topology Patches

These are the practical outcomes from the current stochastic overfit ablations.

### Global coarse graphs helped

Comparing the `effbs16` stochastic overfit with and without dense coarse graphs
at 50 rollout steps:

- local coarse, fresh prior:
  - atom RMSD `4.02A`
  - centroid distance `1.26A`
  - frag centroid RMSD `3.16A`
- global coarse, fresh prior:
  - atom RMSD `3.82A`
  - centroid distance `1.04A`
  - frag centroid RMSD `2.86A`

Interpretation:

- better global fragment and pocket communication is useful
- the global coarse patch is the best topology change tried so far

### Explicit cut-bond edges did not beat the global-coarse baseline

Under the current best inference setting (`late`, `power=3`, `25` steps):

- global coarse only:
  - atom RMSD `3.48A`
  - centroid distance `1.00A`
  - frag centroid RMSD `2.82A`
- global coarse + cut-bond:
  - atom RMSD `3.72A`
  - centroid distance `1.19A`
  - frag centroid RMSD `2.94A`

Interpretation:

- the `cut` edge is still a valid modeling idea
- but it is not the recommended default at the moment
- current recommendation:
  - keep `pf_topology=full`
  - keep `ff_topology=full`
  - leave `use_cut_bond_edges` off by default
