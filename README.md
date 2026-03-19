# FlowFrag

Fragment-based Flow Matching for Protein-Ligand Docking

## Overview

FlowFrag decomposes ligands into rigid fragments and predicts SE(3) velocities (translation + rotation) for each fragment using flow matching. An ODE integrator assembles the final docking pose.

**Key ideas:**
- Rigid fragment decomposition via rotatable bond cuts
- Direct fragment velocity prediction (v_f, omega_f) — no atom-force aggregation
- SE(3)-equivariant network for the docking head
- Flow matching: linear interpolation (translation) + SLERP (rotation)

## Setup

```bash
# Install dependencies
~/.local/bin/uv sync

# Activate environment
source .venv/bin/activate
```

## Project Structure

```
flowfrag/
├── src/
│   ├── models/          # Model architectures
│   ├── data/            # Dataset classes
│   ├── geometry/        # SE(3) operations, flow matching
│   ├── preprocess/      # Raw data → processed tensors
│   └── training/        # Training loop, schedulers
├── scripts/             # Executable scripts
├── configs/             # YAML configs
├── tests/               # Tests
├── data/                # Processed data (gitignored)
└── outputs/             # Checkpoints, logs (gitignored)
```

## Data

Raw complexes: `/mnt/data/PLI/P-L/<year-range>/<pdb_id>/`

Each complex contains:
- `<pdb_id>_pocket.pdb` — binding pocket residues
- `<pdb_id>_ligand.sdf` — ligand structure

## References

- SigmaDock: Fragment-based SE(3) docking
- FlowDock: Geometric flow matching for docking
- DiffDock: Torsional diffusion for molecular docking

## Additional Docs

- [Current Status](docs/CURRENT_STATUS.md)
- [Experiment Log](docs/experiment_log.md)
- [Data Graph Topology](docs/DATA_GRAPH_TOPOLOGY.md)

## Current Recommendation

As of 2026-03-19, the recommended overfit baseline is:

- `neural_1e` rotation head
- dense coarse graphs: `pf_topology=full`, `ff_topology=full`
- ligand-level uniform target rotation augmentation
- stochastic prior/time overfit with effective batch `16`
- inference with `late` time schedule, `power=3`, `25` steps
