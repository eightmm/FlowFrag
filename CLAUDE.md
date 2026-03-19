# CLAUDE.md

FlowFrag - Fragment-based Flow Matching for Protein-Ligand Docking

## Quick Reference

```bash
# Environment (uv manages the venv)
~/.local/bin/uv sync --dev

# Training
~/.local/bin/uv run python scripts/train.py --config configs/train.yaml

# Dataset build
~/.local/bin/uv run python scripts/build_fragment_flow_dataset.py --raw_dir /mnt/data/PLI/P-L --out_dir data/processed

# Tests
~/.local/bin/uv run python -m pytest tests/ -v
```

## Environment

**Required: PyTorch 2.9+** (for native Muon optimizer)

**Key Dependencies:**
- PyTorch >= 2.9.0 (Muon optimizer)
- cuEquivariance (SE(3) tensor products)
- torch_geometric, torch_cluster
- RDKit (molecule parsing, fragment decomposition)

## Architecture Overview

FlowFrag predicts rigid-body velocities for ligand fragments to dock them into protein binding pockets via flow matching.

### Pipeline

```
Protein (PDB) ──► Residue Encoder ──► s_prot [N_res, D]
                                              │
Ligand (SDF)  ──► Ligand Encoder  ──► s_lig  [N_lig, D]
                                              │
                                    ┌─────────┴──────────┐
                                    │  Context Fusion     │
                                    └─────────┬──────────┘
                                              │
Fragment State (T_f, R_f) + t ──────────────►│
                                              ▼
                                    ┌──────────────────┐
                                    │  SE(3)-Equivariant │
                                    │  Docking Head      │
                                    └────────┬─────────┘
                                             │
                                   ┌─────────┴──────────┐
                                   ▼                    ▼
                            v_f [N_frag, 3]     omega_f [N_frag, 3]
                            (translation vel.)  (angular vel.)
```

### Design Principles

1. **Fragment-level prediction**: Predict (v_f, omega_f) directly per fragment — no atom-force aggregation
2. **Protein is static context**: Pocket residues provide context, never move
3. **Minimal features in v1**: Atom type, charge, aromatic, hybridization, ring, bond type
4. **Separate rigid docking from torsion**: Solve fragment placement first, torsion refinement later
5. **SE(3)-equivariant head**: No data augmentation needed for translation/rotation

### Core Components

| Module | Location | Description |
|--------|----------|-------------|
| Protein Encoder | `src/models/protein_encoder.py` | Residue-level (CA coords + residue type) |
| Ligand Encoder | `src/models/ligand_encoder.py` | 2D GNN over molecular graph |
| Docking Head | `src/models/docking_head.py` | SE(3)-equivariant, time-conditioned |
| FlowFrag Model | `src/models/flowfrag.py` | Orchestrates all components |

## Flow Matching

- **State**: {T_f, R_f} per fragment
- **Prior (t=0)**: T ~ N(pocket_center, sigma^2 I), R ~ Uniform(SO(3))
- **Target (t=1)**: Crystal structure fragment poses
- **Interpolation**: Linear (T), SLERP (R)
- **Loss**: MSE on predicted vs ground-truth velocity (v_f, omega_f)

### ODE Integration (Inference)

```
T_{t+dt} = T_t + dt * v_f
R_{t+dt} = exp(dt * omega_f) * R_t
```

## Geometry Conventions

- **Quaternions**: (w, x, y, z) — scalar first
- **Coordinates**: Angstroms
- **Fragment local coords**: Relative to fragment centroid
- **omega_f frame**: Must be consistent — pick global OR body frame, never mix
  - Global angular velocity + left-multiplication, OR
  - Body-frame angular velocity + right-multiplication

## Fragment Decomposition

- Cut non-ring rotatable single bonds
- Keep aromatic/ring systems as single fragments
- Merge very small terminal groups into adjacent fragments
- Deterministic, computed once in preprocessing

## Data

### Raw Source
`/mnt/data/PLI/P-L/<year-range>/<pdb_id>/`
- `<pdb_id>_pocket.pdb`
- `<pdb_id>_ligand.sdf`

### Processed Format (per complex)

**protein.pt:**
- `res_coords: [N_res, 3]` — CA coordinates
- `res_types: [N_res]` — amino acid IDs

**ligand.pt:**
- `atom_coords: [N_atom, 3]` — crystal coordinates
- `atom_feats: [N_atom, F]` — atom features
- `bond_index: [2, E]` — bond connectivity
- `bond_feats: [E, F_bond]` — bond features
- `fragment_id: [N_atom]` — fragment assignment
- `frag_centers: [N_frag, 3]` — fragment centroids
- `frag_local_coords: [N_atom, 3]` — local frame coordinates

## Directory Structure

```
flowfrag/
├── src/
│   ├── models/          # Model architectures
│   ├── data/            # Dataset classes
│   ├── geometry/        # SE(3) ops, quaternions, flow matching
│   ├── preprocess/      # Raw data processing pipeline
│   └── training/        # Training loop, schedulers, losses
├── scripts/             # train.py, evaluate.py, build_dataset.py
├── configs/             # YAML configs
├── tests/               # Tests (mirror src/ structure)
├── data/                # Processed data (gitignored)
└── outputs/             # Checkpoints, logs (gitignored)
```

## Training Config

```yaml
model:
  hidden_dim: 256
  num_encoder_layers: 4
  num_docking_layers: 4

flow_matching:
  prior_sigma: 1.0

training:
  batch_size: 8
  gradient_accumulation_steps: 4
  use_muon: true
  lr: 3.0e-4
  muon_lr: 0.02
  scheduler_type: trapezoidal
```

## Current Status

- **Translation (v)**: Working. cos_v=0.817, median RMSD 4.16Å on val set.
- **Rotation (omega)**: Analytic formula works for re-docking (q_1=I). Neural omega failed with 9 approaches — root cause: q_1=I makes omega target protein-independent.
- **Active experiment**: `local_frame` omega mode with random rotation augmentation (q_1≠I) to force protein-dependent rotation learning.
- **Detailed experiment log**: See `docs/experiment_log.md`

### Omega Modes (model.omega_mode)

| Mode | Status | Description |
|------|--------|-------------|
| `analytic` | Working (re-docking only) | `scale(t) * axis_angle(q_t)`, protein-independent |
| `local_frame` | Testing | Body-frame invariant MLP + R(q_t) rotation, with q_1≠I augmentation |
| `neural_1e` | Failed | Direct 1e output from equivariant head |
| `newton_euler` | Failed | Atom forces → torque → inertia solve |
| `atom_velocity` | Failed | Per-atom velocity → derive omega |

### Key Architecture Features (V1.2)
- Unified irreps: protein/atom/fragment all use `0e + 1o + 1e`
- Protein 1o injection (displacement-gated vectors)
- Triangulation edges (cross-fragment distance constraints)
- Rotation matrix columns as 1o fragment features

## Dependencies

- **PyTorch 2.9+**: Native Muon optimizer
- **cuEquivariance**: SE(3) tensor products (CUDA)
- **torch_geometric, torch_cluster**: Graph operations
- **RDKit**: Molecule parsing, fragment detection
- **e3nn**: Angle utilities (minimal usage)
- **wandb**: Experiment tracking
