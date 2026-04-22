# FlowFrag

**Fragment-based Flow Matching for Protein-Ligand Docking**

FlowFrag predicts protein-ligand docking poses by decomposing ligands into rigid
fragments and learning SE(3) velocity fields via flow matching. A single
SE(3)-equivariant GNN produces per-atom forces that are aggregated via
Newton-Euler rigid-body mechanics into per-fragment translation **v** and
angular **ω** velocities. An ODE integrator then transports a random
prior pose to the docked pose.

## Highlights

- **Fragment-level rigid-body docking** — ligands are split at rotatable bonds
  into rigid fragments; per-fragment (v, ω) is predicted, so rotational and
  translational degrees of freedom are handled natively instead of through
  atom-level displacement.
- **SE(3)-equivariant GNN** — tensor product message passing over a
  heterogeneous protein-ligand graph with irreps up to l=2, accelerated by
  [cuEquivariance](https://docs.nvidia.com/cuda/cuequivariance/) CUDA kernels
  (fused_tp).
- **Flow matching on SE(3)** — linear interpolation for translation, SLERP for
  rotation, logit-normal time sampling. Single-step training objective
  (v, ω regression); ODE-based multi-step inference.
- **CASF-2016 core**: **1.07 Å** median RMSD and **89.4 %** < 2 Å success
  (oracle top-1 of 10 priors, N=284), from a 25 K-step training run on
  8× NVIDIA B200.

## Results

Pocket-conditioned re-docking on CASF-2016 core (N = 284). `best.pt` from the
v3 training run (`configs/train_v3_b200.yaml`, 25 K steps, effective batch 512).

| Selection | Median RMSD | Mean RMSD | < 2 Å | < 5 Å |
|---|---|---|---|---|
| Single prior, 25 ODE steps | 2.15 Å | — | 44.4 % | 90.5 % |
| Oracle top-1 of 10 priors | **1.07 Å** | 1.24 Å | **89.4 %** | **98.9 %** |
| Worst-of-10 (upper bound) | 3.45 Å | — | — | — |

> **Oracle** = best of N by ground-truth RMSD (reachable upper bound on the
> model's sampling distribution; a downstream scoring function would pick
> among the N candidates at deployment).

Integration-step sweep on the same checkpoint (single prior per sample,
showing that the learned velocity field is ODE-smooth — 5 steps are enough):

| ODE steps | Median RMSD | < 2 Å | < 5 Å |
|---|---|---|---|
| 5  | 2.22 Å | 43.0 % | 91.6 % |
| 10 | 2.17 Å | 43.7 % | 90.9 % |
| 20 | 2.16 Å | 44.4 % | 90.9 % |
| 25 | 2.15 Å | 44.4 % | 90.5 % |

## Method

```mermaid
flowchart LR
    subgraph Input
        P["Protein\n(PDB)"]
        L["Ligand\n(SDF / SMILES)"]
    end

    subgraph Preprocessing
        PE["Pocket crop\n8 Å cutoff"]
        FD["Fragment\ndecomposition"]
    end

    subgraph Model["SE(3)-Equivariant GNN"]
        HG["Heterogeneous\ngraph"]
        TP["TP Conv × 6\n+ AdaLN(t)"]
        HEAD["Newton-Euler\naggregation"]
    end

    subgraph Output
        VW["v, ω\nper fragment"]
        ODE["ODE\n25 steps"]
        POSE["Docked\npose"]
    end

    P --> PE --> HG
    L --> FD --> HG
    HG --> TP --> HEAD --> VW --> ODE --> POSE
```

### Flow matching on SE(3)

Each ligand fragment carries a rigid-body state (T, R) — translation and
rotation.

| | Translation | Rotation |
|---|---|---|
| **Prior** (t=0) | N(pocket center, σ²I), σ = 3.0 Å | Uniform on SO(3) |
| **Target** (t=1) | Crystal pose | Crystal pose |
| **Interpolation** | Linear | SLERP |
| **Velocity target** | v = T₁ − T₀ | ω = axis-angle(q₁ · q₀⁻¹), world frame |

The model learns the conditional velocity field. At inference, the ODE
integrates from prior to pose:

```
T(t+dt) = T(t) + dt · v
R(t+dt) = exp(dt · [ω]×) · R(t)           (left-multiply, world frame)
```

### Architecture

Heterogeneous graph with 4 node types (`ligand_atom`, `ligand_fragment`,
`protein_atom`, `protein_res`) and 10 edge types covering intra-ligand,
intra-protein, and cross-modal connectivity. Per-edge gated tensor product
convolution with dual radial scaling and per-edge-type σ decay. Time
conditioning via Equivariant AdaLN. Per-atom forces at l=1 are aggregated
into fragment (v, ω) via eigendecomposed inertia tensor with pseudo-inverse
(projecting into the observable rotation subspace so rank-deficient 2-atom
fragments contribute only their observable axes).

See [docs/architecture.md](docs/architecture.md) for layer-level details.

## Installation

### Requirements

- Python ≥ 3.12
- NVIDIA GPU with CUDA 13 (tested on B200 / sm_100)
- PyTorch ≥ 2.10 (ships native Muon optimizer)

### Setup

```bash
git clone https://github.com/eightmm/FlowFrag.git
cd FlowFrag
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --dev
```

## Data

### Raw layout

Built from PDBbind 2020 refined set:

```
raw_data/<year-range>/<pdb_id>/
├── <pdb_id>_protein.pdb    # full protein (preferred)
├── <pdb_id>_pocket.pdb     # used as fallback if protein missing
└── <pdb_id>_ligand.sdf     # crystal ligand (mol2 fallback accepted)
```

### Preprocess

```bash
uv run python scripts/build_fragment_flow_dataset.py \
    --raw_dir /path/to/pdbbind2020 \
    --out_dir data/processed \
    --workers 8
```

Produces per complex `data/processed/<pdb_id>/{protein.pt, ligand.pt, meta.pt}`:
the full protein is stored once, and the pocket is cropped at runtime in the
dataset's `__getitem__`. See [docs/dataset.md](docs/dataset.md) for tensor
schemas.

### Split (`data/splits/pdbbind2020.json`)

The split is deterministic and shipped in-repo:

| Split | # Complexes | Source |
|---|---|---|
| **train** | 16,463 | PDBbind 2020 refined set, minus CASF-2016 core, minus size-filtered outliers |
| **val** | 284 | CASF-2016 core set (standard re-docking benchmark) |

Size filters applied when building the split:
`min_atoms = 5`, `max_atoms = 80`, `max_frags = 20`, `min_protein_res = 50`.
Covalent binders and complexes with parse failures are also excluded; see
`data/covalent_binders_v2020.txt` for the excluded list.

## Training

```bash
uv run torchrun --standalone --nproc_per_node=8 scripts/train.py \
    --config configs/train_v3_b200.yaml
```

### Training setup (v3, `configs/train_v3_b200.yaml`)

| Category | Value |
|---|---|
| Hardware | 8 × NVIDIA B200 (183 GB HBM3e each) |
| Effective batch size | 512 (64 / GPU × 8 GPU DDP) |
| Optimizer | Muon (lr 0.02, ≥ 2-D params) + AdamW (lr 3e-4, 1-D params) |
| Schedule | Trapezoidal — warmup 4 % (1 K step), stable 66 % (16.5 K), cooldown 30 % (7.5 K) |
| Total steps | 25,000 (≈ 780 epochs, ≈ 12.8 M sample-updates) |
| EMA | decay 0.999, used for val/rollout |
| Wall clock | ≈ 27 hours on 8× B200 |

Loss composition (per the validated formulation; see
[docs/architecture.md](docs/architecture.md)):

| Term | Weight | Role |
|---|---|---|
| `loss_v` | 1 | Translation velocity MSE |
| `loss_ω` | 8 | Angular velocity MSE, projected onto observable subspace |
| `loss_atom_aux` | 0.3 | Per-atom rigid-body velocity consistency |
| `loss_dg` | 3.0 | Inter-fragment pairwise distance MSE after one-step Euler |
| `loss_boundary` | 0.3 | Cut-bond atom velocity consistency (see note) |

> Note: `loss_boundary` is a self-consistency regulariser that plateaus at a
> non-zero floor because the ground-truth velocity field derived from
> per-fragment SLERP does not generally preserve cut-bond continuity during
> interpolation. Keeping the weight small (0.3) limits interference with
> the primary flow-matching objective.

Prior / augmentation choices:

| Parameter | Value | Rationale |
|---|---|---|
| `prior_sigma` | 3.0 Å | Empirically matches the per-axis std of T₁ (= 3.39 Å, measured across 90,489 fragments) — see `scripts/analyze_target_distribution.py` |
| `rotation_augmentation` | `ligand_uniform` | Single Uniform(SO(3)) rotation applied per ligand per training sample |
| `pocket_jitter_sigma` | 2.0 Å | Simulates inference-time pocket-centre uncertainty |
| `pocket_cutoff_noise` | 2.0 Å | Uniform noise on pocket cutoff around the base 8 Å |
| Time sampling | logit-normal (t = σ(𝒩(0,1))) | Concentrates training on t ≈ 0.5 where flow matching is hardest |

### Evaluate a checkpoint

Full val (284 complexes), sharded across ranks:

```bash
# single-prior rollout (fast, matches training-time rollout at S25000)
uv run torchrun --standalone --nproc_per_node=8 scripts/rollout.py \
    --config configs/train_v3_b200.yaml \
    --checkpoint outputs/v3_b200/checkpoints/best.pt

# multi-prior oracle top-1 of N
uv run torchrun --standalone --nproc_per_node=8 scripts/rollout_topN.py \
    --config configs/train_v3_b200.yaml \
    --checkpoint outputs/v3_b200/checkpoints/best.pt \
    --num_priors 10 --num_steps 25 \
    --output outputs/v3_b200/rollout_top10.json
```

## Docking (inference)

```bash
# From SDF / mol2
uv run python scripts/dock.py \
    --protein pocket.pdb \
    --ligand ligand.sdf \
    --checkpoint outputs/v3_b200/checkpoints/best.pt \
    --config configs/train_v3_b200.yaml \
    --num_samples 10

# From SMILES (pocket_center required — user-supplied or predicted)
uv run python scripts/dock.py \
    --protein pocket.pdb \
    --ligand "CCO" \
    --pocket_center 10.0,10.0,10.0 \
    --checkpoint outputs/v3_b200/checkpoints/best.pt \
    --config configs/train_v3_b200.yaml
```

## Project structure

```
flowfrag/
├── src/
│   ├── models/          # SE(3)-equivariant GNN (equivariant.py, unified.py)
│   ├── data/            # Dataset (dataset.py), runtime pocket crop + graph build
│   ├── geometry/        # SE(3) ops, quaternions, flow matching primitives
│   ├── preprocess/      # Raw-data processing pipeline (fragments, graph, protein, ligand)
│   ├── training/        # Training loop, losses, DDP, Muon+AdamW, trapezoidal schedule
│   ├── inference/       # ODE sampler, docking metrics
│   └── scoring/         # Vina energy, PoseBusters validity, pose ranking
├── scripts/
│   ├── train.py                         # main training entrypoint (DDP via torchrun)
│   ├── build_fragment_flow_dataset.py   # raw → processed tensors
│   ├── rollout.py                       # standalone val rollout
│   ├── rollout_topN.py                  # multi-prior oracle top-1 eval
│   ├── dock.py                          # dock a single complex
│   ├── eval_benchmark.py                # PoseBusters / Astex / CASF evaluation
│   ├── analyze_target_distribution.py   # T_1 distribution → prior_sigma tuning
│   └── probe_memory.py                  # per-GPU batch-size ceiling probe
├── configs/             # YAML configuration files (train_v3_b200.yaml, train_v2.yaml, ...)
├── data/splits/         # deterministic train/val splits (PDBbind 2020 → CASF-2016 core)
├── tests/               # unit tests (geometry, losses, equivariance, preprocess)
└── docs/                # architecture & dataset reference
```

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | Model architecture, graph structure, design choices |
| [Dataset](docs/dataset.md) | Data format, preprocessing pipeline, graph topology |

## Citation

```bibtex
@software{flowfrag2026,
    title  = {FlowFrag: Fragment-based Flow Matching for Protein-Ligand Docking},
    author = {Jaemin Sim},
    year   = {2026},
    url    = {https://github.com/eightmm/FlowFrag}
}
```

## License

Released under the MIT License.

## Acknowledgments

- [cuEquivariance](https://docs.nvidia.com/cuda/cuequivariance/) — SE(3)-equivariant tensor product CUDA kernels
- [PoseBusters](https://doi.org/10.1039/D3SC04185A) — validity checks used in downstream scoring
- [CASF-2016](https://doi.org/10.1021/acs.jcim.8b00545) — benchmark core set used for validation
