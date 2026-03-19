# Current Status

Last updated: 2026-03-19

This is the short "what should I use now?" note for the current FlowFrag
implementation.

## Recommended Baseline

Use this as the current mainline overfit / debug setup:

- model:
  - `omega_mode: neural_1e`
  - `hidden_dim: 256`
  - protein encoder layers: `3`
  - ligand encoder layers: `4`
  - docking layers: `4`
  - `pf_topology: full`
  - `ff_topology: full`
- data:
  - `prior_sigma: 1.0`
  - `rotation_augmentation: ligand_uniform`
  - `deterministic_augmentation: true`
  - `deterministic_prior: false`
  - `deterministic_time: false`
- optimization:
  - batch size `4`
  - gradient accumulation `4`
  - effective batch `16`
  - `AdamW`, `lr=1e-4`, `max_grad_norm=0.5`
- inference:
  - `time_schedule: late`
  - `schedule_power: 3`
  - `num_steps: 25`

Reference config:

- `configs/overfit_neural1e_q1neqI_stochastic_priortime_effbs16_globalcoarse_wandb.yaml`

## What Is Implemented

### Data and overfit controls

- deterministic augmentation / prior / time switches in `FlowFragDataset`
- `rotation_augmentation` modes:
  - `none`
  - `ligand_uniform`
  - `per_fragment`
- strict overfit subset selection in the trainer
- `q_target` stored on fragment data for correct evaluation and visualization

Main code:

- `src/data/dataset.py`
- `src/training/trainer.py`

### Topology options

- optional dense coarse graphs:
  - `pf_topology: radius | full`
  - `ff_topology: radius | full`
- optional explicit cut-bond atom edges:
  - dataset edge type: `atom --cut--> atom`
  - model flag: `use_cut_bond_edges: true`

Main code:

- `src/models/docking_head.py`
- `src/models/flowfrag.py`
- `src/data/dataset.py`

### Inference and trajectory schedule controls

- non-uniform rollout grid in `src/inference/sampler.py`
- CLI defaults now favor the current best setting:
  - `scripts/inference.py`
  - `scripts/visualize_trajectory.py`

Important detail:

- only the CLI defaults were changed to `late/power=3/25`
- trainer-side rollout behavior is not silently changed unless the caller passes
  the same arguments

## Best Current Results

### Best rollout setting on the global-coarse overfit checkpoint

- `uniform`, 50 steps:
  - RMSD `3.72A`
  - centroid `1.02A`
  - frag RMSD `2.84A`
- `late`, power 3, 25 steps:
  - RMSD `3.48A`
  - centroid `1.00A`
  - frag RMSD `2.82A`
  - `<2A` success `31.2%`

Artifacts:

- `outputs/schedule_ablation/globalcoarse_late_p3_s25`

### Current topology preference

- keep `pf_topology=full`, `ff_topology=full`
- do not enable `use_cut_bond_edges` by default

Reason:

- the cut-bond ablation did not beat the simpler global-coarse model under the
  current best inference setting

## What Is Still Open

- stochastic rotation learning is still weaker than translation learning
- pointwise velocity metrics still overestimate closed-loop rollout quality
- the full-training path should be revalidated with:
  - global coarse topology
  - `late/power=3/25` inference

## Related Docs

- [experiment_log.md](experiment_log.md)
- [DATA_GRAPH_TOPOLOGY.md](DATA_GRAPH_TOPOLOGY.md)
