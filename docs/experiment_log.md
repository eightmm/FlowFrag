# FlowFrag Experiment Log

Last updated: 2026-03-19

This document records the recent debugging and ablation work around
fragment-rotation learning, overfit protocol design, topology changes, and
inference-time discretization.

For a short implementation snapshot, see [CURRENT_STATUS.md](CURRENT_STATUS.md).

## Executive Summary

Current takeaways:

- The original `neural_1e` rotation failure was not caused by the irreps or the
  basic docking-head layout.
- The main early blocker was a data / protocol issue:
  - `q_target` collapsed to the identity gauge in preprocessing.
  - the original "overfit" path still resampled `t`, `q_0`, and sometimes
    `q_target`, so the target moved while training.
- Once the overfit task was made truly deterministic, `neural_1e` could fit it.
- As the task was relaxed from strict memorization to stochastic prior/time
  sampling, translation remained learnable but rotation stayed harder.
- Among the graph changes tried so far, dense coarse graphs
  (`pf_topology=full`, `ff_topology=full`) helped the most.
- Among inference-only changes tried so far, a non-uniform `late` time grid
  with `power=3` and `25` steps is the best current setting on the overfit
  checkpoint.

## Chronology

## 1. Gauge Fix and Strict Overfit Infrastructure

Problem:

- The processed fragment frame effectively made `q_target = I` in the crystal
  gauge.
- The old overfit path also shuffled data and resampled time / prior variables,
  so the training target was not actually fixed.

Implemented:

- deterministic dataset controls for augmentation, prior, and time
- explicit `rotation_augmentation` modes:
  - `none`
  - `ligand_uniform`
  - `per_fragment`
- fixed overfit subset selection with shuffle disabled
- `fragment.q_target` stored on each sample so evaluation and visualization can
  reconstruct the true target pose correctly

Relevant code:

- `src/data/dataset.py`
- `src/training/trainer.py`
- `scripts/inference.py`
- `scripts/visualize_trajectory.py`

Outcome:

- `neural_1e` passed strict overfit once the task was truly fixed.
- This established that the head is capable of learning a stable rotation
  target.

## 2. Overfit Ladder

The project then moved through increasingly realistic overfit tasks.

### 2.1 Strict deterministic overfit

Setup:

- 16 complexes fixed
- `q_target` fixed per complex
- `q_0` fixed
- `t` fixed

Outcome:

- `neural_1e` fit almost perfectly.
- This only proves pointwise memorization of a single `(prior, time)` target per
  complex, but it removed "model cannot learn rotation at all" as the main
  hypothesis.

Representative config:

- `configs/overfit_neural1e_q1neqI_strict.yaml`
- `configs/overfit_neural1e_q1neqI_strict_wandb.yaml`

### 2.2 Time-only overfit

Setup:

- complexes fixed
- `q_target` fixed
- one prior per complex fixed
- `t` varied

Outcome:

- `neural_1e` learned the time-dependent field for the seen prior.
- trajectory quality improved on the seen prior but not on new priors.

Representative config:

- `configs/overfit_neural1e_q1neqI_tonly_wandb.yaml`

### 2.3 Prior-bank overfit

Setup:

- complexes fixed
- `q_target` fixed
- multiple cached priors and times per complex

Outcome:

- small-model prior-bank overfit was weak
- a larger, more stable optimizer setting improved learning, but trajectory
  quality still lagged behind pointwise metrics

Representative configs:

- `configs/overfit_neural1e_q1neqI_priorbank_wandb.yaml`
- `configs/overfit_neural1e_q1neqI_priorbank_large_stable_wandb.yaml`

### 2.4 Fully stochastic prior/time overfit

Setup:

- 16 complexes fixed
- `q_target` fixed by ligand-level augmentation
- prior and time resampled continuously

Outcome:

- this is the closest overfit proxy to the real training objective so far
- translation remained learnable
- rotation remained significantly harder

Representative configs:

- `configs/overfit_neural1e_q1neqI_stochastic_priortime_wandb.yaml`
- `configs/overfit_neural1e_q1neqI_stochastic_priortime_effbs16_wandb.yaml`

Key observation:

- increasing effective batch from `4` to `16` improved pointwise learning
  substantially, especially `cos_omega`
- rollout quality improved only modestly until topology and inference-schedule
  changes were added

## 3. Topology Ablations

## 3.1 Global coarse graphs

Patch:

- added `pf_topology: radius | full`
- added `ff_topology: radius | full`
- `full` connects all coarse nodes within a complex

Rationale:

- improve global ligand and pocket communication
- reduce reliance on purely local radius graphs at the fragment level

Result:

- this was the most useful graph change tried so far
- at 50 rollout steps, compared with the local coarse baseline, the global
  coarse model reduced average rollout error:
  - sample prior RMSD: `4.00A -> 3.80A`
  - fresh prior RMSD: `4.02A -> 3.82A`
  - centroid distance and fragment-centroid RMSD improved as well

Representative config:

- `configs/overfit_neural1e_q1neqI_stochastic_priortime_effbs16_globalcoarse_wandb.yaml`

## 3.2 Explicit cut-bond edges

Patch:

- promoted `cut_bond_index` into an explicit atom-level message-passing edge
  type: `atom --cut--> atom`
- added `use_cut_bond_edges: true`

Rationale:

- give the model the exact rotatable-bond hinge, not only the wider
  triangulation stencil

Result:

- this did not beat the global-coarse-only model
- under the current best inference setting (`late`, `power=3`, `25` steps):
  - global coarse only: RMSD `3.48A`, centroid `1.00A`, frag RMSD `2.82A`
  - global coarse + cut-bond: RMSD `3.72A`, centroid `1.19A`, frag RMSD `2.94A`

Interpretation:

- explicit cut-bond edges are not obviously harmful
- but they are not the best current default
- the project should keep the simpler global-coarse baseline as the mainline
  model until a stronger win for cut-bond edges is demonstrated

Representative config:

- `configs/overfit_neural1e_q1neqI_stochastic_priortime_effbs16_globalcoarse_cutbond_wandb.yaml`

## 4. Inference-Time Time-Grid Ablations

The project now supports non-uniform time grids during ODE rollout:

- `uniform`
- `late`
- `early`

Here, `t=0` is the prior and `t=1` is the crystal end, so "Karras-like" density
near the data end maps to `late`, not `early`.

Patch:

- `src/inference/sampler.py`
- `scripts/inference.py`
- `scripts/visualize_trajectory.py`

Key results on the current best global-coarse checkpoint:

### 4.1 50-step schedules

- uniform 50:
  - RMSD `3.72A`
  - centroid `1.02A`
  - frag RMSD `2.84A`
- late, power 3, 50:
  - RMSD `3.57A`
  - centroid `1.03A`
  - frag RMSD `2.84A`
- early, power 3, 50:
  - worse mean RMSD than `late`

Conclusion:

- concentrating steps near `t=1` helps on average

### 4.2 Step-count sweep under the late schedule

- late, power 3, 100:
  - RMSD `3.75A`
- late, power 3, 75:
  - RMSD `3.74A`
- late, power 3, 50:
  - RMSD `3.57A`
- late, power 3, 25:
  - RMSD `3.48A`
  - centroid `1.00A`
  - frag RMSD `2.82A`
  - `<2A` success `31.2%`
- late, power 3, 10:
  - RMSD `3.88A`

Conclusion:

- the current model benefits from fewer, later-focused steps
- too many steps likely accumulate integration error
- too few steps under-resolve the field
- the current best rollout setting is `late + power=3 + 25 steps`

Artifacts:

- `outputs/schedule_ablation/globalcoarse_late_p3_s25`
- `outputs/schedule_ablation/cutbond_late_p3_s25`

## 5. Current Recommendation

Current recommended overfit baseline:

- model:
  - `omega_mode: neural_1e`
  - `pf_topology: full`
  - `ff_topology: full`
  - do not enable `use_cut_bond_edges` by default
- data:
  - `rotation_augmentation: ligand_uniform`
  - keep augmentation deterministic per complex when testing `q_target != I`
  - for the realistic overfit task, keep prior/time stochastic
- optimization:
  - effective batch `16`
  - stable `AdamW` setup without Muon
- inference:
  - `time_schedule: late`
  - `schedule_power: 3`
  - `num_steps: 25`

Best current config:

- `configs/overfit_neural1e_q1neqI_stochastic_priortime_effbs16_globalcoarse_wandb.yaml`

## 6. Open Questions

- stochastic rotation learning is still weaker than translation learning
- the gap between pointwise velocity metrics and closed-loop rollout quality is
  still present
- full-training generalization with the new topology and the new inference
  default still needs a fresh validation pass

## 7. Supporting Docs

- [CURRENT_STATUS.md](CURRENT_STATUS.md)
- [DATA_GRAPH_TOPOLOGY.md](DATA_GRAPH_TOPOLOGY.md)
