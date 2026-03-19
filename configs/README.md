# Configs

This directory contains the configs used for the current fragment-flow
implementation and its overfit ablations.

## Recommended Current Baseline

Use this first when checking whether the current model still overfits correctly:

- `overfit_neural1e_q1neqI_stochastic_priortime_effbs16_globalcoarse_wandb.yaml`

This is the current best debug baseline because it combines:

- `neural_1e`
- stochastic prior/time
- effective batch `16`
- dense coarse graphs

## Overfit Ladder

- `overfit_neural1e_q1neqI_strict.yaml`
  - strict deterministic overfit
  - fixed complex, fixed target, fixed prior, fixed time
- `overfit_neural1e_q1neqI_strict_wandb.yaml`
  - W&B version of the strict deterministic overfit
- `overfit_neural1e_q1neqI_tonly_wandb.yaml`
  - fixed target, fixed prior, variable time
- `overfit_neural1e_q1neqI_priorbank_wandb.yaml`
  - fixed bank of priors and times
- `overfit_neural1e_q1neqI_priorbank_large_stable_wandb.yaml`
  - larger, more stable prior-bank variant
- `overfit_neural1e_q1neqI_stochastic_priortime_wandb.yaml`
  - fully stochastic prior/time overfit
- `overfit_neural1e_q1neqI_stochastic_priortime_effbs16_wandb.yaml`
  - stochastic prior/time overfit with larger effective batch
- `overfit_neural1e_q1neqI_stochastic_priortime_effbs16_globalcoarse_wandb.yaml`
  - current best topology baseline
- `overfit_neural1e_q1neqI_stochastic_priortime_effbs16_globalcoarse_cutbond_wandb.yaml`
  - cut-bond edge ablation on top of the global-coarse baseline

## Earlier or Reference Overfit Configs

- `overfit.yaml`
  - original generic overfit config
- `overfit_large.yaml`
  - earlier larger-model overfit config
- `overfit_localframe.yaml`
  - local-frame rotation ablation
- `overfit_localframe_strict.yaml`
  - strict deterministic local-frame ablation
- `overfit_neural1e_q1neqI.yaml`
  - early `q_target != I` neural_1e overfit config

## Training Configs

- `train.yaml`
  - base training config
- `train_sigma1.yaml`
  - sigma-1 training variant
- `train_sigma5.yaml`
  - sigma-5 training variant
- `train_analytic_sigma1.yaml`
  - analytic-omega sigma-1 training
- `train_neural1e_ligand_uniform_fixed.yaml`
  - draft training config for neural_1e with ligand-uniform augmentation

## Related Docs

- `../docs/CURRENT_STATUS.md`
- `../docs/experiment_log.md`
