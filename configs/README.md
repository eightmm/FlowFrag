# Configs

## Training

| Config | Description |
|---|---|
| `train.yaml` | Full training config (19k complexes) |
| `train_unified_ne_contact_adamw_1000.yaml` | Newton-Euler + dynamic contacts, AdamW, 1000 epochs |

## Overfit (Debug)

| Config | Description |
|---|---|
| `overfit_unified.yaml` | Unified model overfit baseline |
| `overfit_unified_ne_contact.yaml` | Unified + Newton-Euler + dynamic contacts |
| `overfit_unified_newton_euler.yaml` | Unified + Newton-Euler (no contacts) |

## Ablation

| Config | Description |
|---|---|
| `abl_baseline_ne_contact_1000.yaml` | Ablation baseline (N-E + contacts, 1000 epochs) |
| `abl_layers2.yaml` | 2-layer ablation |
| `abl_prune_caca.yaml` | Remove C&alpha;-C&alpha; edges |
| `abl_rt_initonly.yaml` | R_t injection at init only (no per-layer) |

## Other Files

| File | Description |
|---|---|
| `split.json` | Train/val/test split by PDB ID |
