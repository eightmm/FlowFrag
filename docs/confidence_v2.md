# Confidence v2 — Multi-config Training

Status: **in progress** (data gen ~85% done as of 2026-04-26).

## Motivation

v1 confidence was trained on a single sampling config (σ=5, γ=0.4, 12 ODE
steps).  The pocket-cutoff sweep (`outputs/cutoff_sweep/`) showed the
confidence head is **OOD-sensitive** — at cutoff=5Å it scores worse than Vina
(Astex 37.6% vs Vina 45.9%, PB 29.9% vs Vina 32.1%).  More directly, real
det-inference (σ=3, γ=0, 25 steps) is *not* in v1 training distribution either.
v2 trains across multiple sampler configs to make calibration robust.

## Sampling configs (after compute trim)

Skipped configs (recorded in `data/conf_train_v2/manifest.json`):

- `s5_g04_s25` — redundant with cfg1 (only step count differs)
- `s7_g0_s25` — broad prior, nice-to-have but not critical

Active 2 configs:

| Tag | σ | γ | num_steps | poses/cx | rationale |
|---|---|---|---|---|---|
| `s3_g0_s25` | 3.0 | 0.0 | 25 | 5 | det inference default (real user setup) |
| `s5_g04_s12` | 5.0 | 0.4 | 12 | 5 | v1 training distribution (backward compat) |

= 10 poses/complex × 16,420 train + 280 val = **164k train + 2.8k val poses**.

## Training data composition

**Initial plan** was v1 (20 poses) + v2 (10 poses) = 30 poses/cx, ~492k poses,
but `load_all_shards` concatenates everything into a single ndarray and the
combined dataset peaked at ~127 GB RSS during numpy concat → killed by OOM
on 125 GB host.

**Actually used (v2-only):** 16,420 × 10 poses = **164k poses, ~13 GB peak RAM**,
in `data/conf_train_v2/merged/` (66 shards).

Effective config breakdown (train, by pose):
- σ=5, γ=0.4, 12 step (v2 cfg2): **82k poses (50%)**
- σ=3, γ=0, 25 step (v2 cfg1): **82k poses (50%)**

Trade-off vs initial plan:
- −80 GB peak RAM (fits)
- 50/50 split between σ=5 (v1-style) and σ=3 (det inference) — actually
  *better* balance for calibration than 83/17.
- 164k vs 492k poses — fewer training samples, but the σ=3 distribution is
  now equally represented.

## Pose RMSD distributions (sanity check, partial v2 data)

Per-pose RMSD <2Å rate:
- v1 (σ=5,γ=0.4,12s, n=328k): 30.1%
- v2 cfg1 (σ=3,γ=0,25s, n=82k): **59.1%** ← much sharper
- v2 cfg2 (σ=5,γ=0.4,12s, n=7.5k partial): 29.7% ✓ matches v1, sanity check OK

Per-complex oracle (best of N):
- v1 (n=20): 86.2% <2Å
- v2 cfg1 (n=5): 88.5% <2Å ← higher with fewer poses, matches σ=3 quality
- v2 cfg2 (n=5): 63.7% <2Å (only 5 vs v1's 20 poses, undersamples)

## Pipeline (auto-chained)

```
gen_conf_v2_chain.sh
├── train cfg_s3_g0_s25 (~14h, DONE)
├── train cfg_s5_g04_s12 (~7h, in progress)
├── val cfg_s3_g0_s25 (~10 min)
├── val cfg_s5_g04_s12 (~5 min)
└── merge → data/conf_{train,val}_v2/merged/
       │
       ▼  (post_v2_train_v2.sh detects ALL DONE)
augment merged with v1 shards (v1_ prefix symlinks)
       │
       ▼
train_confidence.py
  --train_shards_dir data/conf_train_v2/merged
  --val_shards_dir data/conf_val_v2/merged
  --hidden 512 --trunk_depth 4 --head_depth 3
  --pool_mode attention --n_pool_queries 4
  --total_steps 8000 --val_every 500
  --batch_complexes 32 --lr 3e-4 --muon_lr 0.02
  --run_name conf_v2 --out_dir outputs/conf_v2_models
       │
       ▼  (~30 min)
eval_benchmark.py on Astex det + PB v2 det
  (σ=3, γ=0, 25 steps, cutoff=8, --confidence_ckpt last)
  → outputs/conf_v2_eval/{astex,pb}_det/results.json
```

## Logs

- `outputs/logs/conf_v2_gen.log` — gen chain orchestration
- `outputs/logs/conf_v2_train_<tag>.log` — per-config gen log
- `outputs/logs/conf_v2_post.log` — post-gen orchestration (merge + train + eval)
- `outputs/logs/conf_v2_train.log` — confidence training log
- `outputs/logs/conf_v2_eval_{astex,pb}.log` — eval logs

## Expected outcome

Headline target (vs v1 baseline at cutoff=8, det inference):

| Dataset | v1 <2Å | v2 target | Reason |
|---|---|---|---|
| Astex det | 87.1% | 88-90% | Already near top-5 ceiling; calibration on σ=3 may add 1-3pp |
| PB v2 det | 70.5% | 73-76% | Selection room (top-5 ceiling 76.6%); calibration on σ=3 + 1.5x training data |

Floor expectation: v2 ≥ v1 within noise — multi-config + more data shouldn't hurt.

## Caveats

1. The 83/17 split favors σ=5 — if calibration on σ=3 (default inference) is what we
   need most, this dataset is still imbalanced.  Future v3 could sample uniformly
   per-config or upweight σ=3 shards.
2. Pocket cutoff is fixed at 8Å (main model trained at 8±2).  v2 doesn't fix the
   cutoff-OOD failure mode — that needs main model retraining.
3. No physics features (clash, Vina sub-terms, strain) — still planned for v3.
4. v2 cfg2 + v1 effectively double-count the σ=5 distribution.  Could trim v1 to
   5 poses/cx for balance, but user explicitly chose "use as much data as possible".
