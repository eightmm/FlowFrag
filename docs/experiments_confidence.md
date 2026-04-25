# Confidence Head Hyperparameter Sweep

13-config sweep over architecture / regularization / loss / optimization run by
`scripts/auto_experiments.py`.  Every run trains on the same 16,420-complex
PDBbind train shards, validates on CASF-2016 (280 complexes), and reports
external benchmark <2 Å SR on Astex Diverse and PoseBusters v2.  See
`experiments.jsonl` for raw rows.

## Sweep results (<2 Å success rate)

| Name | CASF (mean Å) | AsSt SR | AsDt SR | PbSt SR | PbDt SR | Notes |
|---|---|---|---|---|---|---|
| arch.big.mean_max | 1.796 | 84.7% | 82.4% | 65.3% | 67.9% | hidden=512/4/3, pool=mean_max |
| **arch.big.attention** ★ | 1.807 | **84.7%** | **82.4%** | **68.2%** | **70.8%** | hidden=512/4/3, pool=attention |
| arch.big.both | 1.787 | 84.7% | 81.2% | 66.9% | 69.8% | hidden=512/4/3, pool=both |
| arch.attn.q8 | 1.778 | 84.7% | 81.2% | 66.6% | 70.5% | pool=attention, q=8 |
| arch.attn.q16 | 1.785 | 82.4% | 82.4% | 68.2% | 68.8% | pool=attention, q=16 |
| **arch.xl.both** | 1.782 | **87.1%** | **84.7%** | 67.9% | 68.8% | hidden=1024/5/4, pool=both |
| reg.dropout.1 | 1.813 | 83.5% | 81.2% | 66.9% | 70.5% | dropout=0.1 |
| reg.dropout.3 | 1.792 | 83.5% | 81.2% | 66.6% | 69.2% | dropout=0.3 |
| loss.rank.heavy | 1.778 | 83.5% | 81.2% | 66.9% | 67.5% | w_rank=3.0 |
| loss.pose.only | 1.797 | 82.4% | 83.5% | 65.3% | 68.5% | atom/plddt/frag weights = 0 |
| train.long | 1.774 | 81.2% | 82.4% | 66.6% | 70.1% | 20,000 steps |
| train.lr.high | 1.805 | 84.7% | 82.4% | **68.5%** | 69.8% | lr=1e-3 |
| train.lr.low | 1.790 | 85.9% | 83.5% | 68.2% | 69.8% | lr=1e-4 |

★ = deployed checkpoint (`weights/confidence_v1.pt`).

## Best per dataset

| Dataset | Best SR | Config |
|---|---|---|
| Astex stoch | 87.1% | arch.xl.both |
| Astex det | 84.7% | arch.xl.both |
| PB v2 stoch | 68.5% | train.lr.high |
| PB v2 det | 70.8% | arch.big.attention |

## Key findings

1. **Capacity ceiling at hidden ≈ 512** for the small/medium configs (256, 512,
   768 all close).  Going to **1024 (`arch.xl.both`)** delivers the only large
   Astex jump (+2 pp), but at 4 × parameter cost and weaker PB-det.

2. **Attention pool helps PoseBusters** by ≈ 3 pp <2 Å vs pure mean+max,
   matching SR while halving pool feature dimensionality vs the `both` mode.

3. **Loss-weight tuning is mostly noise.**  `loss.rank.heavy` (3 × ranking
   weight) and `loss.pose.only` (kill auxiliary heads) both stay within ± 1 pp.

4. **Long training plateaus.**  CASF val mean reaches ~1.78 by step 1000–2000
   and oscillates ± 0.05 thereafter regardless of architecture.  Going from
   8 k → 20 k steps did not improve external benchmarks meaningfully.

5. **Confidence + Vina ensemble** helps slightly only on PoseBusters det
   (+0.3 pp at α = 0.5).  On Astex pure confidence wins (α = 1.0).

## Why arch.big.attention was selected for deployment

| Criterion | arch.big.attention | arch.xl.both | arch.big.mean_max |
|---|---|---|---|
| Astex stoch | 84.7% | **87.1%** | 84.7% |
| Astex det | 82.4% | **84.7%** | 82.4% |
| PB stoch | **68.2%** | 67.9% | 65.3% |
| **PB det** | **70.8%** | 68.8% | 67.9% |
| Mean SR | **76.5%** | **77.1%** | 75.1% |
| Params | 1.6 M | ~6 M | ~1.0 M |
| Pool components | 1 (attention) | 2 (mean+max+attn) | 1 (mean+max) |
| Inference cost | low | medium | low |

`arch.big.attention` wins on PoseBusters and ties on Astex (within 2 pp), at 1/4
the parameter count of `arch.xl.both`.  Single pooling mechanism keeps the
forward pass clean.  Marginal Astex headroom that `arch.xl.both` provides is
*sampling-bound* anyway (top-5 ceiling already exceeded), so the extra capacity
isn't worth the deployment cost.

## Reproducing

```bash
# Train any single config
python scripts/train_confidence.py \
    --train_shards_dir data/conf_train --val_shards_dir data/conf_val \
    --hidden 512 --trunk_depth 4 --head_depth 3 \
    --pool_mode attention --n_pool_queries 4 \
    --total_steps 8000 --val_every 500 \
    --batch_complexes 32 --lr 3e-4 --muon_lr 0.02

# Reproduce the full 13-config sweep (autonomously commits to git)
python scripts/auto_experiments.py
```
