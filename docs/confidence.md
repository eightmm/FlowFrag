# Confidence Model

A separate per-pose confidence head that ranks the N=40 sampled poses produced
by FlowFrag's main flow-matching model and selects the one most likely to be
near-native.  It replaces (and significantly outperforms) the previous
Vina-based ranker.

**Headline result** (deployed `confidence_v1`, attention-pool, 1.6 M params):

> Reranking the 40 sampled poses with the confidence head improves <2 Å
> success rate by **+7 to +9 pp** over Vina ranking on Astex Diverse and
> PoseBusters v2 — under FlowFrag's evaluated sampling setup
> (σ = 5, γ = 0.4, 12 ODE steps in training; σ = 3, γ = 0 at inference).
> On Astex this approaches the 40-sample top-5 ceiling; on PoseBusters it
> still has 5–8 pp of selection headroom (see [Bottleneck](#selection-ceiling-and-bottleneck)).

| Dataset | Vina baseline (<2 Å) | **Confidence v1** (<2 Å) | Δpp |
|---|---|---|---|
| Astex stoch | 75.3% | **84.7%** | +9.4 |
| Astex det | 74.1% | **81.2%** | +7.1 |
| PoseBusters v2 stoch | 61.0% | **67.9%** | +6.9 |
| PoseBusters v2 det | 62.7% | **70.5%** | +7.8 |

Astex n=85 means each complex is ~1.2 pp; deltas of 1–2 pp on Astex are within
single-complex noise.  PoseBusters n=308 is more stable.

## Architecture

`src/models/confidence.py::ConfidenceHead`

Inputs (per atom, computed at *t* = 1 with the candidate pose injected as the
ligand's current state):

- `atom_scalar` : `[N_atom, 512]` – the *l = 0* (SO(3)-invariant) block of the
  main model's final hidden state.
- `atom_norms` : `[N_atom, 192]` – per-channel norms of *l = 1o, 1e, 2e, 2o*
  blocks (also SO(3)-invariant).
- `pose_stats` : `[6]` per pose – ‖v_pred‖ and ‖ω_pred‖ aggregates (mean, max,
  p95) of the model's velocity prediction at *t* = 1.

Trunk: 4-layer MLP (512 hidden, SiLU, dropout 0.2) processes per-atom inputs.

Three heads on top of the trunk:

| Head | Pooling | Output |
|---|---|---|
| atom | none — per atom | `(disp_log1p, plddt_logit)` |
| fragment | scatter-mean over atoms by fragment id | `(rmsd_log1p, bad_logit)` |
| pose | **attention pool** (PMA: 4 learnable queries × multihead attn over atoms) | `(pose_rmsd_log1p, prob_2A_logit)` |

The pose head's *attention pool* is the only structural change vs. an
all-mean+max baseline; it gave +3 pp <2 Å on PoseBusters det in the sweep.

## Training

`scripts/train_confidence.py` + `src/training/confidence.py`

| Item | Value |
|---|---|
| Train data | PDBbind 2020 train minus 43 Astex-overlapping complexes → **16,420 complexes**, 20 sampled poses each (σ = 5, γ = 0.4, 12 ODE steps), 328k poses total |
| Validation | **CASF-2016 val (280 complexes)** after Astex overlap filter — disjoint from train |
| Test (external) | Astex Diverse (85), PoseBusters v2 (308) |
| Optimizer | Muon + AdamW hybrid (matches main model) |
| LR schedule | Trapezoidal WSD (10 % warmup / 60 % stable / 30 % cooldown) |
| LR (AdamW / Muon) | 3e-4 / 0.02 |
| Batch | 32 complexes ×~ 20 poses each |
| Steps | 8,000 |
| Loss | atom Huber-log1p + atom pLDDT BCE + frag Huber-log1p + frag-bad BCE + pose Huber-log1p + pose-prob BCE + within-complex pairwise margin ranking |

The `experiments.jsonl` records a 13-config sweep that informed deployment
selection (see `docs/experiments_confidence.md`).

## Inference

```python
import torch, yaml
from src.models.unified import UnifiedFlowFrag
from src.models.confidence import ConfidenceHead
from src.inference.preprocess import load_ligand, preprocess_complex
from src.inference.sampler import sample_unified
from src.inference.confidence_features import extract_per_atom_features

# 1. Sample poses
poses = sample_unified(main_model, graph, lig_data, meta, num_samples=40,
                      num_steps=25, translation_sigma=3.0, time_schedule='late',
                      device=device)

# 2. Per-atom features at t = 1
feats = extract_per_atom_features(
    main_model, graph, lig_data, meta,
    raw_poses=[p['atom_pos_pred'] for p in poses],
    crystal_pocket_centered=torch.zeros(meta['num_atom'], 3),  # unused at inference
    device=device, t_eval=1.0,
)

# 3. Score with confidence head
ckpt = torch.load('weights/confidence_v1.pt')
head = ConfidenceHead(**{k: ckpt[k] for k in
    ('scalar_dim', 'norms_dim', 'pose_stats_dim', 'hidden',
     'trunk_depth', 'head_depth', 'dropout')},
    pool_mode=ckpt.get('pool_mode', 'attention'),
    n_pool_queries=ckpt.get('n_pool_queries', 4),
).to(device)
head.load_state_dict(ckpt['state_dict']); head.train(False)

# normalize + run
# ... see scripts/eval_confidence_benchmarks.py for the canonical inference loop

# 4. argmin(pred_pose_rmsd) over 40 poses → best
```

For end-to-end docking with confidence selection, just use `scripts/dock.py`
with `--num_samples > 1` — `--confidence_ckpt` defaults to
`weights/confidence_v1.pt`.  Pass `--confidence_ckpt ""` to disable and fall
back to first-pose-out-of-N.

For benchmark evaluation, pass the deployed checkpoint explicitly:

```bash
python scripts/eval_benchmark.py \
  --data_dir /path/to/Astex-or-PoseBusters \
  --checkpoint weights/best.pt \
  --config configs/train_v3_b200.yaml \
  --num_samples 40 \
  --confidence_ckpt weights/confidence_v1.pt
```

## Selection ceiling and bottleneck

| Dataset | Oracle <2 Å | Top-5 ranker ceiling | **Confidence v1** | Selection gap | Sampling gap |
|---|---|---|---|---|---|
| Astex stoch | 92.9% | 84.7% | **84.7%** | 0 pp (saturated) | 8.2 pp |
| Astex det | 92.9% | 85.9% | 81.2% | 4.7 pp | 7.0 pp |
| PB v2 stoch | 83.8% | 76.3% | 67.9% | **8.4 pp** | 7.5 pp |
| PB v2 det | 84.4% | 76.6% | 70.5% | **6.1 pp** | 7.8 pp |

Astex selection is at or past the top-5 ceiling — the remaining gap is dominated
by **sampling**: the 40 poses don't include a near-native one.

PoseBusters still has 5–8 pp of selection headroom before hitting top-5 ceiling.
Adding **explicit physics features** (per-atom clash count, Vina sub-terms,
PoseBusters-style validity) is the next planned upgrade and is expected to close
most of that gap.

## Caveats

1. **Single sampling config in training data.**  The 16k training poses were
   generated with σ = 5 and γ = 0.4.  Confidence calibration may degrade if you
   sample with very different settings.  In practice σ = 3 and γ = 0 (the
   benchmark default) work well — there is no measurable drop on Astex/PB —
   but a user changing temperature, ODE steps, or schedule can drift the
   ranking.  Treat the headline result as conditional on FlowFrag's evaluated
   sampler.
2. **PDB-ID disjointness ≠ full external generalization.**  Train (PDBbind),
   val (CASF-2016), Astex and PoseBusters share PDB IDs only — sequence /
   target-family / ligand-scaffold overlap was *not* analyzed.  A reviewer
   wanting strong "external" claims should run that analysis before publishing.
3. **No score combination.**  We deploy pure `argmin pred_pose_rmsd`; combining
   with Vina gave only marginal gain on PoseBusters (+0.3 pp at α = 0.5) and
   would require fixing α on validation before benchmark eval to avoid
   leakage — which would dilute the gain anyway.
4. **CASF-2016 best step**.  Different datasets reach their best at different
   training steps (CASF best ≈ step 3000–8000).  We ship the step-8000
   checkpoint which is roughly stable across the plateau.
5. **No physics features yet.**  Per-atom clash / Vina sub-terms / strain are
   *not* in the input.  Adding them is the planned v2 upgrade and is the most
   promising route to closing the PoseBusters selection gap (currently 5–8 pp
   below top-5 ceiling).
6. **Confidence head only — main model unchanged.**  The improvement comes
   entirely from better selection over the same 40 samples.  If the main
   model's representations have a systematic blind spot, the confidence head
   inherits it.

## Deployment summary

- Checkpoint: `weights/confidence_v1.pt` (~18 MB)
- Inference cost: ≈ 20 ms per complex on RTX 6000 Pro (40 poses, including
  one extra main-model forward at *t* = 1 to extract per-atom features)
- Replaces Vina ranking, no extra dependencies beyond the main model.
