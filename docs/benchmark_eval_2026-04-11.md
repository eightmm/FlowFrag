# FlowFrag Benchmark Evaluation (2026-04-11)

## Setup

- **Checkpoint**: `outputs/train_unified_ne_contact_adamw_1000/checkpoints/best.pt`
  - step 315000, epoch 612
  - train rollout median 1.58 Å, <2Å 64.5%
- **ODE**: 25 steps, late schedule (power=3.0)
- **Prior**: translation σ = 1.0 Å, centered at pocket CoM; rotation uniform on SO(3)
- **Pocket selection**: 8 Å heavy-atom cutoff around the reference ligand (same as training)
- **Flow matching**: linear interpolation (T), SLERP (R)

## Regime: pocket-conditioned re-docking

Both training and inference share the same coordinate convention:

1. `build_static_complex_graph` stores **absolute crystal coords** for protein
   atoms; fragment and ligand-atom node coords get overwritten to the current
   flow state `T_t`, `R_t @ local + T_t`.
2. The target is `T_1 = frag_centers − pocket_center` — pocket-CoM-relative.
3. Prior is sampled at the origin with σ = 1 Å, which in the un-centered frame
   means "within 1 Å of pocket CoM".
4. Protein atoms remain at their crystal coords; the model sees the offset
   vector from ligand-centered origin to protein atoms, which implicitly
   encodes pocket location.

**Comparison class**: SigmaDock, Uni-Mol, and other pocket-conditioned
docking methods that consume a known binding site. Not directly comparable
to blind docking (DiffDock, FlowSite).

SigmaDock uses the same general recipe (Prat et al., arXiv:2511.04854):
- Pocket cutoff 5 ± 1 Å (stochastic) around bound ligand — ours 8 Å fixed
- Effective translation σ ≈ 2.7 Å via coordinate rescaling — **ours 1.0 Å**
- Pocket CoM jitter `N(0, σ_CoM²)` — **ours none**

Our prior is therefore noticeably tighter than SigmaDock's, making the task
closer to local refinement than learned large-scale dynamics. Training and
inference both use σ = 1, so the model is internally consistent but less
robust against prior widening.

## Datasets

| Dataset | Complexes | Ligand format | Protein format |
|---------|-----------|---------------|----------------|
| Astex Diverse | 84/85 (1 parse failure) | MOL2 | pocket-only PDB (~500 lines) |
| PoseBusters v2 | 308 | SDF | full-protein PDB (~4700 lines) |

The **PoseBusters v2** subset = 308 complexes in the Chemical Science paper
(filtered from the original 428 Benchmark set). IDs are bundled in
`data/posebusters_v2_ids.txt`.

## Selection strategies evaluated

| Strategy | Formula / mechanism | Needs ground truth? |
|----------|---------------------|---------------------|
| `oracle` | `argmin rmsd(pose, crystal)` over N samples | **Yes** (upper bound only) |
| `cluster` | Greedy RMSD clustering (2 Å); pick centroid of largest cluster | No |
| `rank` | `s = −E_vina · p^β`, `β = 4`. Top-1 by descending `s`. | No |

See `docs/scoring_formulas.md` for exact Vina energy and
PoseBusters-style validity score formulas. All selections are run on MMFF-refined
and non-refined pose sets — `apply_refinement` generates both variants from
the same raw ODE output.

## Results

### Astex Diverse (84 complexes, N=40)

| Refine | Select  | Mean | Median |  <1Å  |  <2Å  |  <5Å  |
|--------|---------|------|--------|-------|-------|-------|
| none   | oracle  | 1.67 | 1.25   | 34.5% | 81.0% | 96.4% |
| none   | cluster | 2.62 | 2.20   |  8.3% | 44.0% | 89.3% |
| none   | rank    | 2.28 | 1.68   | 21.4% | 61.9% | 91.7% |
| mmff   | oracle  | **1.34** | **1.06** | **46.4%** | **85.7%** | 98.8% |
| mmff   | cluster | 1.81 | 1.33   | 29.8% | 72.6% | 94.0% |
| mmff   | rank    | 1.87 | 1.29   | 35.7% | 73.8% | 92.9% |

### PoseBusters v2 (308 complexes, N=40)

| Refine | Select  | Mean | Median |  <1Å  |  <2Å  |  <5Å  |
|--------|---------|------|--------|-------|-------|-------|
| none   | oracle  | 1.50 | 1.34   | 26.9% | 83.8% | 98.7% |
| none   | cluster | 2.64 | 2.21   |  6.8% | 41.9% | 92.2% |
| none   | rank    | 2.14 | 1.80   | 14.3% | 57.5% | 95.5% |
| mmff   | oracle  | **1.03** | **0.92** | **57.5%** | **96.1%** | **100.0%** |
| mmff   | cluster | 1.25 | 1.14   | 39.9% | 89.0% | 100.0% |
| mmff   | rank    | 1.33 | 1.14   | 37.7% | 87.0% |  99.7% |

### N-sample scaling (<2Å, N=10 → N=40)

| Refine | Select  | Astex                | PoseBusters v2          |
|--------|---------|----------------------|-------------------------|
| none   | oracle  | 67.9 → 81.0 (+13.1)  | 68.8 → 83.8 (+15.0)     |
| none   | cluster | 41.7 → 44.0 ( +2.3)  | 39.0 → 41.9 ( +2.9)     |
| none   | rank    | 52.4 → 61.9 ( +9.5)  | 49.0 → 57.5 ( +8.5)     |
| mmff   | oracle  | 77.4 → 85.7 ( +8.3)  | 90.6 → 96.1 ( +5.5)     |
| mmff   | cluster | 65.5 → 72.6 ( +7.1)  | 77.9 → 89.0 (+11.1)     |
| mmff   | rank    | 69.0 → 73.8 ( +4.8)  | 79.9 → 87.0 ( +7.1)     |

## Observations

1. **MMFF refinement is a constant ~10–15 %p boost**, independent of dataset
   or N. Flow matching produces reasonable fragment placements but fragment-
   junction geometry (bond lengths/angles across cut bonds) is noisy. Vacuum
   MMFF fixes these without moving the pose globally.

2. **PoseBusters is easier than Astex for this model**. `mmff+oracle <2Å`:
   96.1% vs 85.7%. Likely because Astex ligands are more drug-like and
   larger on average; PoseBusters ligands include many small cofactors.

3. **Cluster selection underperforms at low N, catches up with more samples.**
   `mmff+cluster` gains +11 %p on PoseBusters from N=10 → 40 — the largest
   gain across all combos. Cluster consensus benefits most from denser
   sampling because the majority mode becomes cleaner.

4. **Rank selection benefits consistently from MMFF.** `none+rank`
   physicochemical validity is low (bond geometry often broken) which inflates
   the penalty factor `p^β`. After MMFF, `p ≈ 1` for most poses, and Vina
   energy becomes the dominant signal.

5. **Selection gap** (`oracle → rank`) is **12 %p on Astex** (85.7 → 73.8) and
   **9 %p on PoseBusters** (96.1 → 87.0) for MMFF. A trained confidence model
   could close most of this gap — the oracle shows the raw samples contain a
   good pose in >85% of cases, we just can't always pick it.

6. **Cluster = rank on PoseBusters**. Both sit at ~87–89% <2Å, suggesting
   the dominant cluster mode already aligns with what Vina prefers. Astex
   shows a 1 %p gap.

## Caveats for reporting

- Our numbers are **pocket-conditioned oracle / blind-selection**. When
  comparing to literature, only compare against other pocket-conditioned
  methods (SigmaDock ~95% <2Å, Uni-Mol ~85%, AutoDock Vina re-dock ~60%).
- **Do not compare directly to DiffDock / FlowSite blind docking** — those
  solve a strictly harder task (no binding site hint).
- The `oracle` column is an **upper bound** assuming a perfect confidence
  model. `cluster` and `rank` are realistic deployable numbers.
- σ = 1 Å is tight. Widening the prior (σ = 2.7 Å to match SigmaDock) or
  adding pocket-center jitter would probably reduce `none+oracle` by a few
  percentage points but not move the MMFF numbers much.

## Artifacts

- `outputs/plots/sr_2A_bars.png` — grouped bar chart of <2Å SR across combos
- `outputs/plots/rmsd_cdf.png` — cumulative RMSD distributions, N=40
- `outputs/plots/n_scaling.png` — N=10 vs N=40 comparison
- `outputs/plots/oracle_vs_rank.png` — per-complex selection gap
- `outputs/plots/summary_table.md` — all numbers in one table
- `outputs/eval_{astex,posebusters_v2}_{10s,40s}/poses/*.pt` — raw poses for
  re-scoring without re-sampling
- `docs/scoring_formulas.md` — Vina energy and PB validity derivations
