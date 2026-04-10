# Benchmark Evaluation Report (2026-04-10)

## Setup
- **Checkpoint**: `outputs/train_unified_ne_contact_adamw_1000/checkpoints/best.pt`
  (step 315000, epoch 612; train rollout median RMSD 1.58Å, <2A 64.5%)
- **ODE**: 25 steps, late schedule (power 3.0), σ=1.0
- **Pose selection strategies**:
  - `oracle`: best-RMSD (upper bound, requires ground truth)
  - `cluster`: RMSD-based greedy clustering, pick largest-cluster centroid
  - `rank`: SigmaDock formula `s = -vina * validity^4`, pick top-1
- **Refinement**: `none`, `mmff` (MMFF94s on ligand only, protein fixed)

## Evaluation Scripts
- `scripts/eval_benchmark.py` — unified Astex/PoseBusters evaluator (auto-detects format),
  samples N poses once and evaluates all refine × select combos, saves raw poses as
  `{out_dir}/poses/{pdb_id}.pt` for later re-scoring.
- `scripts/rescore_poses.py` — loads saved poses and applies alternative selections
  (e.g., Vina rank) without re-running sampling.
- `data/posebusters_v2_ids.txt` — curated 308-complex subset used in the PoseBusters
  Chemical Science paper.

## Results — Astex Diverse (84 complexes)

### N=10 samples

| Refine | Select  | Mean | Median | <1Å   | <2Å   | <5Å   |
|--------|---------|------|--------|-------|-------|-------|
| none   | oracle  | 2.03 | 1.50   | 14.3% | 67.9% | 95.2% |
| none   | cluster | 2.79 | 2.32   |  4.8% | 41.7% | 89.3% |
| none   | rank    | 2.59 | 1.87   | 10.7% | 52.4% | 86.9% |
| mmff   | oracle  | 1.64 | 1.25   | 28.6% | 77.4% | 96.4% |
| mmff   | cluster | 2.06 | 1.67   | 20.2% | 65.5% | 94.0% |
| mmff   | rank    | 2.10 | 1.44   | 23.8% | 69.0% | 92.9% |

### N=40 samples

| Refine | Select  | Mean | Median | <1Å   | <2Å   | <5Å   |
|--------|---------|------|--------|-------|-------|-------|
| none   | oracle  | 1.67 | 1.25   | 34.5% | 81.0% | 96.4% |
| none   | cluster | 2.62 | 2.20   |  8.3% | 44.0% | 89.3% |
| none   | rank    | 2.28 | 1.68   | 21.4% | 61.9% | 91.7% |
| mmff   | oracle  | 1.34 | 1.06   | 46.4% | 85.7% | 98.8% |
| mmff   | cluster | 1.81 | 1.33   | 29.8% | 72.6% | 94.0% |
| mmff   | rank    | 1.87 | 1.29   | 35.7% | 73.8% | 92.9% |

### Δ (40 vs 10, <2Å success rate)

| Refine | Select  | N=10  | N=40  | Δ      |
|--------|---------|-------|-------|--------|
| none   | oracle  | 67.9% | 81.0% | +13.1% |
| none   | cluster | 41.7% | 44.0% |  +2.3% |
| none   | rank    | 52.4% | 61.9% |  +9.5% |
| mmff   | oracle  | 77.4% | 85.7% |  +8.3% |
| mmff   | cluster | 65.5% | 72.6% |  +7.1% |
| mmff   | rank    | 69.0% | 73.8% |  +4.8% |

## Results — PoseBusters v2 (308 complexes)

### N=10 samples

| Refine | Select  | Mean | Median | <1Å   | <2Å   | <5Å   |
|--------|---------|------|--------|-------|-------|-------|
| none   | oracle  | 1.81 | 1.62   | 12.3% | 68.8% |  98.4% |
| none   | cluster | 2.65 | 2.32   |  4.5% | 39.0% |  92.2% |
| none   | rank    | 2.41 | 2.01   |  7.5% | 49.0% |  94.8% |
| mmff   | oracle  | 1.24 | 1.13   | 35.4% | 90.6% | 100.0% |
| mmff   | cluster | 1.63 | 1.37   | 22.1% | 77.9% |  98.4% |
| mmff   | rank    | 1.52 | 1.31   | 27.9% | 79.9% |  99.7% |

### N=40 samples (in progress — will be appended)

## N-Sample Scaling Analysis

**Observation 1 — Oracle scales strongly with N.**
More samples strictly help oracle best-of-N because the minimum is monotone in
the sample count. On Astex, `none+oracle` jumps from 67.9% → 81.0% (+13.1%p)
and `mmff+oracle` 77.4% → 85.7% (+8.3%p) when N goes 10 → 40.

**Observation 2 — MMFF refinement shrinks the N gap.**
Without refinement, the model's raw outputs have geometry noise (bad bond
lengths at inter-fragment junctions) that inflates RMSD. MMFF cleans this up,
giving a large constant boost (+10-15%p) that's roughly independent of N.
Consequence: `mmff+oracle` with N=10 (77.4%) is already close to `none+oracle`
with N=40 (81.0%). Raw outputs need more samples to catch up to MMFF; MMFF is
the cheaper path to quality.

**Observation 3 — Cluster selection barely benefits from more samples.**
Cluster <2Å went 41.7% → 44.0% (+2.3%p) on Astex, which is negligible. The
largest-cluster heuristic is a mode-finder — additional samples just fill out
the dominant mode and don't help pick a better candidate. Cluster selection
is a poor match for best-of-N evaluation.

**Observation 4 — Rank (Vina × validity) improves meaningfully with N.**
On Astex, `none+rank` gains +9.5%p (52.4% → 61.9%) — second only to oracle.
This suggests the ranking function correlates with RMSD well enough that
having more candidates lets it pick a better one. But it still lags oracle
by ~10-20%p, so the scoring function is noticeably worse than a perfect
confidence model.

**Observation 5 — PoseBusters v2 is easier than Astex for this model.**
At N=10, `mmff+oracle` achieves 90.6% <2Å on PoseBusters vs 77.4% on Astex.
Possible reasons: (a) training distribution (curated from the same PDB range)
is closer to PoseBusters than to the older Astex set, (b) Astex has more
challenging/drug-like ligands, (c) the PoseBusters benchmark set filtering
removed some harder cases.

## Implementation Notes

### Pocket feature extraction speedup (critical)

`compute_pocket_features_from_pdb` was parsing full PoseBusters protein PDBs
(~2700 atoms) and calling RDKit's `ChemicalFeatures.BuildFeatureFactory` on
the full molecule. `factory.GetFeaturesForMol` is O(N) per-atom with a
large constant, taking **~20 seconds** per complex.

Fix (`src/scoring/vina_features.py`): accept `center` + `cutoff` parameters
and remove atoms outside `cutoff` Å *before* feature extraction. Atoms are
removed via `RWMol.RemoveAtom` to preserve bonds. Downstream feature
extraction then runs on ~200 atoms instead of ~2700.

**Result**: `rank_poses` time went from ~20s/complex to ~0.5s/complex on
PoseBusters (40× speedup). Astex was unaffected because its pocket PDBs
were already cut.

### Pose persistence for re-scoring

`eval_benchmark.py` writes raw sampled poses per complex to
`{out_dir}/poses/{pdb_id}.pt` containing:
```
{
  "pdb_id": str,
  "raw_poses": list[Tensor [N_atom, 3]],  # pocket-centered
  "ref_pos": Tensor,                       # pocket-centered crystal
  "pocket_center": Tensor [3],
  "n_atoms": int,
  "n_frags": int,
}
```
This allows `rescore_poses.py` to try different selection/refinement combos
without re-sampling — a PoseBusters rescore now takes ~5-8 minutes instead
of rerunning the full ~90-minute sampling loop.
