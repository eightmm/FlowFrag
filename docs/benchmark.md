# Benchmark Evaluation

## Setup

- **ODE**: 25 steps, late schedule (power=3.0)
- **Prior**: translation &sigma; = 1.0 &Aring; centered at pocket CoM, rotation uniform on SO(3)
- **Pocket**: 8 &Aring; heavy-atom cutoff around the reference ligand
- **Flow matching**: linear interpolation (T), SLERP (R)

## Evaluation Regime

FlowFrag performs **pocket-conditioned re-docking**: the binding site is known and the model predicts the ligand pose within it. This is comparable to methods like SigmaDock, Uni-Mol, and AutoDock Vina re-docking. It is **not** directly comparable to blind docking methods (DiffDock, FlowSite) that must also identify the binding site.

## Datasets

| Dataset | Complexes | Ligand Format | Protein Format |
|---|---|---|---|
| Astex Diverse | 84 / 85 (1 parse failure) | MOL2 | Pocket PDB |
| PoseBusters v2 | 308 | SDF | Full-protein PDB |

PoseBusters v2 is the 308-complex subset from the Chemical Science paper, filtered from the original 428 Benchmark set.

## Pose Selection Strategies

| Strategy | Description | Requires Ground Truth |
|---|---|---|
| Oracle | argmin RMSD to crystal over N samples | Yes (upper bound) |
| Cluster | Centroid of largest cluster (2 &Aring; RMSD threshold) | No |
| Rank | Top-1 by combined score: s = &minus;E_vina &middot; p^&beta; | No |

See [scoring.md](scoring.md) for the Vina energy and validity score formulas.

## Results

### Astex Diverse (84 complexes, N=40)

| Refinement | Selection | Mean RMSD | Median RMSD | &lt;1&Aring; | &lt;2&Aring; | &lt;5&Aring; |
|---|---|---|---|---|---|---|
| None | Oracle | 1.67 | 1.25 | 34.5% | 81.0% | 96.4% |
| None | Cluster | 2.62 | 2.20 | 8.3% | 44.0% | 89.3% |
| None | Rank | 2.28 | 1.68 | 21.4% | 61.9% | 91.7% |
| MMFF | Oracle | **1.34** | **1.06** | **46.4%** | **85.7%** | 98.8% |
| MMFF | Cluster | 1.81 | 1.33 | 29.8% | 72.6% | 94.0% |
| MMFF | Rank | 1.87 | 1.29 | 35.7% | 73.8% | 92.9% |

### PoseBusters v2 (308 complexes, N=40)

| Refinement | Selection | Mean RMSD | Median RMSD | &lt;1&Aring; | &lt;2&Aring; | &lt;5&Aring; |
|---|---|---|---|---|---|---|
| None | Oracle | 1.50 | 1.34 | 26.9% | 83.8% | 98.7% |
| None | Cluster | 2.64 | 2.21 | 6.8% | 41.9% | 92.2% |
| None | Rank | 2.14 | 1.80 | 14.3% | 57.5% | 95.5% |
| MMFF | Oracle | **1.03** | **0.92** | **57.5%** | **96.1%** | **100%** |
| MMFF | Cluster | 1.25 | 1.14 | 39.9% | 89.0% | 100% |
| MMFF | Rank | 1.33 | 1.14 | 37.7% | 87.0% | 99.7% |

### N-Sample Scaling (&lt;2&Aring; success rate, N=10 &rarr; N=40)

| Refinement | Selection | Astex Diverse | PoseBusters v2 |
|---|---|---|---|
| None | Oracle | 67.9 &rarr; 81.0 (+13.1) | 68.8 &rarr; 83.8 (+15.0) |
| None | Cluster | 41.7 &rarr; 44.0 (+2.3) | 39.0 &rarr; 41.9 (+2.9) |
| None | Rank | 52.4 &rarr; 61.9 (+9.5) | 49.0 &rarr; 57.5 (+8.5) |
| MMFF | Oracle | 77.4 &rarr; 85.7 (+8.3) | 90.6 &rarr; 96.1 (+5.5) |
| MMFF | Cluster | 65.5 &rarr; 72.6 (+7.1) | 77.9 &rarr; 89.0 (+11.1) |
| MMFF | Rank | 69.0 &rarr; 73.8 (+4.8) | 79.9 &rarr; 87.0 (+7.1) |

## Key Observations

1. **MMFF refinement provides a consistent ~10-15 %p boost.** Flow matching produces reasonable fragment placements, but junction geometry across cut bonds is noisy. Vacuum MMFF fixes bond lengths and angles without moving the pose globally.

2. **PoseBusters is easier than Astex for this model.** MMFF+Oracle &lt;2&Aring;: 96.1% vs 85.7%. Astex ligands tend to be larger and more drug-like; PoseBusters includes many smaller cofactors.

3. **Cluster selection benefits most from denser sampling.** MMFF+Cluster gains +11 %p on PoseBusters from N=10 to N=40 &mdash; the largest gain across all combinations. The majority mode becomes cleaner with more samples.

4. **Rank selection benefits from MMFF.** Without refinement, physicochemical validity is low (broken bond geometry), inflating the penalty factor p^&beta;. After MMFF, p &approx; 1 for most poses and Vina energy becomes the dominant signal.

5. **Selection gap (Oracle &rarr; Rank)** is 12 %p on Astex and 9 %p on PoseBusters for MMFF. A trained confidence model could close most of this gap.

## Caveats

- These are **pocket-conditioned** results. Compare only against other pocket-conditioned methods (SigmaDock, Uni-Mol, AutoDock Vina re-dock), not blind docking.
- The **Oracle** column is an upper bound assuming a perfect confidence model. **Cluster** and **Rank** are realistic deployable numbers.
- Prior &sigma; = 1 &Aring; is relatively tight compared to SigmaDock's effective &sigma; &approx; 2.7 &Aring;. This makes the task closer to local refinement.

## References

- AutoDock Vina: Trott & Olson, *J. Comput. Chem.* 31(2):455-461, 2010.
- PoseBusters: Buttenschoen, Morris & Deane, *Chem. Sci.* 15:3130-3139, 2024.
- SigmaDock: Prat et al., arXiv:2511.04854, 2025.
