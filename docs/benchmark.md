# Benchmark Evaluation

## Setup

- **Input**: **SMILES only** — 3D starting conformers are re-embedded per complex via RDKit ETKDGv3 + MMFF94s. Crystal ligand coordinates are used only for the pocket-center reference and as the RMSD ground truth.
- **SMILES source**: RCSB Chemical Component Dictionary fetched per-PDB by `scripts/fetch_astex_smiles.py` and `scripts/fetch_pb_smiles.py`; cached in `data/astex_smiles.json` and `data/pb_smiles.json`.
- **ODE**: 10 steps, late schedule (power = 3.0).
- **Prior**: translation &sigma; = 3.0 &Aring; centered at pocket, rotation uniform on SO(3).
- **Pocket center**: centroid of protein residue virtual nodes within 8 &Aring; of the crystal ligand (matches the training definition).
- **Sampling**: N = 40 poses per complex.
- **Refinement**: vacuum MMFF94s with position restraints (constraint strength 50, tolerance 0.5 &Aring;) so the pose is only locally relaxed, not dragged back to the gas-phase minimum.
- **RMSD**: heavy-atom, **symmetry-aware** via RDKit `rdMolAlign.CalcRMS` (no alignment). For complexes where the SMILES-derived topology differs from the crystal (partial build, alternate tautomer), atoms are matched by MCS and RMSD is computed on the matched subset.

## Evaluation Regime

FlowFrag performs **pocket-conditioned re-docking**: the binding site is known and the model predicts the ligand pose within it. Comparable to Uni-Mol, AutoDock Vina re-dock, SigmaDock. **Not** directly comparable to blind docking (DiffDock, FlowSite).

Because the input is SMILES-only (starting 3D conformer is re-embedded from scratch), the atom ordering of the docked mol generally differs from the crystal; matching is done per complex before RMSD is computed.

## Datasets

| Dataset | Complexes evaluated | Ligand source (crystal) | Protein source |
|---|---|---|---|
| Astex Diverse | 84 / 85 (1 parse failure: 1u1c) | MOL2 | Pocket PDB |
| PoseBusters v2 | 308 / 308 | SDF | Full-protein PDB |

## Pose Selection

| Strategy | Description | Requires ground truth? |
|---|---|---|
| Oracle | argmin RMSD to crystal over N samples | Yes (upper bound) |
| Cluster | Centroid of largest RMSD cluster (2 &Aring; threshold) | No |
| Vina | Top-1 by combined score `s = -E_vina &middot; p^&beta;`, &beta; = 4 | No |

`E_vina` is the AutoDock Vina energy computed on the predicted pose, `p` is the PoseBusters-style physicochemical validity score. See [scoring.md](scoring.md).

## Results

### Astex Diverse (N = 40, SMILES input)

| Refinement | Selection | Mean | Median | &lt;1&Aring; | &lt;2&Aring; | &lt;5&Aring; |
|---|---|---|---|---|---|---|
| None | Oracle | 1.01 | 0.81 | 66.7% | 91.7% | 98.8% |
| None | Cluster | 1.96 | 1.18 | 38.1% | 71.4% | 90.5% |
| None | Vina | 2.02 | 1.27 | 34.5% | 72.6% | 90.5% |
| MMFF | Oracle | **0.97** | **0.77** | **70.2%** | **91.7%** | **98.8%** |
| MMFF | Cluster | 1.99 | 1.09 | 40.5% | 69.0% | 90.5% |
| MMFF | Vina | 1.96 | 1.27 | 35.7% | 71.4% | 90.5% |

### PoseBusters v2 (N = 40, SMILES input)

| Refinement | Selection | Mean | Median | &lt;1&Aring; | &lt;2&Aring; | &lt;5&Aring; |
|---|---|---|---|---|---|---|
| None | Oracle | 1.40 | 1.05 | 46.1% | 80.2% | 98.4% |
| None | Cluster | 2.81 | 1.94 | 17.9% | 53.2% | 83.4% |
| None | Vina | 2.80 | 2.01 | 16.9% | 49.7% | 86.0% |
| MMFF | Oracle | **1.38** | **1.04** | **46.8%** | **79.5%** | **98.7%** |
| MMFF | Cluster | 2.81 | 1.93 | 19.5% | 51.6% | 84.1% |
| MMFF | Vina | 2.76 | 1.83 | 18.5% | **54.9%** | 84.7% |

## Key Observations

1. **Oracle ceiling is high.** Med 0.81 &Aring; on Astex and 1.05 &Aring; on PoseBusters with SMILES-only input means the sampler produces a near-crystal pose within 40 tries for most complexes. The remaining challenge is selection.

2. **MMFF refinement improves the median by 0.05–0.2 &Aring; and is otherwise neutral.** With position restraints it only removes local strain (bond lengths, short clashes) without moving the ligand; without restraints the gas-phase minimum pulls the ligand tens of &Aring; away from the pocket (the earlier unrestrained MMFF regressed PoseBusters Mean to 22 &Aring;).

3. **Vina selection vs Cluster selection.** On Astex they are within 1–3 %p of each other. On PoseBusters v2, `MMFF + Vina` is the best realistic combination (&lt;2&Aring; 54.9%), slightly edging out `MMFF + Cluster` (51.6%). Vina + MMFF is the most physically grounded — pose geometry is healthy *and* interaction energy drives the pick.

4. **Selection gap (Oracle → Vina)** is 20 %p on Astex and 25 %p on PoseBusters v2 for `&lt;2&Aring;`. A trained pose-confidence model should close most of this.

5. **Astex is harder than PB for the oracle, easier for selection.** Astex has more varied drug-like ligands with many rotatable bonds (wider pose distribution → higher oracle ceiling but harder to cluster/score). PB v2 is biased toward smaller cofactors and fragments (higher concentration around the true pose, so cluster/score work better relative to oracle).

## Visualization

Per-sample ODE trajectories can be rendered with `scripts/viz_traj.py`:

```bash
uv run python scripts/dock.py \
    --protein /path/to/pocket.pdb --ligand "<SMILES>" \
    --pocket_center x,y,z \
    --checkpoint weights/best.pt --config configs/train_v3_b200.yaml \
    --num_samples 1 --num_steps 25 --save_traj --out_dir outputs/traj_<id>

uv run python scripts/viz_traj.py \
    --traj outputs/traj_<id>/traj_0.sdf \
    --protein /path/to/pocket.pdb \
    --crystal_ligand /path/to/crystal_ligand.{sdf,mol2} \
    --out_dir outputs/viz_<id>
```

The renderer shows:
- **Main 3D view**: predicted ligand as ball-and-stick (atoms colored per fragment), crystal ligand as translucent green ghost, protein heavy atoms as a "contact heatmap" (red glow for atoms &lt; 2.5 &Aring; of the ligand, fading to faint grey past 7 &Aring; — pocket residues "blink" as the ligand arrives).
- **Fragment centroid trails**: each fragment leaves a colored tail as it translates; rigid-body flow becomes visible per fragment.
- **2D ligand sketch (top-left)**: RDKit depiction of the molecular graph with identical fragment coloring, so you can map rings in the 3D view to atoms in the chemical structure.
- **RMSD inset (top-right)**: full curve in grey, progress up to the current frame in red.
- **Flow time bar (bottom)**: t = 0 → 1 filled proportionally.

### Gallery

Oracle-best trajectory (best of 10 samples by RMSD) per complex, 25 ODE steps, SMILES-only input. RMSD is heavy-atom vs crystal with symmetry-aware matching.

#### Astex Diverse

| Complex | Atoms / Fragments | Oracle RMSD | Trajectory |
|---|---|---|---|
| `1hq2` (rigid purine) | 14 / 2 | **0.39 &Aring;** | ![1hq2](assets/traj_1hq2.gif) |
| `1sqn` (steroid) | 22 / 2 | **0.50 &Aring;** | ![1sqn](assets/traj_1sqn.gif) |
| `2bsm` (inhibitor) | 27 / 6 | **0.54 &Aring;** | ![2bsm](assets/traj_2bsm.gif) |
| `1gkc` (peptidomimetic) | 22 / 6 | **0.70 &Aring;** | ![1gkc](assets/traj_1gkc.gif) |
| `1p62` (bifunctional) | 18 / 3 | **0.73 &Aring;** | ![1p62](assets/traj_1p62.gif) |
| `1v0p` (flexible) | 30 / 7 | **0.87 &Aring;** | ![1v0p](assets/traj_1v0p.gif) |
| `1oyt` (mid-size) | 30 / 4 | **0.92 &Aring;** | ![1oyt](assets/traj_1oyt.gif) |

#### PoseBusters v2

| Complex | Atoms / Fragments | Oracle RMSD | Trajectory |
|---|---|---|---|
| `7WJB_BGC` (sugar) | 12 / 2 | **0.46 &Aring;** | ![7WJB_BGC](assets/traj_7WJB_BGC.gif) |
| `7UAW_MF6` (lipid) | 35 / 2 | **0.67 &Aring;** | ![7UAW_MF6](assets/traj_7UAW_MF6.gif) |
| `8D39_QDB` (mid-size) | 17 / 4 | **1.02 &Aring;** | ![8D39_QDB](assets/traj_8D39_QDB.gif) |
| `7NP6_UK8` (flexible) | 32 / 7 | **1.16 &Aring;** | ![7NP6_UK8](assets/traj_7NP6_UK8.gif) |
| `7N6F_0I1` (multi-ring) | 26 / 4 | **1.27 &Aring;** | ![7N6F_0I1](assets/traj_7N6F_0I1.gif) |
| `7KM8_WPD` (very flexible) | 29 / 8 | **2.42 &Aring;** | ![7KM8_WPD](assets/traj_7KM8_WPD.gif) |

These are hand-picked to span easy → hard (fragment count 2 → 8). All still land within 5 &Aring; of the crystal pose even at 8 fragments.

## Caveats

- These are **pocket-conditioned** results. Compare only against other pocket-conditioned methods, not blind docking.
- **Pocket center leaks crystal-ligand position** (residues within 8 &Aring; of the crystal). This matches training but is an oracle-level assumption. For a harder, realistic setting, use apo-structure + fpocket / P2Rank center.
- **Oracle** is an upper bound assuming a perfect confidence model. **Cluster** and **Vina** are realistic deployable numbers.
- 1u1c failed at the protein PDB parsing stage; the reported Astex denominator is 84.

## References

- AutoDock Vina: Trott & Olson, *J. Comput. Chem.* 31(2):455–461, 2010.
- PoseBusters: Buttenschoen, Morris & Deane, *Chem. Sci.* 15:3130–3139, 2024.
- SigmaDock: Prat et al., arXiv:2511.04854, 2025.
