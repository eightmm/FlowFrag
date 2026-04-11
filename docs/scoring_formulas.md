# FlowFrag Scoring Functions

Detailed reference for the Vina energy and PoseBusters-style validity score
used by `scripts/rescore_poses.py` and `src/scoring/pose_ranking.py`.

## 1. AutoDock Vina scoring

Implementation: `src/scoring/vina_scoring.py::vina_scoring`

Compute pairwise ligand-pocket distances, then an atomwise energy from five
terms weighted by the Vina preset.

### Notation
- `d_ij` = Euclidean distance between ligand atom *i* and pocket atom *j* (Å)
- `R_ij = r_vdw(i) + r_vdw(j)` = sum of van-der-Waals radii (Å)
- `Δ_ij = d_ij − R_ij` = distance past steric contact
- `w_*` = Vina (or Vinardo) weights (`src/scoring/vina_params.py`)

### Energy terms (Vina preset)

| Term | Formula | Weight |
|------|---------|--------|
| `gauss1` | `exp(−(Δ/0.5)²)` | `−0.03558` |
| `gauss2` | `exp(−((Δ−3.0)/2.0)²)` | `−0.00516` |
| `repulsion` | `Δ² if Δ<0 else 0` | `+0.84024` |
| `hydrophobic` | `1` if `Δ≤0.5`; `(1.5−Δ)` if `0.5<Δ<1.5`; else `0`. Gated by both atoms being hydrophobic. | `−0.03507` |
| `hbond` | `1` if `Δ≤−0.7`; `(−Δ/0.7)` if `−0.7<Δ<0`; else `0`. Gated by donor–acceptor pairing. | `−0.58744` |

### Full score

```
E_inter = Σ_ij [ w₁·gauss1 + w₂·gauss2 + w_rep·repulsion
               + w_hyd·hydrophobic + w_hbd·hbond ]

E_vina  = E_inter / (1 + w_rot · N_rot_bonds)
```

where `N_rot_bonds` is the ligand's number of rotatable bonds and
`w_rot = 0.05846` (Vina torsional entropy penalty).

Lower (more negative) = stronger predicted binding.

### Vinardo variant (unused by default)

- `gauss1` uses width `0.8`: `exp(−(Δ/0.8)²)`
- `gauss2` dropped
- `hydrophobic` extended to `Δ<2.5` with linear tail `(1 − Δ/2.5)`
- `hbond` cutoff tightened to `Δ≤−0.6`

### Pocket-atom filtering (speed)

`compute_pocket_features_from_pdb(..., center, cutoff)` keeps only protein atoms
within `cutoff` Å of `center` *before* running RDKit's
`ChemicalFeatures.BuildFeatureFactory`, which is O(N) per atom with a
large constant. For full PoseBusters protein PDBs (~2.7k atoms), this cuts
per-complex scoring from ~20 s to ~0.5 s.

## 2. PoseBusters-style physicochemical validity

Implementation: `src/scoring/pose_ranking.py::check_physicochemical_validity`

Five internal-geometry checks on the ligand pose (protein not involved).
Each check produces a pass fraction in `[0, 1]`; the validity score is the
mean of the five fractions.

### Checks

1. **Bond lengths** — pass iff `0.75 ≤ d_bond ≤ 2.2` Å.
   - `bond_pass = 1 − n_bad_bonds / n_bonds`

2. **Bond angles** — for each atom, for every pair of neighbors, compute the
   angle and compare to hybridization ideal:
   - SP → 180°, SP² → 120°, SP³ → 109.5°, SP³D/SP³D² → 90°
   - Pass iff `|angle − ideal| ≤ 15°`.
   - `angle_pass = 1 − n_bad_angles / n_angles`

3. **Internal steric clash** — for every atom pair that is NOT 1–2 (bonded) or
   1–3 (shares a neighbor), check:
   - Pass iff `d_ij ≥ 0.75·(r_vdw_i + r_vdw_j)`.
   - `clash_pass = 1 − n_clashes / n_checked_pairs`

4. **Tetrahedral chirality** — for each chiral center, compute the signed
   volume of the tetrahedron formed by the center and its first three
   neighbors, compare sign against the reference mol's conformer.
   - Pass iff `sign(ref_vol) == sign(pred_vol)` (no chirality inversion).
   - `chiral_pass = 1 − n_chiral_wrong / n_chiral_centers`

5. **Aromatic ring planarity** — for every aromatic ring, fit a plane via SVD
   of centered coordinates and compute `max_dev = smallest_singular_value / √n_ring_atoms`.
   - Pass iff `max_dev ≤ 0.25` Å.
   - `planar_pass = 1 − n_nonplanar / n_aromatic_rings`

### Aggregate

```
p = (bond_pass + angle_pass + clash_pass + chiral_pass + planar_pass) / 5
```

`p ∈ [0, 1]`. A pose with all checks perfect has `p = 1`.

Note: these checks approximate the PoseBusters suite but do NOT include
protein-ligand intermolecular checks (clash with pocket, covalent contact),
which are handled implicitly by the Vina `repulsion` term.

## 3. Combined score (SigmaDock formula)

Implementation: `src/scoring/pose_ranking.py::rank_poses`

```
s = −E_vina · p^β
```

with `β = 4.0` (default, `validity_beta` argument).

- Higher `s` = better pose.
- `−E_vina` rewards strong binding (negative Vina → positive `−E_vina`).
- `p^β` acts as a multiplicative gate: an invalid pose (`p ≈ 0.8`) incurs
  `0.8⁴ ≈ 0.41` penalty, suppressing geometrically broken poses even if
  they have good Vina energy.

Poses are ranked by descending `s`; top-1 is reported as the `rank`
selection.

## 4. When each is used

| Pipeline | What's computed |
|----------|-----------------|
| `eval_benchmark.py` | Sampling + oracle/cluster selection. No Vina, no validity. |
| `rescore_poses.py` | Loads saved raw poses, optionally applies MMFF, runs `rank_poses` to produce top-1 by combined score. |
| `scripts/dock.py --rank` | Same `rank_poses` call in the production docking entry point. |

## 5. References

- AutoDock Vina: Trott & Olson, J. Comput. Chem. 31(2):455-461, 2010.
  <https://doi.org/10.1002/jcc.21334>
- Vinardo: Quiroga & Villarreal, PLoS ONE 11(5):e0155183, 2016.
  <https://doi.org/10.1371/journal.pone.0155183>
- PoseBusters: Buttenschoen, Morris & Deane, Chem. Sci. 15:3130-3139, 2024.
  <https://doi.org/10.1039/D3SC04185A>
- SigmaDock combined score (`−E · p^β`): Prat et al., arXiv:2511.04854, 2025.
