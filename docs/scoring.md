# Scoring Functions

FlowFrag uses a combined score for pose ranking that integrates AutoDock Vina binding energy with PoseBusters-style physicochemical validity.

## 1. AutoDock Vina Scoring

Implementation: `src/scoring/vina_scoring.py`

Computes pairwise ligand-pocket atom distances, then sums five energy terms weighted by the Vina preset.

### Notation

- **d_ij** = Euclidean distance between ligand atom *i* and pocket atom *j* (&Aring;)
- **R_ij** = r_vdw(i) + r_vdw(j) = sum of van der Waals radii (&Aring;)
- **&Delta;_ij** = d_ij &minus; R_ij = distance past steric contact
- **w** = Vina preset weights (`src/scoring/vina_params.py`)

### Energy Terms

| Term | Formula | Weight |
|---|---|---|
| Gauss 1 | exp(&minus;(&Delta;/0.5)&sup2;) | &minus;0.03558 |
| Gauss 2 | exp(&minus;((&Delta;&minus;3)/2)&sup2;) | &minus;0.00516 |
| Repulsion | &Delta;&sup2; if &Delta; &lt; 0, else 0 | +0.84024 |
| Hydrophobic | Piecewise linear, gated by both atoms being hydrophobic | &minus;0.03507 |
| H-bond | Piecewise linear, gated by donor-acceptor pairing | &minus;0.58744 |

### Full Score

```
E_inter = Σ_ij [ w₁·gauss1 + w₂·gauss2 + w_rep·repulsion
               + w_hyd·hydrophobic + w_hbd·hbond ]

E_vina  = E_inter / (1 + w_rot · N_rot)
```

where N_rot is the number of rotatable bonds and w_rot = 0.05846 (torsional entropy penalty). Lower (more negative) = stronger predicted binding.

## 2. Physicochemical Validity

Implementation: `src/scoring/pose_ranking.py`

Five internal-geometry checks on the ligand pose. Each produces a pass fraction in [0, 1]; the validity score is the mean.

| Check | Criterion |
|---|---|
| **Bond lengths** | 0.75 &le; d_bond &le; 2.2 &Aring; |
| **Bond angles** | &vert;angle &minus; ideal&vert; &le; 15&deg; (ideal from hybridization: SP=180&deg;, SP&sup2;=120&deg;, SP&sup3;=109.5&deg;) |
| **Internal clash** | d_ij &ge; 0.75 &middot; (r_vdw_i + r_vdw_j) for non-bonded, non-1,3 pairs |
| **Chirality** | sign(volume) matches reference conformer at each chiral center |
| **Ring planarity** | max deviation &le; 0.25 &Aring; for aromatic rings (via SVD) |

```
p = (bond_pass + angle_pass + clash_pass + chiral_pass + planar_pass) / 5
```

p &isin; [0, 1]. A geometrically perfect pose has p = 1.

## 3. Combined Score

Implementation: `src/scoring/pose_ranking.py`

Following the SigmaDock formula:

```
s = −E_vina · p^β
```

with &beta; = 4 (default). Poses are ranked by descending *s*.

- **&minus;E_vina** rewards strong binding (negative Vina &rarr; positive score).
- **p^&beta;** penalizes geometrically broken poses: p = 0.8 &rarr; 0.8&sup4; &approx; 0.41 multiplicative penalty.

## 4. Usage in the Pipeline

| Script | What It Computes |
|---|---|
| `eval_benchmark.py` | Sampling + oracle/cluster selection (no Vina scoring) |
| `rescore_poses.py` | Loads saved poses, optionally applies MMFF, runs `rank_poses` |
| `dock.py --rank` | Same `rank_poses` call in the docking entry point |

## References

- AutoDock Vina: Trott & Olson, *J. Comput. Chem.* 31(2):455-461, 2010. [doi:10.1002/jcc.21334](https://doi.org/10.1002/jcc.21334)
- PoseBusters: Buttenschoen, Morris & Deane, *Chem. Sci.* 15:3130-3139, 2024. [doi:10.1039/D3SC04185A](https://doi.org/10.1039/D3SC04185A)
- SigmaDock: Prat et al., arXiv:2511.04854, 2025.
