# Unified Model Ablation Plan

Last updated: 2026-03-22

Overfit-level experiments to validate before full training (19k complexes).
Results guide architecture/training decisions that are expensive to change later.

## Completed Experiments

| # | Experiment | Result | Decision |
|---|-----------|--------|----------|
| A1 | Head type: direct vs N-E vs hybrid | direct RMSD=1.63Å best, N-E cos_w=0.778 best | hybrid running |
| A2 | 2e removal (l2_dim=0) | cos_w drops, RMSD worse | keep 2e |
| A3 | 2o addition (l2o_dim=16) | no benefit, some complexes worse | don't add 2o |
| A4 | 2e+2o + N-E | peak cos_w=0.877 but RMSD unstable | not worth complexity |

## Planned Experiments (Priority Order)

### P1. Edge Topology — HIGHEST PRIORITY

**Why:** 72% of edges are protein-internal. TP conv processes all edges equally.
Additionally, **dynamic protein-ligand atom contact edges are missing** from the
current unified model — graph.py explicitly leaves them out for dynamic rebuild,
but UnifiedDataset never rebuilds them.

**Correct edge type mapping (from `src/preprocess/graph.py`):**

| Type ID | Code Name | Connects | Count (10gs) | % |
|---------|-----------|----------|-------------|---|
| 0 | `ligand_bond` | lig_atom ↔ lig_atom | 68 | 1.0 |
| 1 | `ligand_tri` | lig_atom ↔ lig_atom | 62 | 0.9 |
| 2 | `ligand_cut` | lig_atom ↔ lig_atom | 26 | 0.4 |
| 3 | `ligand_atom_frag` | lig_atom ↔ fragment | 118 | 1.8 |
| 4 | `ligand_frag_frag` | fragment ↔ fragment (full) | 182 | 2.7 |
| 5 | `protein_bond` | prot_atom ↔ prot_atom | 1600 | 24.0 |
| 6 | `protein_atom_ca` | prot_atom ↔ prot_CA (hierarchy) | 1616 | 24.3 |
| 7 | `protein_ca_ca` | prot_CA ↔ prot_CA (≤18Å) | 1588 | 23.8 |
| 8 | `protein_ca_frag` | **prot_CA ↔ fragment** (full bipartite) | 1400 | 21.0 |

**Critical observation:** There are NO protein_atom ↔ ligand_atom edges.
The only protein→ligand path is: prot_atom →(6)→ prot_CA →(8)→ fragment.
Protein chemical features (donor/acceptor/hydrophobe) reach ligand atoms
only indirectly through 2 hops via CA virtual nodes.

**Sub-experiments (ordered):**

1a. **Add dynamic prot_atom ↔ lig_atom contact edges** (4Å cutoff at pose t)
   - New edge_type=9. Rebuilt each forward pass from current atom positions.
   - Expected: direct chemical interaction signal → better docking accuracy.
   - Risk: adds edges + dynamic rebuild cost per forward.

1b. **Prune protein_ca_ca** (type 7, -24% edges)
   - CA backbone connectivity at 18Å cutoff — very dense (1588 edges).
   - CA nodes still connected to prot_atoms (type 6) and fragments (type 8).
   - Likely redundant: CA info can flow through prot_atom→CA→frag without CA-CA.

1c. **Prune protein_bond** (type 5, -24% edges)
   - Internal protein covalent bonds. Info already flows via atom→CA hierarchy.
   - But removes fine-grained protein chemistry propagation.

1d. **Prune protein_atom_ca** (type 6, -24% edges) + remove prot_atom nodes
   - Most aggressive: only keep CA + ligand nodes.
   - Loses protein heavy atom chemistry entirely — likely too aggressive.

**Metric:** cos_v, cos_w, rollout RMSD, speed (s/epoch), VRAM.

**Implementation:** 1a requires dataset/forward changes. 1b-1d are edge masks.

---

### P2. R_t Re-injection Ablation

**Why:** Per-layer R_t re-injection was critical in legacy model (cos_w +0.225).
But unified model has richer representation — may no longer be needed.
Removing saves per-layer clone + cat on 1o features.

**Sub-experiments:**
1. `per-layer` (current) — baseline
2. `init-only` — inject R_t at input, no per-layer re-injection
3. `off` — no R_t injection at all

**Metric:** cos_w is the key metric here. If init-only ≈ per-layer → simplify.

---

### P3. Irrep Ratio Rebalancing

**Why:** Current 256x0e + 32x1o + 32x1e + 16x2e is 89% scalar.
Omega prediction is fundamentally vector (1o/1e) — shifting capacity
from scalars to vectors may improve omega without adding parameters.

**Sub-experiments:**
1. Current: 256x0e + 32x1o + 32x1e + 16x2e (baseline)
2. Balanced: 128x0e + 64x1o + 64x1e + 16x2e (same total dim ~similar params)
3. Vector-heavy: 128x0e + 48x1o + 48x1e + 32x2e

**Metric:** cos_w improvement, cos_v maintenance, RMSD.

---

### P4. Layer Count (2 vs 4)

**Why:** With 955-node unified graph, receptive field is already large.
2 layers = ~2x speed, ~50% VRAM. If performance holds, big win.

**Sub-experiments:**
1. 2 layers
2. 4 layers (current baseline)
3. 3 layers (compromise if 2 is too few)

**Metric:** cos_v, cos_w, RMSD. Speed/VRAM comparison.

---

### P5. t-Decile Diagnostics

**Why:** Flow matching error may not be uniform over t ∈ [0,1].
Understanding where the model struggles informs loss weighting
and time schedule choices.

**Method:** Run eval with fixed t values (0.1, 0.2, ..., 0.9).
Log cos_v, cos_w per decile. Identify weak spots.

**Depends on:** Best architecture from P1-P4.

---

### P6. omega_weight Sweep

**Why:** loss_v ≈ 0.4 vs loss_w ≈ 1.0 at convergence.
Omega gradient signal may be too weak or too strong.

**Sub-experiments:** omega_weight ∈ {0.5, 1.0, 2.0}

**Depends on:** Architecture + loss form finalized.

---

### P7. Muon Optimizer

**Why:** Muon's Newton-Schulz orthogonalization may help TP conv weight
matrices. But won't fix structural bottlenecks.

**Sub-experiments:** use_muon=true vs false on best architecture.

**Depends on:** All structural decisions finalized.

---

### P8. sh_lmax=3

**Why:** Opens 3e/3o coupling paths. But O(L^4) cost increase.

**Decision rule:** Only try if P1-P6 results suggest angular representation
is still the bottleneck. If 2e + R_t injection + vector rebalancing solve
cos_w, skip this entirely.

---

## Experiment Execution Notes

- Each experiment: 2000 epochs overfit, 16 samples, eval with 10 seeds × 16 complexes
- Compare on: mean RMSD, median RMSD, <2Å rate, cos_v, cos_w
- Log to wandb project `flowfrag` with descriptive run names
- Save checkpoint for trajectory visualization on failure cases

## Decision Gate

Before moving to full training, we need:
- cos_v > 0.95 (overfit)
- cos_w > 0.80 (overfit)
- Mean RMSD < 1.5Å (overfit, 10-seed eval)
- <2Å rate > 80% (overfit)
- Epoch time < 1.5s (for feasible full training)
- VRAM < 50GB (to allow bs=8 or larger complexes)
