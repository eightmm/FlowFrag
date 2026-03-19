# FlowFrag Project Rules

## Environment

```bash
source .venv/bin/activate
# or: source ~/miniforge3/etc/profile.d/conda.sh && conda activate torch-2.9
```

## Training Commands

```bash
# Full training
python scripts/train.py --config configs/train.yaml

# Overfit test (debug)
python scripts/train.py --config configs/overfit.yaml

# Resume from checkpoint
python scripts/train.py --config configs/train.yaml --resume outputs/checkpoints/latest.pt
```

## Testing

```bash
pytest tests/ -v
pytest tests/test_geometry.py -v
```

## Key Architectural Decisions

### 1. Direct Fragment Velocity Prediction
- Predict (v_f, omega_f) directly per fragment
- No atom-force aggregation, no Newton-Euler
- Reduces gradient path complexity and numerical instability

### 2. Fragment-Based Flow Matching
- Ligands decomposed into rigid fragments (rotatable bond cuts)
- Each fragment has SE(3) state: (T, R)
- Velocities: translation v ∈ R^3, angular omega ∈ R^3

### 3. Protein as Static Context
- Residue-level only (CA coords + residue type)
- No ESM features in v1
- Pocket residues only (10A cutoff around ligand)

### 4. omega_f Frame Convention
**CRITICAL**: Pick one and never mix:
- Global angular velocity + left-multiplication for integration, OR
- Body-frame angular velocity + right-multiplication
- Write explicit tests to verify convention consistency

## Code Conventions

### Fragment Data
```python
T_frag: [N_frag, 3]         # Fragment centers
R_frag: [N_frag, 3, 3]      # Rotation matrices
atom_to_frag_idx: [N_atoms]  # Fragment assignment per atom
local_coords: [N_atoms, 3]   # Coords in fragment local frame
```

### Quaternion Convention
```python
# (w, x, y, z) - scalar first
q = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity
```

## Common Pitfalls

1. **omega frame mismatch** — body vs global frame inconsistency
2. **Single-atom fragments** — degenerate rotation, merge into neighbors
3. **Rotation matrix orthogonality** — quaternion round-trip for cleanup
4. **Fragment connectivity** — fragments can drift apart without torsion-adjacent edges
