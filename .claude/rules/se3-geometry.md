# SE(3) Geometry Rules

## Quaternion Operations

### Convention: (w, x, y, z) - Scalar First
```python
# Identity quaternion
q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0])

# Components
w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
```

### Key Functions (src/geometry/se3.py)

```python
# Conversion
R = quaternion_to_matrix(q)      # [*, 4] → [*, 3, 3]
q = matrix_to_quaternion(R)      # [*, 3, 3] → [*, 4]

# Composition
q_composed = quaternion_multiply(q1, q2)  # q1 ∘ q2

# Interpolation (geodesic)
q_t = quaternion_slerp(q0, q1, t)  # t ∈ [0, 1]

# Random sampling
q_random = sample_uniform_quaternion(n)  # Uniform on SO(3)
```

## Rigid Body Transforms

### Fragment Realization
```python
# Global coords from local coords + fragment pose
x_global = R @ x_local + T

# In code:
R_atoms = R_frag[atom_to_frag_idx]  # [N_atoms, 3, 3]
T_atoms = T_frag[atom_to_frag_idx]  # [N_atoms, 3]
x_global = torch.einsum('nij,nj->ni', R_atoms, local_coords) + T_atoms
```

### Velocity to Pose Update (ODE Step)
```python
# Translation update
T_new = T + dt * v

# Rotation update (via exponential map)
omega_quat = axis_angle_to_quaternion(dt * omega)
q_new = quaternion_multiply(omega_quat, q)
R_new = quaternion_to_matrix(q_new)
```

## Flow Matching on SE(3)

### Prior Sampling (t=0)
```python
# Translation: Gaussian around pocket center
T_prior = pocket_center + sigma * torch.randn(N_frag, 3)

# Rotation: Uniform on SO(3)
R_prior = quaternion_to_matrix(sample_uniform_quaternion(N_frag))
```

### Interpolation (t ∈ [0, 1])
```python
# Translation: Linear
T_t = (1 - t) * T_0 + t * T_1

# Rotation: SLERP (geodesic on SO(3))
q_t = quaternion_slerp(q_0, q_1, t)
R_t = quaternion_to_matrix(q_t)
```

### Ground Truth Velocity
```python
# Translation velocity
v_gt = (T_1 - T_0) / (1 - t + eps)

# Angular velocity (from SLERP derivative)
# ω such that R_t' = [ω]× R_t
omega_gt = compute_angular_velocity(q_0, q_1, t)
```

## Newton-Euler Aggregation

### Atom Forces → Fragment Velocities
```python
# Total force
F_frag = scatter_add(f_atom, atom_to_frag_idx)

# Translation velocity (mass-weighted mean)
v_frag = F_frag / frag_sizes

# Torque
r_arm = x_atom - T_frag[atom_to_frag_idx]
torque = cross(r_arm, f_atom)
tau = scatter_add(torque, atom_to_frag_idx)

# Inertia tensor
I = scatter_add(|r|²I₃ - r⊗r, atom_to_frag_idx)

# Angular velocity
omega_frag = solve(I, tau)
```

## Common Bugs

1. **Quaternion sign ambiguity**: q and -q represent same rotation
   - Fix: Ensure `q.dot(q_ref) > 0` before SLERP

2. **Gimbal lock with Euler angles**: Never use Euler angles
   - Always use quaternions or rotation matrices

3. **Non-orthogonal rotation matrices**: Accumulated numerical error
   - Fix: Re-orthogonalize via SVD or quaternion round-trip

4. **Wrong SLERP direction**: Taking long path instead of short
   - Fix: Flip quaternion sign if dot product < 0
