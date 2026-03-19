# cuEquivariance Usage Guide

## Overview

[cuEquivariance](https://docs.nvidia.com/cuda/cuequivariance/) is NVIDIA's library for high-performance SE(3)-equivariant neural networks. It accelerates models like DiffDock, MACE, Allegro, and NEQUIP through optimized CUDA kernels.

**Key Benefits:**
- Up to 10x speedup over e3nn for MACE-like models
- Optimized tensor products, spherical harmonics, and triangle operations
- Native support for DiffDock-style architectures

## Installation

```bash
# Core + PyTorch bindings
pip install cuequivariance cuequivariance-torch

# CUDA kernels (choose cu12 or cu13)
pip install cuequivariance-ops-torch-cu12
```

## Memory Layout: `mul_ir` vs `ir_mul`

### Layout Definitions
- **`mul_ir`** (e3nn-compatible): Shape `[batch, multiplicity, irrep_dim]` - irrep components (x,y,z) contiguous
- **`ir_mul`** (cuEquivariance-native): Shape `[batch, irrep_dim, multiplicity]` - all x-terms contiguous, then y, then z

### Performance Impact
```
e3nn:          10 vectors → [10, 3] tensor (mul_ir)
cuEquivariance: 10 vectors → [3, 10] tensor (ir_mul) - FASTER
```

**Best Practice:** Use `ir_mul` throughout your pipeline to avoid expensive transpose operations. The `TransposeIrrepsLayout` module handles conversions when needed.

### Current Project Layout
This project uses `mul_ir` for e3nn compatibility:
```python
# src/models/layers/equivariant.py
self.linear = Linear(irreps_in, irreps_out, layout=cue_base.mul_ir)
self.tp_message = FullyConnectedTensorProduct(
    in_irreps, edge_irreps, out_irreps,
    layout_in1=cue_base.mul_ir,
    layout_in2=cue_base.mul_ir,
    layout_out=cue_base.mul_ir
)
```

## Core API Reference

### Irreps Definition
```python
import cuequivariance as cue

# O(3) group irreps: multiplicity x ell (parity: e=even, o=odd)
scalars = cue.Irreps("O3", "32x0e")                    # 32 scalars
vectors = cue.Irreps("O3", "16x1o")                    # 16 odd vectors
mixed = cue.Irreps("O3", "64x0e + 32x1o + 16x2e")     # scalars + vectors + rank-2

# Pseudo-vectors (even parity l=1)
pseudo_vec = cue.Irreps("O3", "8x1e")
```

### SphericalHarmonics
```python
from cuequivariance_torch import SphericalHarmonics

# Compute Y_l for l=0,1,2
sh = SphericalHarmonics(
    ls=[0, 1, 2],      # List of ell values
    normalize=True      # Normalize input vectors
)

# Usage
edge_vec = positions[dst] - positions[src]  # [E, 3]
edge_sh = sh(edge_vec)                       # [E, sh_dim]
```

### Linear Layer
```python
from cuequivariance_torch import Linear

linear = Linear(
    irreps_in=cue.Irreps("O3", "64x0e + 32x1o"),
    irreps_out=cue.Irreps("O3", "128x0e + 64x1o"),
    layout=cue.mul_ir  # or cue.ir_mul for native performance
)

output = linear(input)  # [N, out_dim]
```

### FullyConnectedTensorProduct
```python
from cuequivariance_torch import FullyConnectedTensorProduct

tp = FullyConnectedTensorProduct(
    irreps_in1=cue.Irreps("O3", "64x0e + 32x1o"),   # Node features
    irreps_in2=cue.Irreps("O3", "1x0e + 1x1o"),    # Edge SH
    irreps_out=cue.Irreps("O3", "64x0e + 32x1o"),   # Output

    # Layout options
    layout_in1=cue.mul_ir,
    layout_in2=cue.mul_ir,
    layout_out=cue.mul_ir,

    # Weight options
    shared_weights=True,      # Share weights across batch
    internal_weights=True,    # Create learnable parameters

    # Performance
    method="fused_tp",        # Use CUDA kernels (default)
    # method="naive",         # PyTorch fallback
)

# Forward: (batch, in1_dim), (batch, in2_dim) -> (batch, out_dim)
output = tp(node_feats[src], edge_features)
```

### FullyConnectedTensorProductConv (DiffDock-style)
```python
from cuequivariance_torch.layers import FullyConnectedTensorProductConv

conv = FullyConnectedTensorProductConv(
    in_irreps=cue.Irreps("O3", "64x0e + 32x1o"),
    sh_irreps=cue.Irreps("O3", "1x0e + 1x1o + 1x2e"),
    out_irreps=cue.Irreps("O3", "64x0e + 32x1o"),

    # MLP for weight generation from edge embeddings
    mlp_channels=[64, 128, 128],
    mlp_activation=nn.GELU(),

    batch_norm=True,
    layout=cue.mul_ir,
)

# Forward with graph structure
output = conv(
    src_features=node_feats,      # [N_src, in_dim]
    edge_sh=edge_sh,              # [E, sh_dim]
    edge_emb=edge_embeddings,     # [E, emb_dim]
    graph=(src_idx, dst_idx),     # COO format
    reduce="sum"                  # or "mean"
)
```

### BatchNorm
```python
from cuequivariance_torch.layers import BatchNorm

bn = BatchNorm(
    irreps=cue.Irreps("O3", "64x0e + 32x1o"),
    layout=cue.mul_ir
)

output = bn(input)  # Normalizes scalars and vector magnitudes separately
```

## Triangle Operations (AF2-style)

```python
from cuequivariance_torch import (
    triangle_attention,
    triangle_multiplicative_update,
    attention_pair_bias,
)

# Triangle attention (pair representation)
out = triangle_attention(q, k, v, bias, mask)

# Triangle multiplicative update
out = triangle_multiplicative_update(x, outgoing=True)

# Attention with pairwise bias (diffusion models)
out = attention_pair_bias(s, q, k, v, z, mask)
```

## AMP / Mixed Precision Compatibility

**cuEquivariance has limited AMP support.** The CUDA kernels require specific dtypes and may not work correctly with `torch.cuda.amp.autocast()`.

**Workaround Options:**
1. **Disable AMP entirely** (current approach in this project)
2. **Selective autocast** - exclude cuEquivariance modules:
   ```python
   with torch.cuda.amp.autocast(enabled=False):
       equivariant_output = self.tp(x1.float(), x2.float())
   ```
3. **Use fp32 for equivariant layers** only

## Performance Optimization Tips

### 1. Minimize Layout Conversions
```python
# BAD: Convert layout every forward pass
x_ir_mul = TransposeIrrepsLayout(irreps)(x)  # Expensive!
out = tp(x_ir_mul)

# GOOD: Use consistent layout throughout
# Set layout=cue.ir_mul everywhere
```

### 2. Sort Species Indices (for indexed linear)
```python
# Sorting enables faster indexed_linear kernels (8.8x speedup)
sorted_idx = torch.argsort(atom_types)
features = features[sorted_idx]
```

### 3. Use Fused Kernels
```python
# Automatically uses CUDA kernels when available
tp = FullyConnectedTensorProduct(..., method="fused_tp")

# Force PyTorch fallback (debugging)
tp = FullyConnectedTensorProduct(..., method="naive")
```

### 4. Batch Operations
cuEquivariance kernels achieve best performance with large batch sizes. Consider:
- Increasing batch size
- Batching multiple graphs together
- Using gradient accumulation for effective larger batches

## Current Usage in FlowFrag

### File: `src/models/layers/equivariant.py`

**Imported Components:**
```python
import cuequivariance as cue_base
from cuequivariance_torch import (
    SphericalHarmonics,
    FullyConnectedTensorProduct,
    Linear,
)
from cuequivariance_torch.layers import BatchNorm as EquivariantBatchNorm
```

**Key Classes:**
- `EquivariantAdaLN`: Adaptive LayerNorm for scalars + vector magnitude scaling
- `EquivariantActivation`: SiLU for scalars, norm-based gating for vectors
- `EquivariantDropout`: Separate dropout for scalars and vectors
- `EquivariantMLPBlock`: Pre-LN block with residual
- `GatingEquivariantLayer`: Tensor product message passing
- `UnifiedEquivariantNetwork`: Full SE(3)-equivariant GNN

**Irreps Pattern:**
```python
# Input: scalar-only from joint trunk
self.in_irreps = cue.Irreps("O3", f"{input_dim}x0e")

# Hidden: scalars + odd/even vectors
self.hidden_irreps = cue.Irreps(
    "O3", f"{hidden_scalar_dim}x0e + {hidden_vector_dim}x1o + {hidden_vector_dim}x1e"
)

# Output: true vectors for velocity
self.out_irreps = cue.Irreps("O3", f"{output_vector_dim}x1o")

# Spherical harmonics
self.sh_irreps = cue.Irreps("O3", "1x0e + 1x1o + 1x2e")  # l=0,1,2
```

## Resources

- [Official Documentation](https://docs.nvidia.com/cuda/cuequivariance/)
- [GitHub Repository](https://github.com/nvidia/cuequivariance)
- [MACE Operations Tutorial](https://docs.nvidia.com/cuda/cuequivariance/tutorials/pytorch/MACE.html)
- [Segmented Tensor Product Guide](https://docs.nvidia.com/cuda/cuequivariance/tutorials/stp.html)
- [PyTorch API Reference](https://docs.nvidia.com/cuda/cuequivariance/api/cuequivariance_torch.html)
