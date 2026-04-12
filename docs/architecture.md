# FlowFrag Model Architecture

## Overview

FlowFrag uses a unified SE(3)-equivariant GNN over a heterogeneous graph
containing protein and ligand nodes. The model predicts per-fragment
translation velocity `v` and angular velocity `ω` for flow matching-based
docking.

## Architecture Diagram

```mermaid
graph TD
    subgraph Input["Input"]
        G["Unified Graph\n574 nodes × 5024 edges"]
        T["Time step t ∈ [0,1]"]
        Q["Fragment rotation R_t"]
    end

    subgraph Embedding["Node Embedding (scalar)"]
        NE["UnifiedNodeEmbedding\nelement + charge + aromatic + hybrid + ring\n+ node_type + amino_acid + pharmacophore\n→ MLP(120 → 256)"]
        TE["Time Embedding\nsinusoidal(t, 32) → MLP(32→128→128)"]
        FE["Fragment Init\nh_scalar ⊕ size_emb(16) ⊕ t_emb\n→ MLP(400→256→256)"]
    end

    subgraph VecInit["Equivariant State Init"]
        V1O["1o init: gate(h) · displacement r\nvec_gate: Linear(256→32) + Tanh"]
        VRT["R_t injection: gate(h) · R_cols\nfrag_rot_gate: Linear(256→96) + Tanh\n→ fragment 1o channels"]
        V1E["1e init: zeros"]
        V2E["2e init: zeros"]
        CAT["Concat → h\n256×0e + 32×1o + 32×1e + 16×2e\nD = 528"]
    end

    subgraph Layers["Interaction Layers ×4"]
        subgraph Layer["UnifiedInteractionLayer"]
            ES["Edge Scalars\nRBF(16) + edge_type(16) + bond_feats(20)\n+ ref_dist(8) + src_h(256) + dst_h(256) + t(128)\n= 700-dim"]
            TP["SE(3) Tensor Product Conv\ncuEquivariance FCTP\nSH l=0,1,2"]
            UP["Update: Linear → SiLU(0e) → Dropout"]
            RES["Residual: h = h + update"]
            ALN["AdaLN(h, t_emb)\ntime-conditioned normalization"]
        end
        REINJ["Per-layer R_t re-injection\ninto fragment 1o channels"]
    end

    subgraph DynEdge["Dynamic Edges (optional)"]
        DC["Contact edges: cdist ≤ cutoff\n→ edge_type = 9"]
    end

    subgraph Output["Output Heads"]
        subgraph Direct["direct mode"]
            HP["h_frag → Linear → SiLU → Linear"]
            VP["v_pred (1o)"]
            WP["ω_pred (1e)"]
        end
        subgraph NE_mode["newton_euler mode"]
            FA["h_lig_atom → Linear → SiLU → Linear\n→ f_atom (1o)"]
            NE2["Newton-Euler Aggregation\nv = mean(f) per frag\nτ = Σ r×f, I·ω = τ"]
        end
        MASK["Mask ω=0 for single-atom frags"]
    end

    G --> NE
    T --> TE
    NE --> FE
    TE --> FE
    FE --> V1O
    FE --> VRT
    V1O --> CAT
    VRT --> CAT
    V1E --> CAT
    V2E --> CAT

    CAT --> Layer
    ES --> TP
    TP --> UP
    UP --> RES
    RES --> ALN
    ALN --> REINJ
    REINJ -->|"×4 loop"| Layer

    G --> DynEdge
    DynEdge --> ES

    ALN -->|"final h"| HP
    ALN -->|"final h"| FA
    HP --> VP
    HP --> WP
    FA --> NE2
    NE2 --> VP
    NE2 --> WP
    VP --> MASK
    WP --> MASK
```

## Node Types

| ID | Type | Description | Count (example) |
|----|------|-------------|-----------------|
| 0 | `ligand_atom` | Ligand heavy atoms | 27 |
| 1 | `ligand_dummy` | Dummy atoms at cut bonds (optional) | 0 |
| 2 | `ligand_fragment` | Fragment center nodes | 6 |
| 3 | `protein_atom` | Pocket heavy atoms (8Å cutoff) | 485 |
| 4 | `protein_ca` | Cα virtual nodes per residue | 56 |

## Edge Types

| ID | Type | Description | Count (example) |
|----|------|-------------|-----------------|
| 0 | `ligand_bond` | Covalent bonds in ligand | 29 |
| 1 | `ligand_tri` | Triangulation (cross-fragment distance constraints) | 13 |
| 2 | `ligand_cut` | Cut bonds (inter-fragment) | 5 |
| 3 | `ligand_atom_frag` | Atom ↔ parent fragment | 27 |
| 4 | `ligand_frag_frag` | Adjacent fragment pairs | 15 |
| 5 | `protein_bond` | Protein covalent bonds | 485 |
| 6 | `protein_atom_ca` | Protein atom ↔ parent Cα | 485 |
| 7 | `protein_ca_ca` | Cα ↔ Cα (distance-based) | 1117 |
| 8 | `protein_ca_frag` | Cα ↔ nearby fragment | 336 |
| 9 | `dynamic_contact` | Runtime protein-ligand contacts (optional) | varies |

## Irreps Layout

All nodes share the same irreps space:

```
h = [256×0e] + [32×1o] + [32×1e] + [16×2e]
     scalar     vector    pseudo-v   quadrupole
     (256)      (96)      (96)       (80)
                                     = 528 total
```

- **0e (scalars)**: Chemical features, embeddings
- **1o (odd vectors)**: Displacement-gated directions, forces
- **1e (even pseudo-vectors)**: Angular velocity output channel
- **2e (quadrupoles)**: Higher-order geometric features

## Key Design Choices

1. **Single TP conv for all edge types**: Edge-type specialization via
   edge scalar features (type embedding + bond features), not separate
   convolution layers per type.

2. **R_t injection per layer**: Fragment rotation matrix columns are
   gated and added to fragment 1o channels at every layer, not just
   initialization. This provides continuous rotation state awareness.

3. **Newton-Euler mode**: Optional atom-level force prediction with
   physical aggregation (torque → inertia solve → ω). Alternative to
   direct fragment-level ω prediction.

4. **AdaLN time conditioning**: Adaptive layer normalization modulates
   node features based on the flow matching time step, allowing the
   model to behave differently at early (coarse) vs late (fine) stages.

5. **cuEquivariance acceleration**: All tensor products use NVIDIA's
   cuEquivariance CUDA kernels with `mul_ir` layout for performance.

## Implementation

- Model: `src/models/unified.py::UnifiedFlowFrag`
- Equivariant layers: `src/models/equivariant.py`
- Utility layers: `src/models/layers.py`
- Graph construction: `src/preprocess/graph.py`
