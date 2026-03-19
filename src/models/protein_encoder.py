"""Protein encoder: residue-level invariant GNN over CA coordinates."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from .layers import GatedGCNLayer, build_radius_graph, rbf_encode


class ProteinEncoder(nn.Module):
    """Invariant residue-level encoder over CA coordinates.

    Args:
        hidden_dim: Output feature dimension.
        num_layers: Number of GatedGCN message-passing layers.
        radius: Radius graph cutoff in Angstroms.
        max_neighbors: Maximum neighbors per node.
        n_rbf: Number of RBF basis functions for edge features.
        res_type_vocab: Vocabulary size for residue type embeddings.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        radius: float = 8.0,
        max_neighbors: int = 32,
        n_rbf: int = 16,
        res_type_vocab: int = 21,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.n_rbf = n_rbf

        # Node embedding: residue type → hidden_dim
        self.res_type_emb = nn.Embedding(res_type_vocab, 64)
        self.node_proj = nn.Linear(64, hidden_dim)

        # Edge embedding: RBF distances → hidden_dim
        self.edge_proj = nn.Linear(n_rbf, hidden_dim)

        # GatedGCN layers
        self.layers = nn.ModuleList(
            [GatedGCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        pos: Tensor,
        res_type: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Encode protein residues.

        Args:
            pos: CA coordinates of shape ``[N_res, 3]``.
            res_type: Residue type indices of shape ``[N_res]``.
            batch: Optional batch vector of shape ``[N_res]`` for batched graphs.

        Returns:
            Tensor of shape ``[N_res, hidden_dim]``.
        """
        # Node features
        h = self.node_proj(self.res_type_emb(res_type))

        # Build radius graph
        edge_index = build_radius_graph(
            pos,
            r=self.radius,
            batch=batch,
            max_num_neighbors=self.max_neighbors,
        )

        # Edge features: RBF over distances
        src, dst = edge_index
        diff = pos[dst] - pos[src]
        dist = torch.linalg.vector_norm(diff, dim=-1)
        e = self.edge_proj(rbf_encode(dist, n_rbf=self.n_rbf))

        # Message passing
        for layer in self.layers:
            h, e = layer(h, edge_index, e)

        return h


__all__ = ["ProteinEncoder"]
