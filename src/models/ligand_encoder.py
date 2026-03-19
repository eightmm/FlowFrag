"""Ligand encoder: invariant 2D molecular graph GNN."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .layers import GatedGCNLayer


class LigandEncoder(nn.Module):
    """Invariant encoder over the 2D molecular graph.

    Args:
        hidden_dim: Output feature dimension.
        num_layers: Number of GatedGCN message-passing layers.
    """

    NUM_ELEMENTS = 13
    NUM_HYBRIDIZATIONS = 6
    NUM_BOND_TYPES = 5

    def __init__(self, hidden_dim: int = 128, num_layers: int = 4) -> None:
        super().__init__()

        # Atom feature embeddings
        self.elem_emb = nn.Embedding(self.NUM_ELEMENTS, 32)
        self.charge_proj = nn.Linear(1, 8)
        self.aromatic_emb = nn.Embedding(2, 8)
        self.hybrid_emb = nn.Embedding(self.NUM_HYBRIDIZATIONS, 16)
        self.ring_emb = nn.Embedding(2, 8)
        # 32 + 8 + 8 + 16 + 8 = 72
        self.atom_proj = nn.Linear(72, hidden_dim)

        # Bond feature embeddings
        self.bond_type_emb = nn.Embedding(self.NUM_BOND_TYPES, 16)
        self.bond_conj_emb = nn.Embedding(2, 8)
        self.bond_ring_emb = nn.Embedding(2, 8)
        # 16 + 8 + 8 = 32
        self.bond_proj = nn.Linear(32, hidden_dim)

        # GatedGCN layers
        self.layers = nn.ModuleList(
            [GatedGCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        atom_element: Tensor,
        atom_charge: Tensor,
        atom_aromatic: Tensor,
        atom_hybridization: Tensor,
        atom_in_ring: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """Encode ligand atoms from molecular graph.

        Args:
            atom_element: Element indices ``[N_atom]`` int64.
            atom_charge: Formal charges ``[N_atom]`` (any int type).
            atom_aromatic: Aromatic flags ``[N_atom]`` bool.
            atom_hybridization: Hybridization indices ``[N_atom]`` int.
            atom_in_ring: Ring flags ``[N_atom]`` bool.
            edge_index: Bond connectivity ``[2, E]`` int64.
            edge_attr: Bond features ``[E, 3]`` float32
                (bond_type, bond_conjugated, bond_in_ring).

        Returns:
            Tensor of shape ``[N_atom, hidden_dim]``.
        """
        # Build node features
        elem = self.elem_emb(atom_element.long())  # [N, 32]
        charge = self.charge_proj(atom_charge.float().unsqueeze(-1))  # [N, 8]
        arom = self.aromatic_emb(atom_aromatic.long())  # [N, 8]
        hybrid = self.hybrid_emb(atom_hybridization.long())  # [N, 16]
        ring = self.ring_emb(atom_in_ring.long())  # [N, 8]
        h = self.atom_proj(torch.cat([elem, charge, arom, hybrid, ring], dim=-1))

        # Build edge features from bond attributes [E, 3]
        bond_type = edge_attr[:, 0].long()
        bond_conj = edge_attr[:, 1].long()
        bond_ring = edge_attr[:, 2].long()
        e = self.bond_proj(
            torch.cat(
                [
                    self.bond_type_emb(bond_type),
                    self.bond_conj_emb(bond_conj),
                    self.bond_ring_emb(bond_ring),
                ],
                dim=-1,
            )
        )

        # Message passing
        for layer in self.layers:
            h, e = layer(h, edge_index, e)

        return h


__all__ = ["LigandEncoder"]
