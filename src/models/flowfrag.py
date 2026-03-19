"""FlowFrag: orchestrates ProteinEncoder, LigandEncoder, and DockingHead."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from .docking_head import DockingHead
from .ligand_encoder import LigandEncoder
from .protein_encoder import ProteinEncoder


class FlowFrag(nn.Module):
    """Fragment-based flow matching model for protein-ligand docking."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_encoder_layers_prot: int = 3,
        num_encoder_layers_lig: int = 4,
        num_docking_layers: int = 3,
        hidden_scalar_dim: int = 128,
        hidden_vec_dim: int = 32,
        pf_radius: float = 10.0,
        ff_radius: float = 6.0,
        pf_topology: str = "radius",
        ff_topology: str = "radius",
        use_cut_bond_edges: bool = False,
        omega_mode: str = "analytic",
        **kwargs,  # ignore unused V2 params from old configs
    ) -> None:
        super().__init__()

        self.protein_encoder = ProteinEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers_prot,
        )
        self.ligand_encoder = LigandEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers_lig,
        )
        self.docking_head = DockingHead(
            atom_dim=hidden_dim,
            prot_dim=hidden_dim,
            hidden_scalar_dim=hidden_scalar_dim,
            hidden_vec_dim=hidden_vec_dim,
            num_layers=num_docking_layers,
            pf_radius=pf_radius,
            ff_radius=ff_radius,
            pf_topology=pf_topology,
            ff_topology=ff_topology,
            use_cut_bond_edges=use_cut_bond_edges,
            omega_mode=omega_mode,
        )

    def forward(self, batch: HeteroData) -> dict[str, Tensor]:
        prot_batch: Tensor | None = getattr(batch["protein"], "batch", None)
        h_prot = self.protein_encoder(
            batch["protein"].pos,
            batch["protein"].x,
            batch=prot_batch,
        )

        bond_edge_index = batch["atom", "bond", "atom"].edge_index
        bond_edge_attr = batch["atom", "bond", "atom"].edge_attr
        h_atom = self.ligand_encoder(
            batch["atom"].x,
            batch["atom"].charge,
            batch["atom"].aromatic,
            batch["atom"].hybridization,
            batch["atom"].in_ring,
            bond_edge_index,
            bond_edge_attr,
        )

        head_out = self.docking_head(h_prot, h_atom, batch)

        if isinstance(head_out, tuple) and len(head_out) == 2 and isinstance(head_out[0], tuple):
            (v_pred, omega_pred), v_atom_pred = head_out
            return {"v_pred": v_pred, "omega_pred": omega_pred, "v_atom_pred": v_atom_pred}
        else:
            v_pred, omega_pred = head_out
            return {"v_pred": v_pred, "omega_pred": omega_pred}


__all__ = ["FlowFrag"]
