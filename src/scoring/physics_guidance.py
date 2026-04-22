"""Vina-gradient physics guidance for the flow-matching ODE sampler.

At each ODE step, the guidance computes f_atom = -∇U_vina(atom_pos_t), then
aggregates per-atom forces into fragment (v_phys, omega_phys) via the same
Newton-Euler operator used during training. The caller mixes these into the
learned drift as classifier-free guidance:

    v_final = v_pred + λ(t) * v_phys
    omega_final = omega_pred + λ(t) * omega_phys
"""

from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem

from src.models.unified import newton_euler_aggregate
from src.scoring.vina import (
    compute_pocket_features_from_pdb,
    compute_vina_features,
    precompute_interaction_matrices,
    vina_scoring,
)


class PhysicsGuidance:
    """Caches Vina features and computes (v_phys, omega_phys) per ODE step.

    All coordinate inputs to ``compute_drift`` must be in the *pocket-centered*
    frame used inside ``sample_unified`` (i.e. with ``pocket_center`` subtracted).
    ``pocket_coords`` is shifted into the same frame at construction time.
    """

    def __init__(
        self,
        mol: Chem.Mol,
        pocket_pdb: str | Path,
        pocket_center: torch.Tensor,
        device: torch.device,
        pocket_cutoff: float = 8.0,
        weight_preset: str = "vina",
        max_force_per_atom: float = 10.0,
    ):
        self.device = device
        self.weight_preset = weight_preset
        self.max_force_per_atom = float(max_force_per_atom)

        self.lig_features = compute_vina_features(mol, device)

        # compute_pocket_features_from_pdb expects the UNSHIFTED center
        # (cutoff filter runs in the original PDB frame).
        pocket_center_cpu = pocket_center.detach().cpu().float()
        pocket_feat, pocket_coords_raw = compute_pocket_features_from_pdb(
            str(pocket_pdb),
            device=device,
            center=pocket_center_cpu,
            cutoff=pocket_cutoff,
        )

        # Shift pocket coords into the pocket-centered frame used by the sampler.
        self.pocket_coords = pocket_coords_raw - pocket_center.to(device)

        self.precomputed = precompute_interaction_matrices(
            self.lig_features, pocket_feat, device
        )

    @torch.enable_grad()
    def compute_drift(
        self,
        atom_pos_t: torch.Tensor,
        T_frag: torch.Tensor,
        frag_id: torch.Tensor,
        frag_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (v_phys, omega_phys), both [N_frag, 3], fully detached."""
        x = atom_pos_t.detach().clone().requires_grad_(True)

        energy = vina_scoring(
            x.unsqueeze(0),
            self.pocket_coords,
            self.precomputed,
            weight_preset=self.weight_preset,
            num_rotatable_bonds=None,
        )

        (grad,) = torch.autograd.grad(
            energy.sum(), x, create_graph=False, retain_graph=False,
        )
        f_atom = -grad.detach()

        norms = f_atom.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (self.max_force_per_atom / norms).clamp(max=1.0)
        f_atom = f_atom * scale

        n_frag = int(frag_sizes.shape[0])
        v_phys, omega_phys, _ = newton_euler_aggregate(
            f_atom=f_atom,
            atom_pos=atom_pos_t.detach(),
            T_frag=T_frag.detach(),
            frag_id=frag_id,
            n_frag=n_frag,
            frag_sizes=frag_sizes,
        )
        return v_phys.detach(), omega_phys.detach()
