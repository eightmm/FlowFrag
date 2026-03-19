"""ODE sampler for FlowFrag: integrates fragment poses from prior to docked state."""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.geometry.flow_matching import integrate_se3_step, sample_prior_poses
from src.geometry.se3 import quaternion_to_matrix


class FlowFragSampler:
    """Euler ODE sampler for fragment-based flow matching.

    Args:
        model: Trained FlowFrag model (eval mode).
        num_steps: Number of ODE integration steps.
        translation_sigma: Prior translation noise scale (Angstroms).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_steps: int = 20,
        translation_sigma: float = 10.0,
        time_schedule: str = "uniform",
        schedule_power: float = 3.0,
    ) -> None:
        self.model = model
        self.num_steps = num_steps
        self.translation_sigma = translation_sigma
        self.time_schedule = time_schedule
        self.schedule_power = schedule_power

    @torch.no_grad()
    def sample(self, data: HeteroData, device: torch.device | None = None) -> dict[str, Tensor]:
        """Generate a docked pose from a single complex.

        Args:
            data: HeteroData with protein, atom, bond, and fragment metadata.
                  Must have ``fragment.size``, ``atom.local_pos``, ``atom.fragment_id``.
            device: Device to run on.

        Returns:
            Dict with:
            - ``T_pred``: predicted fragment translations ``[N_frag, 3]``
            - ``q_pred``: predicted fragment quaternions ``[N_frag, 4]``
            - ``atom_pos_pred``: predicted atom positions ``[N_atom, 3]``
        """
        if device is None:
            device = next(self.model.parameters()).device

        data = data.to(device)
        n_frags = data["fragment"].num_nodes
        frag_sizes = data["fragment"].size

        # Sample prior poses
        pocket_center = torch.zeros(3, device=device, dtype=torch.float32)
        T, q = sample_prior_poses(
            n_frags, pocket_center, self.translation_sigma,
            frag_sizes=frag_sizes, device=device, dtype=torch.float32,
        )

        time_grid = build_time_grid(
            self.num_steps,
            schedule=self.time_schedule,
            power=self.schedule_power,
            device=device,
            dtype=torch.float32,
        )

        for step_idx in range(self.num_steps):
            t = time_grid[step_idx]
            dt = time_grid[step_idx + 1] - time_grid[step_idx]

            # Update fragment poses in data for model forward
            data["fragment"].T_frag = T
            data["fragment"].q_frag = q
            data.t = t.view(1)

            # Compute atom positions at current state
            R = quaternion_to_matrix(q)
            frag_id = data["atom"].fragment_id
            local_pos = data["atom"].local_pos
            data["atom"].pos_t = (
                torch.einsum("nij,nj->ni", R[frag_id], local_pos) + T[frag_id]
            )

            # Model prediction
            out = self.model(data)
            v_pred = out["v_pred"]
            omega_pred = out["omega_pred"]

            # Integrate one step
            T, q = integrate_se3_step(
                T, q, v_pred, omega_pred, dt, frag_sizes=frag_sizes,
            )

        # Reconstruct atom positions from final poses
        R_final = quaternion_to_matrix(q)
        frag_id = data["atom"].fragment_id
        local_pos = data["atom"].local_pos
        atom_pos = (
            torch.einsum("nij,nj->ni", R_final[frag_id], local_pos) + T[frag_id]
        )

        return {"T_pred": T, "q_pred": q, "atom_pos_pred": atom_pos}

    @torch.no_grad()
    def sample_batch(
        self, batch: HeteroData, device: torch.device | None = None,
    ) -> list[dict[str, Tensor]]:
        """Generate docked poses for a batched HeteroData.

        Unbatches, samples each complex independently, returns list of results.
        """
        if device is None:
            device = next(self.model.parameters()).device

        batch = batch.to(device)

        # Unbatch by fragment batch vector
        frag_batch = batch["fragment"].batch
        atom_batch = batch["atom"].batch
        prot_batch = batch["protein"].batch
        n_graphs = frag_batch.max().item() + 1

        results = []
        for i in range(n_graphs):
            data_i = _extract_single(batch, i, frag_batch, atom_batch, prot_batch)
            results.append(self.sample(data_i, device=device))

        return results


def _extract_single(
    batch: HeteroData,
    idx: int,
    frag_batch: Tensor,
    atom_batch: Tensor,
    prot_batch: Tensor,
) -> HeteroData:
    """Extract a single complex from a batched HeteroData."""
    data = HeteroData()

    # Protein
    prot_mask = prot_batch == idx
    data["protein"].pos = batch["protein"].pos[prot_mask]
    data["protein"].x = batch["protein"].x[prot_mask]
    data["protein"].num_nodes = prot_mask.sum().item()

    # Atoms
    atom_mask = atom_batch == idx
    data["atom"].x = batch["atom"].x[atom_mask]
    data["atom"].charge = batch["atom"].charge[atom_mask]
    data["atom"].aromatic = batch["atom"].aromatic[atom_mask]
    data["atom"].hybridization = batch["atom"].hybridization[atom_mask]
    data["atom"].in_ring = batch["atom"].in_ring[atom_mask]
    data["atom"].local_pos = batch["atom"].local_pos[atom_mask]
    data["atom"].num_nodes = atom_mask.sum().item()

    # Remap fragment_id to local indices
    frag_mask = frag_batch == idx
    frag_offset = (frag_batch < idx).sum().item()
    atom_frag_id = batch["atom"].fragment_id[atom_mask]
    data["atom"].fragment_id = atom_frag_id - frag_offset

    # Bonds — remap edge indices
    bond_ei = batch["atom", "bond", "atom"].edge_index
    bond_attr = batch["atom", "bond", "atom"].edge_attr
    atom_offset = (atom_batch < idx).sum().item()
    atom_end = atom_offset + atom_mask.sum().item()
    bond_mask = (bond_ei[0] >= atom_offset) & (bond_ei[0] < atom_end)
    data["atom", "bond", "atom"].edge_index = bond_ei[:, bond_mask] - atom_offset
    data["atom", "bond", "atom"].edge_attr = bond_attr[bond_mask]

    # Fragments
    data["fragment"].num_nodes = frag_mask.sum().item()
    data["fragment"].size = batch["fragment"].size[frag_mask]
    if hasattr(batch["fragment"], "T_target"):
        data["fragment"].T_target = batch["fragment"].T_target[frag_mask]

    return data


def build_time_grid(
    num_steps: int,
    *,
    schedule: str = "uniform",
    power: float = 3.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Construct a monotone time grid on ``[0, 1]`` for ODE integration.

    Args:
        num_steps: Number of integration intervals.
        schedule: One of ``uniform``, ``late``, or ``early``.
            ``late`` concentrates more steps near ``t=1`` (crystal end),
            ``early`` concentrates more steps near ``t=0`` (prior end).
        power: Exponent controlling how strongly steps are concentrated.
            Must be positive. ``1.0`` reduces to the uniform grid.
        device: Optional output device.
        dtype: Optional floating output dtype.

    Returns:
        Tensor of shape ``[num_steps + 1]`` with endpoints ``0`` and ``1``.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    if power <= 0:
        raise ValueError(f"power must be positive, got {power}.")

    if dtype is None:
        dtype = torch.float32

    u = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    if schedule == "uniform" or power == 1.0:
        return u
    if schedule == "late":
        return 1.0 - (1.0 - u) ** power
    if schedule == "early":
        return u ** power
    raise ValueError(f"Unknown time schedule '{schedule}'.")


__all__ = ["FlowFragSampler", "build_time_grid"]
