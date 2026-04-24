"""ODE sampler utilities for the unified fragment-flow pipeline.

Primary entry point: :func:`sample_unified` runs a batched ODE integration
from the SE(3) prior to t=1 for one complex at a time; :func:`build_batched_graph`
replicates a single-complex graph across samples for parallel sampling.

Additional helper :func:`build_time_grid` produces non-uniform t-grids
(``late`` schedule concentrates steps near t=1 to match the main model's
late-biased training distribution).
"""
from __future__ import annotations

import torch
from torch import Tensor

from src.geometry.flow_matching import integrate_se3_step, sample_prior_poses
from src.geometry.se3 import quaternion_to_matrix


# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------
def build_time_grid(
    num_steps: int,
    *,
    schedule: str = "uniform",
    power: float = 3.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Construct a monotone time grid on ``[0, 1]`` for ODE integration."""
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


# ---------------------------------------------------------------------------
# Batched graph replication
# ---------------------------------------------------------------------------
_SKIP_REPLICATE_KEYS = frozenset((
    "edge_index", "node_fragment_id",              # offsetted below
    "num_nodes", "num_prot_atom", "num_prot_res",
    "lig_frag_slice", "lig_atom_slice",             # metadata
))


def build_batched_graph(
    graph: dict[str, Tensor],
    B: int,
    n_frags_per: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """Replicate a single-complex graph ``B`` times with per-sample offsets.

    Auto-detects which tensors to replicate from their leading dim (n_nodes or
    n_edges). Index tensors referencing node ids (``edge_index``) or fragment
    ids (``node_fragment_id``) get per-sample offsets so sample blocks do not
    cross-talk.
    """
    n_nodes = graph["node_coords"].shape[0]
    n_edges = graph["edge_index"].shape[1]
    out: dict[str, Tensor] = {}

    for k, v in graph.items():
        if k in _SKIP_REPLICATE_KEYS:
            out[k] = v
            continue
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        t = v.to(device)
        if t.ndim >= 1 and (t.shape[0] == n_nodes or t.shape[0] == n_edges):
            out[k] = t.repeat(B, *([1] * (t.ndim - 1))) if t.ndim > 1 else t.repeat(B)
        else:
            out[k] = t

    ei = graph["edge_index"].to(device)
    offsets = torch.arange(B, device=device, dtype=ei.dtype).repeat_interleave(n_edges) * n_nodes
    out["edge_index"] = ei.repeat(1, B) + offsets.unsqueeze(0)

    if "node_fragment_id" in graph:
        nf = graph["node_fragment_id"].to(device).repeat(B).clone()
        for i in range(B):
            sl_lo, sl_hi = i * n_nodes, (i + 1) * n_nodes
            seg = nf[sl_lo:sl_hi]
            pos = seg >= 0
            seg[pos] = seg[pos] + i * n_frags_per
            nf[sl_lo:sl_hi] = seg
        out["node_fragment_id"] = nf

    out["batch"] = torch.arange(B, device=device).repeat_interleave(n_nodes)
    out["frag_batch"] = torch.arange(B, device=device).repeat_interleave(n_frags_per)
    return out


# ---------------------------------------------------------------------------
# Unified ODE sampler (batched across N samples of one complex)
# ---------------------------------------------------------------------------
def sample_unified(
    model: torch.nn.Module,
    graph: dict[str, Tensor],
    lig_data: dict,
    meta: dict,
    num_samples: int = 1,
    *,
    num_steps: int = 25,
    translation_sigma: float = 5.0,
    time_schedule: str = "late",
    schedule_power: float = 3.0,
    device: torch.device = torch.device("cpu"),
    save_traj: bool = False,
    phys_guidance=None,
    phys_lambda_max: float = 0.0,
    phys_power: float = 2.0,
    phys_start_t: float = 0.3,
    stochastic_gamma: float = 0.0,
) -> list[dict[str, Tensor]]:
    """Run batched ODE integration for ``num_samples`` poses of one complex.

    All samples share the same protein/ligand graph but have independent SE(3)
    priors; the graph is replicated ``num_samples`` times and fed through the
    model once per ODE step instead of once per sample, for a ~4x speedup.

    ``stochastic_gamma > 0`` turns the deterministic ODE into an annealed SDE by
    adding per-step Gaussian noise of scale ``γ · √(dt · (1 - t))`` to both
    translation and angular velocity. The ``√(1 - t)`` factor guarantees the
    perturbation vanishes as t → 1 so the trajectory still converges to the
    target manifold; mid-trajectory it broadens the sample distribution and
    helps escape local modes of the learnt drift.

    Returns a list of length ``num_samples``. Each entry has
    ``atom_pos_pred: [N_atoms, 3]`` (pocket-centered frame) and, if
    ``save_traj=True``, ``traj: list[Tensor]`` + ``traj_times: list[float]``
    covering the ``num_steps + 1`` recorded frames.
    """
    B = int(num_samples)
    n_frags = int(meta["num_frag"])
    pocket_center = meta["pocket_center"]
    frag_sizes = lig_data["frag_sizes"]
    frag_id = lig_data["fragment_id"]
    local_pos = lig_data["frag_local_coords"]
    n_real_atoms = local_pos.shape[0]

    batch = build_batched_graph(graph, B, n_frags, device)
    batch["node_coords"] = batch["node_coords"] - pocket_center.to(device)

    frag_sizes_flat = frag_sizes.to(device).repeat(B)
    local_pos_d = local_pos.to(device)
    frag_id_d = frag_id.to(device)
    frag_id_flat = (
        frag_id_d.repeat(B)
        + torch.arange(B, device=device).repeat_interleave(n_real_atoms) * n_frags
    )

    # Prior on CPU so torch.manual_seed reproduces sequential-era RNG streams.
    T_flat, q_flat = sample_prior_poses(
        B * n_frags, pocket_center=torch.zeros(3),
        translation_sigma=translation_sigma, frag_sizes=frag_sizes_flat.cpu(),
    )
    T_flat = T_flat.to(device)
    q_flat = q_flat.to(device)

    time_grid = build_time_grid(
        num_steps, schedule=time_schedule, power=schedule_power,
        device=device, dtype=torch.float32,
    )

    frag_start, frag_end = graph["lig_frag_slice"][0].item(), graph["lig_frag_slice"][1].item()
    atom_start = graph["lig_atom_slice"][0].item()
    n_nodes = graph["node_coords"].shape[0]

    frag_slots = torch.cat([
        torch.arange(frag_start, frag_end, device=device) + i * n_nodes for i in range(B)
    ])
    atom_slots = torch.cat([
        torch.arange(atom_start, atom_start + n_real_atoms, device=device) + i * n_nodes for i in range(B)
    ])

    traj_frames: list[Tensor] = []
    traj_times: list[float] = []

    for step_idx in range(num_steps):
        t = time_grid[step_idx]
        dt = time_grid[step_idx + 1] - time_grid[step_idx]

        R_flat = quaternion_to_matrix(q_flat)
        atom_pos_flat = (
            torch.einsum("nij,nj->ni", R_flat[frag_id_flat], local_pos_d.repeat(B, 1))
            + T_flat[frag_id_flat]
        )

        nc = batch["node_coords"].clone()
        nc[frag_slots] = T_flat
        nc[atom_slots] = atom_pos_flat
        batch["node_coords"] = nc

        if save_traj:
            traj_frames.append(atom_pos_flat.view(B, n_real_atoms, 3).cpu())
            traj_times.append(t.item())

        batch["T_frag"] = T_flat
        batch["q_frag"] = q_flat
        batch["frag_sizes"] = frag_sizes_flat
        batch["t"] = t.view(1, 1).expand(B, 1).contiguous()
        batch["frag_id_for_atoms"] = frag_id_flat

        with torch.no_grad():
            out = model(batch)

        v_use = out["v_pred"]
        omega_use = out["omega_pred"]

        if (
            phys_guidance is not None
            and phys_lambda_max > 0.0
            and t.item() >= phys_start_t
        ):
            lam = phys_lambda_max * (t.item() ** phys_power)
            v_phys, omega_phys = phys_guidance.compute_drift_batched(
                atom_pos_t_flat=atom_pos_flat,
                T_frag_flat=T_flat,
                frag_id_flat=frag_id_flat,
                frag_sizes_flat=frag_sizes_flat,
                batch_size=B,
                n_atoms_per_sample=n_real_atoms,
            )
            v_use = v_use + lam * v_phys
            omega_use = omega_use + lam * omega_phys

        if stochastic_gamma > 0.0 and t.item() < 1.0:
            # Annealed Langevin correction: perturb velocity by γ·√((1-t)/dt)·N(0,I).
            # The 1/√dt normalization makes the noise kick size γ·√((1-t)·dt) once
            # integrated over ``dt`` via the Euler step below — matching the standard
            # Euler-Maruyama discretization of dX = v dt + σ(t) dW.
            sigma_t = stochastic_gamma * ((1.0 - t.item()) / max(dt.item(), 1e-6)) ** 0.5
            v_use = v_use + sigma_t * torch.randn_like(v_use)
            omega_use = omega_use + sigma_t * torch.randn_like(omega_use)

        T_flat, q_flat = integrate_se3_step(
            T_flat, q_flat, v_use, omega_use, dt, frag_sizes=frag_sizes_flat,
        )

    R_final = quaternion_to_matrix(q_flat)
    atom_pos_pred_flat = (
        torch.einsum("nij,nj->ni", R_final[frag_id_flat], local_pos_d.repeat(B, 1))
        + T_flat[frag_id_flat]
    )
    final_per_sample = atom_pos_pred_flat.view(B, n_real_atoms, 3).cpu()

    if save_traj:
        traj_frames.append(final_per_sample)
        traj_times.append(1.0)

    results: list[dict[str, Tensor]] = []
    for i in range(B):
        res: dict[str, Tensor] = {"atom_pos_pred": final_per_sample[i]}
        if save_traj:
            res["traj"] = [frame[i] for frame in traj_frames]
            res["traj_times"] = list(traj_times)
        results.append(res)
    return results


__all__ = ["build_time_grid", "sample_unified", "build_batched_graph"]
