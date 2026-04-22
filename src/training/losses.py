"""Flow matching losses for fragment-based docking."""

from __future__ import annotations

import torch
from torch import Tensor


def _omega_mse(pred: Tensor, target: Tensor) -> Tensor:
    """MSE between omega vectors (standard)."""
    return ((pred - target) ** 2).mean()


def _omega_direction_magnitude(
    pred: Tensor,
    target: Tensor,
    dir_weight: float = 1.0,
    mag_weight: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Split omega loss into direction (1 - cos_sim) and magnitude MSE.

    Returns (total, dir_loss, mag_loss).
    """
    cos = torch.nn.functional.cosine_similarity
    loss_dir = (1.0 - cos(pred, target, dim=-1)).mean()
    loss_mag = ((pred.norm(dim=-1) - target.norm(dim=-1)) ** 2).mean()
    total = dir_weight * loss_dir + mag_weight * loss_mag
    return total, loss_dir.detach(), loss_mag.detach()


def _omega_cosine_weighted(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """MSE on omega, downweighted by target magnitude squared."""
    weights = 1.0 / (target.norm(dim=-1, keepdim=True) ** 2 + eps)
    return (weights * (pred - target) ** 2).mean()


def compute_time_weight(t: Tensor) -> Tensor:
    """Early-t emphasis: w(t) = (2 - t) / 1.5, mean=1 over U[0,1].

    Rationale: at small t the ligand is far from the pocket with fewer
    contact edges (harder task), so we put more gradient weight there.
    Linear, bounded in [0.67, 1.33] so it never dominates the loss scale.

    Args:
        t: Shape ``[N]`` per-fragment time values.
    Returns:
        Weight tensor ``[N]``.
    """
    return (2.0 - t) / 1.5


def flow_matching_loss(
    v_pred: Tensor,
    omega_pred: Tensor,
    v_target: Tensor,
    omega_target: Tensor,
    frag_sizes: Tensor,
    omega_weight: float = 1.0,
    *,
    R_t: Tensor | None = None,
    omega_loss_frame: str = "world",
    omega_loss_type: str = "mse",
    omega_dir_weight: float = 1.0,
    omega_mag_weight: float = 0.1,
    time_weight: Tensor | None = None,
    P_observable: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute flow matching loss: MSE on translation and angular velocities.

    Args:
        v_pred: Predicted translation velocity ``[N_frag, 3]``.
        omega_pred: Predicted angular velocity ``[N_frag, 3]``.
        v_target: Ground-truth translation velocity ``[N_frag, 3]``.
        omega_target: Ground-truth angular velocity ``[N_frag, 3]``.
        frag_sizes: Fragment sizes ``[N_frag]``.
        omega_weight: Weight for angular velocity loss relative to translation.
        R_t: Rotation matrices ``[N_frag, 3, 3]`` (required for body frame).
        omega_loss_frame: ``"world"`` (default) or ``"body"``.
        omega_loss_type: ``"mse"`` / ``"direction_magnitude"`` / ``"cosine_weighted"``.
        omega_dir_weight / omega_mag_weight: for ``direction_magnitude``.
        time_weight: Per-fragment time weight ``[N_frag]`` for early-t emphasis.
            Compute via ``compute_time_weight(t_per_frag)``. None = uniform.

    Returns:
        Dict with ``"loss"``, ``"loss_v"``, ``"loss_omega"``, ``"cos_v"``, ``"cos_omega"``.
    """
    # Per-fragment translation squared error then time-weighted mean
    v_sq = ((v_pred - v_target) ** 2).sum(-1)  # [N_frag]
    if time_weight is not None:
        loss_v = (time_weight * v_sq).mean() / 3.0  # /3 to match old MSE scale
    else:
        loss_v = v_sq.mean() / 3.0

    multi_mask = frag_sizes > 1
    zero = torch.zeros(1, device=v_pred.device, dtype=v_pred.dtype).squeeze()

    extras: dict[str, Tensor] = {}

    if multi_mask.any():
        op = omega_pred[multi_mask]
        ot = omega_target[multi_mask]

        # Project omega_target into observable subspace (C2: rank-deficient handling).
        # P_observable zeroes axes where the inertia tensor is near-singular
        # (e.g., 2-atom fragments have one unobservable rotation axis).
        if P_observable is not None:
            P = P_observable[multi_mask]  # [M, 3, 3]
            ot = torch.einsum("nij,nj->ni", P, ot)

        # Body-frame canonicalization
        if omega_loss_frame == "body":
            assert R_t is not None, "R_t required for body-frame omega loss"
            Rt_inv = R_t[multi_mask].transpose(-1, -2)  # [M, 3, 3]
            op = torch.einsum("nij,nj->ni", Rt_inv, op)
            ot = torch.einsum("nij,nj->ni", Rt_inv, ot)

        # Per-fragment weight slice for omega path
        tw_omega = time_weight[multi_mask] if time_weight is not None else None

        if omega_loss_type == "mse":
            w_sq = ((op - ot) ** 2).sum(-1)  # [M]
            if tw_omega is not None:
                loss_omega = (tw_omega * w_sq).mean() / 3.0
            else:
                loss_omega = w_sq.mean() / 3.0
        elif omega_loss_type == "direction_magnitude":
            loss_omega, ld, lm = _omega_direction_magnitude(
                op, ot, omega_dir_weight, omega_mag_weight,
            )
            extras["loss_omega_dir"] = ld
            extras["loss_omega_mag"] = lm
        elif omega_loss_type == "cosine_weighted":
            loss_omega = _omega_cosine_weighted(op, ot)
        else:
            raise ValueError(f"Unknown omega_loss_type '{omega_loss_type}'")

        cos_omega = torch.nn.functional.cosine_similarity(op, ot, dim=-1).mean()
    else:
        loss_omega = zero
        cos_omega = zero

    loss = loss_v + omega_weight * loss_omega

    cos = torch.nn.functional.cosine_similarity
    cos_v = cos(v_pred, v_target, dim=-1).mean()

    # World-frame cos_omega for consistent monitoring across all modes
    if multi_mask.any() and omega_loss_frame == "body":
        cos_omega_world = cos(
            omega_pred[multi_mask], omega_target[multi_mask], dim=-1,
        ).mean()
        extras["cos_omega_world"] = cos_omega_world.detach()

    result = {
        "loss": loss,
        "loss_v": loss_v.detach(),
        "loss_omega": loss_omega.detach(),
        "cos_v": cos_v.detach(),
        "cos_omega": cos_omega.detach(),
        **{k: v for k, v in extras.items()},
    }
    return result


def atom_velocity_loss(
    v_atom_pred: Tensor,
    v_atom_target: Tensor,
    v_frag_pred: Tensor,
    omega_pred: Tensor,
    v_frag_target: Tensor,
    omega_target: Tensor,
    frag_sizes: Tensor,
) -> dict[str, Tensor]:
    """Loss for atom_velocity mode: MSE on per-atom velocities.

    Also computes fragment-level v/omega metrics for monitoring.
    """
    # Primary loss: per-atom velocity MSE
    loss_atom = torch.mean((v_atom_pred - v_atom_target) ** 2)

    # Fragment-level metrics for monitoring (not in loss)
    loss_v = torch.mean((v_frag_pred - v_frag_target) ** 2).detach()

    multi_mask = frag_sizes > 1
    if multi_mask.any():
        loss_omega = torch.mean((omega_pred[multi_mask] - omega_target[multi_mask]) ** 2).detach()
    else:
        loss_omega = torch.zeros(1, device=v_atom_pred.device, dtype=v_atom_pred.dtype).squeeze()

    cos = torch.nn.functional.cosine_similarity
    cos_v = cos(v_frag_pred, v_frag_target, dim=-1).mean().detach()
    if multi_mask.any():
        cos_omega = cos(omega_pred[multi_mask], omega_target[multi_mask], dim=-1).mean().detach()
    else:
        cos_omega = torch.zeros(1, device=v_atom_pred.device, dtype=v_atom_pred.dtype).squeeze()

    return {
        "loss": loss_atom,
        "loss_v": loss_v,
        "loss_omega": loss_omega,
        "cos_v": cos_v,
        "cos_omega": cos_omega,
        "loss_atom": loss_atom.detach(),
    }


def distance_geometry_loss(
    v_pred: Tensor,
    omega_pred: Tensor,
    T_t: Tensor,
    q_t: Tensor,
    t_per_sample: Tensor,
    frag_batch: Tensor,
    T_target: Tensor,
    q_target: Tensor,
    local_pos: Tensor,
    frag_id_for_atoms: Tensor,
    atom_batch: Tensor,
    lig_atom_slice: Tensor,
    lig_frag_slice: Tensor,
) -> dict[str, Tensor]:
    """One-step Euler integration to t=1, then pairwise-distance MSE vs crystal.

    Reconstructs final molecule coordinates by applying the predicted
    ``(v, omega)`` as a single rigid-body step from current ``(T_t, q_t)``
    to ``t=1``.  The pairwise atom-atom distance matrix of the predicted
    pose is compared to the target pose's distance matrix (from
    ``T_target, q_target``).

    Within-fragment distances are rigid-body invariant (loss contribution = 0
    if the fragment is a single rigid body), so this loss effectively
    supervises *inter-fragment* spacing, which encodes bond lengths across
    cut bonds and global molecular shape validity.

    All tensors assume the unified_collate layout:
        v_pred, omega_pred, T_t, q_t, T_target, q_target: [N_frag_total, *]
        local_pos: [N_atom_total, 3]
        frag_id_for_atoms: [N_atom_total] (global frag indices after collate)
        atom_batch: [N_atom_total] sample index per atom
        frag_batch: [N_frag_total] sample index per fragment
        t_per_sample: [B, 1] or [B]

    Returns dict with ``"loss_dg"``.
    """
    from ..geometry.se3 import (
        axis_angle_to_quaternion,
        quaternion_multiply,
        quaternion_to_matrix,
    )

    # Per-fragment remaining time
    t_flat = t_per_sample.view(-1)[frag_batch]  # [N_frag_total]
    dt = (1.0 - t_flat).unsqueeze(-1)  # [N_frag_total, 1]

    # One-step Euler: T_1 = T_t + dt * v_pred
    T_1_pred = T_t + dt * v_pred

    # Rotation: q_1 = exp(dt * omega_pred) ∘ q_t
    dq_pred = axis_angle_to_quaternion(dt * omega_pred)
    q_1_pred = quaternion_multiply(dq_pred, q_t)
    R_1_pred = quaternion_to_matrix(q_1_pred)

    # Reconstruct predicted atom positions at t=1
    atom_pos_pred = (
        torch.einsum("nij,nj->ni", R_1_pred[frag_id_for_atoms], local_pos)
        + T_1_pred[frag_id_for_atoms]
    )

    # Target atom positions at t=1
    R_1_target = quaternion_to_matrix(q_target)
    atom_pos_target = (
        torch.einsum("nij,nj->ni", R_1_target[frag_id_for_atoms], local_pos)
        + T_target[frag_id_for_atoms]
    )

    # Per-sample pairwise distance MSE with time-dependent weight.
    # w(t) = t² — stronger near t=1 where one-step Euler should be exact,
    # weak near t=0 where a single step is a crude approximation of the trajectory.
    t_sample_flat = t_per_sample.view(-1)
    total_weighted = 0.0
    total_weight = 0.0
    B = int(atom_batch.max().item()) + 1 if atom_batch.numel() > 0 else 0
    for s in range(B):
        mask = atom_batch == s
        if mask.sum() < 2:
            continue
        pp = atom_pos_pred[mask]
        pt = atom_pos_target[mask]
        fids = frag_id_for_atoms[mask]
        n = pp.shape[0]
        iu = torch.triu_indices(n, n, offset=1, device=pp.device)
        # Only inter-fragment pairs contribute meaningful geometry signal;
        # within-fragment distances are rigid-body invariant (≡ 0 loss).
        inter_mask = fids[iu[0]] != fids[iu[1]]
        if inter_mask.sum() == 0:
            continue
        d_pred = (pp[iu[0][inter_mask]] - pp[iu[1][inter_mask]]).norm(dim=-1)
        d_target = (pt[iu[0][inter_mask]] - pt[iu[1][inter_mask]]).norm(dim=-1)
        sample_mse = ((d_pred - d_target) ** 2).mean()

        w = float(t_sample_flat[s].item()) ** 2
        total_weighted = total_weighted + w * sample_mse
        total_weight += w

    if total_weight <= 0.0:
        loss = torch.zeros(1, device=v_pred.device, dtype=v_pred.dtype).squeeze()
    else:
        loss = total_weighted / total_weight

    return {"loss_dg": loss}


def atom_position_auxiliary_loss(
    v_pred: Tensor,
    omega_pred: Tensor,
    v_target: Tensor,
    omega_target: Tensor,
    atom_pos_t: Tensor,
    T_frag: Tensor,
    fragment_id: Tensor,
    frag_sizes: Tensor,
) -> dict[str, Tensor]:
    """Auxiliary loss comparing atom-level velocities induced by fragment (v, omega).

    Reconstructs per-atom velocity from rigid-body kinematics and compares with
    ground-truth.  Gradient flows through ``v_pred`` and ``omega_pred``.

    The cross-product ``omega x r`` naturally weights rotation errors by lever
    arm length, making this loss geometry-aware without manual fragment-size
    weighting.

    Args:
        v_pred: Predicted fragment translation velocity ``[N_frag, 3]``.
        omega_pred: Predicted fragment angular velocity ``[N_frag, 3]``.
        v_target: Ground-truth fragment translation velocity ``[N_frag, 3]``.
        omega_target: Ground-truth fragment angular velocity ``[N_frag, 3]``.
        atom_pos_t: Current atom positions ``[N_atom, 3]``.
        T_frag: Current fragment centroids ``[N_frag, 3]``.
        fragment_id: Fragment assignment per atom ``[N_atom]``.
        frag_sizes: Fragment sizes ``[N_frag]``.

    Returns:
        Dict with ``"loss_atom_aux"`` (scalar, gradients enabled).
    """
    r = atom_pos_t - T_frag[fragment_id]  # lever arm [N_atom, 3]

    v_atom_pred = v_pred[fragment_id] + torch.cross(omega_pred[fragment_id], r, dim=-1)
    v_atom_gt = v_target[fragment_id] + torch.cross(omega_target[fragment_id], r, dim=-1)

    loss = torch.mean((v_atom_pred - v_atom_gt) ** 2)
    return {"loss_atom_aux": loss}


def dummy_position_loss(
    v_pred: Tensor,
    omega_pred: Tensor,
    atom_pos_t: Tensor,
    T_frag: Tensor,
    fragment_id: Tensor,
    dummy_to_real: Tensor,
) -> dict[str, Tensor]:
    """Position-matching loss between dummy atoms and their real counterparts.

    Each dummy atom moves with its assigned fragment.  Its real counterpart
    moves with the original fragment.  This loss penalizes the difference in
    their predicted next-step velocities, forcing adjacent fragments to agree
    at the boundary.

    Args:
        v_pred: Predicted fragment translation velocity ``[N_frag, 3]``.
        omega_pred: Predicted fragment angular velocity ``[N_frag, 3]``.
        atom_pos_t: Current atom positions ``[N_atom, 3]``.
        T_frag: Current fragment centroids ``[N_frag, 3]``.
        fragment_id: Fragment assignment per atom ``[N_atom]``.
        dummy_to_real: ``[N_dummy, 2]`` — ``(dummy_idx, real_idx)`` pairs.

    Returns:
        Dict with ``"loss_dummy"`` (scalar, gradients enabled).
    """
    zero = torch.zeros(1, device=v_pred.device, dtype=v_pred.dtype).squeeze()
    if dummy_to_real.shape[0] == 0:
        return {"loss_dummy": zero}

    d_idx = dummy_to_real[:, 0]  # dummy atom indices
    r_idx = dummy_to_real[:, 1]  # real counterpart indices

    # Velocity of dummy atom (moved by its assigned fragment)
    r_d = atom_pos_t[d_idx] - T_frag[fragment_id[d_idx]]
    v_dummy = v_pred[fragment_id[d_idx]] + torch.cross(omega_pred[fragment_id[d_idx]], r_d, dim=-1)

    # Velocity of real counterpart (moved by its own fragment)
    r_r = atom_pos_t[r_idx] - T_frag[fragment_id[r_idx]]
    v_real = v_pred[fragment_id[r_idx]] + torch.cross(omega_pred[fragment_id[r_idx]], r_r, dim=-1)

    loss = torch.mean((v_dummy - v_real) ** 2)
    return {"loss_dummy": loss}


def boundary_alignment_loss(
    v_pred: Tensor,
    omega_pred: Tensor,
    atom_pos_t: Tensor,
    T_frag: Tensor,
    fragment_id: Tensor,
    cut_src: Tensor,
    cut_dst: Tensor,
) -> dict[str, Tensor]:
    """Boundary loss: predicted velocities at cut-bond atoms should agree.

    For each cut bond (a in frag_A, b in frag_B), atom a's velocity predicted
    by frag_A and atom b's velocity predicted by frag_B should bring them to
    the same point.  This transforms rotation into a local point-matching
    problem at fragment boundaries.

    Args:
        v_pred: Predicted fragment translation velocity ``[N_frag, 3]``.
        omega_pred: Predicted fragment angular velocity ``[N_frag, 3]``.
        atom_pos_t: Current atom positions ``[N_atom, 3]``.
        T_frag: Current fragment centroids ``[N_frag, 3]``.
        fragment_id: Fragment assignment per atom ``[N_atom]``.
        cut_src: Source atom indices of cut-bond edges ``[E_cut]``.
        cut_dst: Destination atom indices of cut-bond edges ``[E_cut]``.

    Returns:
        Dict with ``"loss_boundary"`` (scalar, gradients enabled).
    """
    zero = torch.zeros(1, device=v_pred.device, dtype=v_pred.dtype).squeeze()
    if cut_src.shape[0] == 0:
        return {"loss_boundary": zero}

    # Compute predicted atom velocity for each cut-bond endpoint
    # v_atom_i = v_frag[frag_of_i] + cross(omega_frag[frag_of_i], r_i)
    r_src = atom_pos_t[cut_src] - T_frag[fragment_id[cut_src]]
    r_dst = atom_pos_t[cut_dst] - T_frag[fragment_id[cut_dst]]

    v_src = v_pred[fragment_id[cut_src]] + torch.cross(omega_pred[fragment_id[cut_src]], r_src, dim=-1)
    v_dst = v_pred[fragment_id[cut_dst]] + torch.cross(omega_pred[fragment_id[cut_dst]], r_dst, dim=-1)

    # After a dt step, src and dst should land at similar positions
    # Current distance + velocity difference should shrink
    # Loss: MSE between predicted velocities at the boundary pair
    # If fragments move consistently, v_src ≈ v_dst for bonded atoms
    loss = torch.mean((v_src - v_dst) ** 2)
    return {"loss_boundary": loss}


__all__ = [
    "flow_matching_loss",
    "atom_velocity_loss",
    "atom_position_auxiliary_loss",
    "dummy_position_loss",
    "boundary_alignment_loss",
]
