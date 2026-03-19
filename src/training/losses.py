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
) -> dict[str, Tensor]:
    """Compute flow matching loss: MSE on translation and angular velocities.

    Args:
        v_pred: Predicted translation velocity ``[N_frag, 3]``.
        omega_pred: Predicted angular velocity ``[N_frag, 3]``.
        v_target: Ground-truth translation velocity ``[N_frag, 3]``.
        omega_target: Ground-truth angular velocity ``[N_frag, 3]``.
        frag_sizes: Fragment sizes ``[N_frag]``.
        omega_weight: Weight for angular velocity loss relative to translation.
        R_t: Current rotation matrices ``[N_frag, 3, 3]``. Required when
            ``omega_loss_frame="body"``.
        omega_loss_frame: ``"world"`` (default) or ``"body"``.  When ``"body"``,
            both pred and target are rotated into the fragment body frame via
            ``R_t^T`` before comparison.
        omega_loss_type: ``"mse"`` (default), ``"direction_magnitude"``, or
            ``"cosine_weighted"``.
        omega_dir_weight: Weight for direction loss when using
            ``direction_magnitude``.
        omega_mag_weight: Weight for magnitude loss when using
            ``direction_magnitude``.

    Returns:
        Dict with ``"loss"``, ``"loss_v"``, ``"loss_omega"``, ``"cos_v"``,
        ``"cos_omega"`` (all scalar), plus optional ``"loss_omega_dir"`` and
        ``"loss_omega_mag"`` when using ``direction_magnitude``.
    """
    loss_v = torch.mean((v_pred - v_target) ** 2)

    multi_mask = frag_sizes > 1
    zero = torch.zeros(1, device=v_pred.device, dtype=v_pred.dtype).squeeze()

    extras: dict[str, Tensor] = {}

    if multi_mask.any():
        op = omega_pred[multi_mask]
        ot = omega_target[multi_mask]

        # Body-frame canonicalization
        if omega_loss_frame == "body":
            assert R_t is not None, "R_t required for body-frame omega loss"
            Rt_inv = R_t[multi_mask].transpose(-1, -2)  # [M, 3, 3]
            op = torch.einsum("nij,nj->ni", Rt_inv, op)
            ot = torch.einsum("nij,nj->ni", Rt_inv, ot)

        # Omega loss computation
        if omega_loss_type == "mse":
            loss_omega = _omega_mse(op, ot)
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


__all__ = ["flow_matching_loss", "atom_velocity_loss", "atom_position_auxiliary_loss"]
