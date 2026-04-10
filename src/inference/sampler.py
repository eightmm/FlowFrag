"""ODE sampler utilities for the unified fragment-flow pipeline."""

from __future__ import annotations

import torch
from torch import Tensor


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


__all__ = ["build_time_grid"]
