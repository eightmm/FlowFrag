"""Tests for inference time discretization."""

import torch

from src.inference.sampler import build_time_grid


def test_build_time_grid_uniform_matches_linspace():
    grid = build_time_grid(5, schedule="uniform", power=3.0)
    expected = torch.linspace(0.0, 1.0, 6)
    torch.testing.assert_close(grid, expected)


def test_build_time_grid_late_is_denser_near_t1():
    grid = build_time_grid(8, schedule="late", power=3.0)
    diffs = grid[1:] - grid[:-1]
    assert torch.isclose(grid[0], torch.tensor(0.0))
    assert torch.isclose(grid[-1], torch.tensor(1.0))
    assert torch.all(diffs > 0)
    assert diffs[0] > diffs[-1]


def test_build_time_grid_early_is_denser_near_t0():
    grid = build_time_grid(8, schedule="early", power=3.0)
    diffs = grid[1:] - grid[:-1]
    assert torch.isclose(grid[0], torch.tensor(0.0))
    assert torch.isclose(grid[-1], torch.tensor(1.0))
    assert torch.all(diffs > 0)
    assert diffs[0] < diffs[-1]
