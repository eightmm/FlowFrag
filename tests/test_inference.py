"""Tests for inference pipeline: sampler and metrics."""

from __future__ import annotations

import torch
import pytest

from src.inference.metrics import ligand_rmsd, centroid_distance, frag_centroid_rmsd, success_rate


class TestMetrics:
    def test_rmsd_identical_is_zero(self):
        pos = torch.randn(20, 3)
        assert ligand_rmsd(pos, pos).item() == pytest.approx(0.0, abs=1e-6)

    def test_rmsd_known_value(self):
        pred = torch.zeros(4, 3)
        true = torch.ones(4, 3)
        # sqrt(mean(3)) = sqrt(3)
        expected = 3.0 ** 0.5
        assert ligand_rmsd(pred, true).item() == pytest.approx(expected, rel=1e-4)

    def test_centroid_distance_identical_is_zero(self):
        pos = torch.randn(10, 3)
        assert centroid_distance(pos, pos).item() == pytest.approx(0.0, abs=1e-6)

    def test_centroid_distance_known_shift(self):
        true = torch.zeros(5, 3)
        pred = torch.ones(5, 3) * 3.0  # centroid at (3,3,3)
        expected = (3.0 * 9.0) ** 0.5  # sqrt(27)
        assert centroid_distance(pred, true).item() == pytest.approx(expected, rel=1e-4)

    def test_frag_centroid_rmsd(self):
        T_pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        T_true = torch.zeros(2, 3)
        # sqrt(mean([1, 1])) = 1.0
        assert frag_centroid_rmsd(T_pred, T_true).item() == pytest.approx(1.0, rel=1e-4)

    def test_success_rate_all_below(self):
        rmsds = torch.tensor([0.5, 1.0, 1.5, 1.9])
        assert success_rate(rmsds, threshold=2.0).item() == pytest.approx(1.0)

    def test_success_rate_none_below(self):
        rmsds = torch.tensor([3.0, 4.0, 5.0])
        assert success_rate(rmsds, threshold=2.0).item() == pytest.approx(0.0)

    def test_success_rate_partial(self):
        rmsds = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # Only 1.0 < 2.0
        assert success_rate(rmsds, threshold=2.0).item() == pytest.approx(0.25)


class TestSampler:
    @pytest.fixture
    def model_and_data(self):
        """Build a minimal FlowFrag model and a single test complex."""
        from src.models.flowfrag import FlowFrag
        from src.data.dataset import FlowFragDataset

        model = FlowFrag(
            hidden_dim=64, num_encoder_layers_prot=1, num_encoder_layers_lig=1,
            num_docking_layers=1, hidden_scalar_dim=64, hidden_vec_dim=16,
        )
        model.train(False)

        ds = FlowFragDataset(root="data/processed", max_atoms=80, max_frags=20, min_atoms=5)
        assert len(ds) > 0, "Need at least one processed complex for testing"
        data = ds[0]
        return model, data

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sampler_output_shapes(self, model_and_data):
        from src.inference.sampler import FlowFragSampler

        model, data = model_and_data
        device = torch.device("cuda")
        model = model.to(device)

        sampler = FlowFragSampler(model, num_steps=3, translation_sigma=10.0)
        result = sampler.sample(data, device=device)

        n_frags = data["fragment"].num_nodes
        n_atoms = data["atom"].num_nodes
        assert result["T_pred"].shape == (n_frags, 3)
        assert result["q_pred"].shape == (n_frags, 4)
        assert result["atom_pos_pred"].shape == (n_atoms, 3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sampler_quaternions_are_unit(self, model_and_data):
        from src.inference.sampler import FlowFragSampler

        model, data = model_and_data
        device = torch.device("cuda")
        model = model.to(device)

        sampler = FlowFragSampler(model, num_steps=3, translation_sigma=10.0)
        result = sampler.sample(data, device=device)

        q_norms = torch.linalg.vector_norm(result["q_pred"], dim=-1)
        assert torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-4)
