"""Tests for FlowFragDataset."""

import tempfile
from pathlib import Path

import torch

from src.data.dataset import FlowFragDataset


def _make_synthetic_complex(out_dir: Path, pdb_id: str = "test_001") -> Path:
    """Create a minimal synthetic complex for testing."""
    complex_dir = out_dir / pdb_id
    complex_dir.mkdir(parents=True)

    n_res = 10
    n_atom = 8
    n_frag = 3
    n_bonds = 7

    protein = {
        "res_coords": torch.randn(n_res, 3, dtype=torch.float32),
        "res_type": torch.randint(0, 21, (n_res,), dtype=torch.int64),
    }

    # Fragment assignment: 3 atoms in frag 0, 3 in frag 1, 2 in frag 2
    fragment_id = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int64)
    atom_coords = torch.randn(n_atom, 3, dtype=torch.float32)

    # Compute centroids and local coords
    frag_centers = torch.zeros(n_frag, 3, dtype=torch.float32)
    frag_sizes = torch.zeros(n_frag, dtype=torch.int64)
    for f in range(n_frag):
        mask = fragment_id == f
        frag_centers[f] = atom_coords[mask].mean(dim=0)
        frag_sizes[f] = mask.sum()
    frag_local_coords = atom_coords - frag_centers[fragment_id]

    # Simple chain bonds: 0-1, 1-2, ..., 6-7 (directed both ways)
    src = list(range(n_bonds)) + list(range(1, n_bonds + 1))
    dst = list(range(1, n_bonds + 1)) + list(range(n_bonds))
    e_dir = len(src)

    ligand = {
        "atom_coords": atom_coords,
        "atom_element": torch.randint(0, 13, (n_atom,), dtype=torch.int64),
        "atom_charge": torch.zeros(n_atom, dtype=torch.int8),
        "atom_aromatic": torch.zeros(n_atom, dtype=torch.bool),
        "atom_hybridization": torch.ones(n_atom, dtype=torch.int8),
        "atom_in_ring": torch.zeros(n_atom, dtype=torch.bool),
        "bond_index": torch.tensor([src, dst], dtype=torch.int64),
        "bond_type": torch.zeros(e_dir, dtype=torch.int8),
        "bond_conjugated": torch.zeros(e_dir, dtype=torch.bool),
        "bond_in_ring": torch.zeros(e_dir, dtype=torch.bool),
        "fragment_id": fragment_id,
        "frag_centers": frag_centers,
        "frag_local_coords": frag_local_coords,
        "frag_sizes": frag_sizes,
        "cut_bond_index": torch.tensor([[2, 5], [3, 6]], dtype=torch.int64),
    }

    pocket_center = protein["res_coords"].mean(dim=0)
    meta = {
        "pdb_id": pdb_id,
        "pocket_center": pocket_center,
        "num_res": torch.tensor(n_res, dtype=torch.int64),
        "num_atom": torch.tensor(n_atom, dtype=torch.int64),
        "num_frag": torch.tensor(n_frag, dtype=torch.int64),
        "used_mol2_fallback": torch.tensor(False),
        "schema_version": 1,
    }

    torch.save(protein, complex_dir / "protein.pt")
    torch.save(ligand, complex_dir / "ligand.pt")
    torch.save(meta, complex_dir / "meta.pt")
    return complex_dir


def test_dataset_loads_synthetic_complex():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_complex(Path(tmpdir), "complex_001")
        ds = FlowFragDataset(tmpdir, min_atoms=1)
        assert len(ds) == 1

        sample = ds[0]
        assert sample.pdb_id == "complex_001"
        assert sample["protein"].pos.shape == (10, 3)
        assert sample["fragment"].T_frag.shape == (3, 3)
        assert sample["fragment"].q_frag.shape == (3, 4)
        assert sample["fragment"].v_target.shape == (3, 3)
        assert sample["fragment"].omega_target.shape == (3, 3)
        assert sample["atom"].pos_t.shape == (8, 3)
        assert sample["atom", "cut", "atom"].edge_index.shape == (2, 4)


def test_dataset_filters_by_atom_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_complex(Path(tmpdir), "small")
        # n_atom=8, so max_atoms=5 should filter it out
        ds = FlowFragDataset(tmpdir, max_atoms=5, min_atoms=1)
        assert len(ds) == 0

        # min_atoms=10 should also filter
        ds = FlowFragDataset(tmpdir, min_atoms=10)
        assert len(ds) == 0

        # Permissive limits should include it
        ds = FlowFragDataset(tmpdir, max_atoms=80, min_atoms=1)
        assert len(ds) == 1


def test_dataset_filters_by_frag_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_complex(Path(tmpdir), "many_frags")
        # n_frag=3, max_frags=2 should exclude
        ds = FlowFragDataset(tmpdir, max_frags=2, min_atoms=1)
        assert len(ds) == 0

        ds = FlowFragDataset(tmpdir, max_frags=10, min_atoms=1)
        assert len(ds) == 1


def test_dataset_split_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_complex(Path(tmpdir), "a001")
        _make_synthetic_complex(Path(tmpdir), "b002")

        split_path = Path(tmpdir) / "split.txt"
        split_path.write_text("a001\n")

        ds = FlowFragDataset(tmpdir, split_file=str(split_path), min_atoms=1)
        assert len(ds) == 1
        assert ds[0].pdb_id == "a001"


def test_dataset_time_in_unit_interval():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_complex(Path(tmpdir), "t_check")
        ds = FlowFragDataset(tmpdir, min_atoms=1)
        sample = ds[0]
        assert 0.0 <= sample.t.item() <= 1.0


def test_dataset_deterministic_sampling_repeats_identical_targets():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_complex(Path(tmpdir), "repeatable")
        ds = FlowFragDataset(
            tmpdir,
            min_atoms=1,
            rotation_augmentation="ligand_uniform",
            deterministic=True,
            seed=123,
        )

        sample_a = ds[0]
        sample_b = ds[0]

        assert torch.allclose(sample_a.t, sample_b.t)
        assert torch.allclose(sample_a["fragment"].q_frag, sample_b["fragment"].q_frag)
        assert torch.allclose(sample_a["fragment"].q_target, sample_b["fragment"].q_target)
        assert torch.allclose(sample_a["fragment"].omega_target, sample_b["fragment"].omega_target)
        assert torch.allclose(sample_a["atom"].pos_t, sample_b["atom"].pos_t)


def test_dataset_ligand_uniform_rotation_uses_shared_target_quaternion():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_complex(Path(tmpdir), "uniform_aug")
        ds = FlowFragDataset(
            tmpdir,
            min_atoms=1,
            rotation_augmentation="ligand_uniform",
            deterministic=True,
            seed=7,
        )

        sample = ds[0]
        q_target = sample["fragment"].q_target
        ref = q_target[0].expand_as(q_target)
        assert torch.allclose(q_target, ref, atol=1e-6)


def test_dataset_can_fix_prior_and_target_while_time_varies():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_complex(Path(tmpdir), "t_only")
        ds = FlowFragDataset(
            tmpdir,
            min_atoms=1,
            rotation_augmentation="ligand_uniform",
            deterministic=False,
            deterministic_augmentation=True,
            deterministic_prior=True,
            deterministic_time=False,
            seed=11,
        )

        samples = [ds[0] for _ in range(4)]
        ref = samples[0]

        for sample in samples[1:]:
            assert torch.allclose(sample["fragment"].q_target, ref["fragment"].q_target)
            assert torch.allclose(sample["fragment"].T_prior, ref["fragment"].T_prior)
            assert torch.allclose(sample["fragment"].q_prior, ref["fragment"].q_prior)

        rounded_t = {round(float(sample.t.item()), 6) for sample in samples}
        assert len(rounded_t) > 1
        assert any(
            not torch.allclose(sample["atom"].pos_t, ref["atom"].pos_t)
            for sample in samples[1:]
        )


def test_dataset_prior_and_time_banks_expand_fixed_variants():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_complex(Path(tmpdir), "banked")
        ds = FlowFragDataset(
            tmpdir,
            min_atoms=1,
            rotation_augmentation="ligand_uniform",
            deterministic=False,
            deterministic_augmentation=True,
            deterministic_prior=True,
            deterministic_time=True,
            prior_bank_size=2,
            time_bank_size=3,
            seed=17,
        )

        assert len(ds) == 6

        sample_p0_t0 = ds[0]
        sample_p1_t0 = ds[1]
        sample_p0_t1 = ds[2]

        assert sample_p0_t0.pdb_id == "banked"
        assert sample_p1_t0.pdb_id == "banked"
        assert sample_p0_t1.pdb_id == "banked"

        assert torch.allclose(sample_p0_t0["fragment"].q_target, sample_p1_t0["fragment"].q_target)
        assert torch.allclose(sample_p0_t0["fragment"].q_target, sample_p0_t1["fragment"].q_target)

        assert not torch.allclose(sample_p0_t0["fragment"].q_prior, sample_p1_t0["fragment"].q_prior)
        assert torch.allclose(sample_p0_t0["fragment"].q_prior, sample_p0_t1["fragment"].q_prior)

        assert torch.allclose(sample_p0_t0.t, sample_p1_t0.t)
        assert not torch.allclose(sample_p0_t0.t, sample_p0_t1.t)


def test_dataset_default_rotation_target_is_identity():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_complex(Path(tmpdir), "identity_target")
        ds = FlowFragDataset(tmpdir, min_atoms=1)
        sample = ds[0]

        q_target = sample["fragment"].q_target
        expected = torch.zeros_like(q_target)
        expected[:, 0] = 1.0
        assert torch.allclose(q_target, expected, atol=1e-6)


def test_dataset_single_atom_fragment_gets_zero_omega():
    """Single-atom fragments should have zero angular velocity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        complex_dir = Path(tmpdir) / "single_atom"
        complex_dir.mkdir()

        n_atom = 4
        fragment_id = torch.tensor([0, 0, 0, 1], dtype=torch.int64)
        atom_coords = torch.randn(n_atom, 3, dtype=torch.float32)
        frag_centers = torch.zeros(2, 3, dtype=torch.float32)
        frag_sizes = torch.tensor([3, 1], dtype=torch.int64)
        for f in range(2):
            mask = fragment_id == f
            frag_centers[f] = atom_coords[mask].mean(dim=0)

        protein = {
            "res_coords": torch.randn(5, 3, dtype=torch.float32),
            "res_type": torch.randint(0, 21, (5,), dtype=torch.int64),
        }
        ligand = {
            "atom_coords": atom_coords,
            "atom_element": torch.zeros(n_atom, dtype=torch.int64),
            "atom_charge": torch.zeros(n_atom, dtype=torch.int8),
            "atom_aromatic": torch.zeros(n_atom, dtype=torch.bool),
            "atom_hybridization": torch.zeros(n_atom, dtype=torch.int8),
            "atom_in_ring": torch.zeros(n_atom, dtype=torch.bool),
            "bond_index": torch.tensor(
                [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.int64,
            ),
            "bond_type": torch.zeros(6, dtype=torch.int8),
            "bond_conjugated": torch.zeros(6, dtype=torch.bool),
            "bond_in_ring": torch.zeros(6, dtype=torch.bool),
            "fragment_id": fragment_id,
            "frag_centers": frag_centers,
            "frag_local_coords": atom_coords - frag_centers[fragment_id],
            "frag_sizes": frag_sizes,
        }
        meta = {
            "pdb_id": "single_atom",
            "pocket_center": protein["res_coords"].mean(dim=0),
            "num_res": torch.tensor(5, dtype=torch.int64),
            "num_atom": torch.tensor(n_atom, dtype=torch.int64),
            "num_frag": torch.tensor(2, dtype=torch.int64),
            "used_mol2_fallback": torch.tensor(False),
            "schema_version": 1,
        }
        torch.save(protein, complex_dir / "protein.pt")
        torch.save(ligand, complex_dir / "ligand.pt")
        torch.save(meta, complex_dir / "meta.pt")

        ds = FlowFragDataset(tmpdir, min_atoms=1)
        sample = ds[0]

        # Single-atom fragment (idx 1) should have zero omega
        omega_single = sample["fragment"].omega_target[1]
        assert torch.allclose(omega_single, torch.zeros(3), atol=1e-6)
