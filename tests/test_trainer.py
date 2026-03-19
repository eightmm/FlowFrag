"""Trainer dataloader tests."""

import tempfile
from pathlib import Path

import torch

from src.training.trainer import Trainer


def _make_synthetic_complex(out_dir: Path, pdb_id: str) -> None:
    complex_dir = out_dir / pdb_id
    complex_dir.mkdir(parents=True)

    n_res = 6
    n_atom = 8
    n_frag = 3
    fragment_id = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int64)
    atom_coords = torch.randn(n_atom, 3, dtype=torch.float32)

    frag_centers = torch.zeros(n_frag, 3, dtype=torch.float32)
    frag_sizes = torch.zeros(n_frag, dtype=torch.int64)
    for f in range(n_frag):
        mask = fragment_id == f
        frag_centers[f] = atom_coords[mask].mean(dim=0)
        frag_sizes[f] = mask.sum()

    protein = {
        "res_coords": torch.randn(n_res, 3, dtype=torch.float32),
        "res_type": torch.randint(0, 21, (n_res,), dtype=torch.int64),
    }
    ligand = {
        "atom_coords": atom_coords,
        "atom_element": torch.randint(0, 13, (n_atom,), dtype=torch.int64),
        "atom_charge": torch.zeros(n_atom, dtype=torch.int8),
        "atom_aromatic": torch.zeros(n_atom, dtype=torch.bool),
        "atom_hybridization": torch.ones(n_atom, dtype=torch.int8),
        "atom_in_ring": torch.zeros(n_atom, dtype=torch.bool),
        "bond_index": torch.tensor(
            [[0, 1, 1, 2, 3, 4, 5, 6], [1, 0, 2, 1, 4, 3, 6, 5]],
            dtype=torch.int64,
        ),
        "bond_type": torch.zeros(8, dtype=torch.int8),
        "bond_conjugated": torch.zeros(8, dtype=torch.bool),
        "bond_in_ring": torch.zeros(8, dtype=torch.bool),
        "fragment_id": fragment_id,
        "frag_centers": frag_centers,
        "frag_local_coords": atom_coords - frag_centers[fragment_id],
        "frag_sizes": frag_sizes,
    }
    meta = {
        "pdb_id": pdb_id,
        "pocket_center": protein["res_coords"].mean(dim=0),
        "num_res": torch.tensor(n_res, dtype=torch.int64),
        "num_atom": torch.tensor(n_atom, dtype=torch.int64),
        "num_frag": torch.tensor(n_frag, dtype=torch.int64),
        "used_mol2_fallback": torch.tensor(False),
        "schema_version": 1,
    }

    torch.save(protein, complex_dir / "protein.pt")
    torch.save(ligand, complex_dir / "ligand.pt")
    torch.save(meta, complex_dir / "meta.pt")


def test_trainer_strict_overfit_subset_is_fixed_and_repeatable():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for i in range(6):
            _make_synthetic_complex(root, f"complex_{i:03d}")

        trainer = Trainer.__new__(Trainer)
        trainer.cfg = {
            "data": {
                "data_dir": tmpdir,
                "min_atoms": 1,
                "max_atoms": 80,
                "max_frags": 20,
                "prior_sigma": 1.0,
                "num_workers": 0,
                "val_split": 0.0,
                "rotation_augmentation": "ligand_uniform",
                "deterministic": True,
            },
            "training": {
                "batch_size": 2,
                "overfit_batches": 2,
                "seed": 123,
            },
        }
        trainer.world_size = 1
        trainer.rank = 0
        trainer.is_main = False

        trainer._build_dataloaders()

        assert len(trainer.train_loader.dataset) == 4

        iter_a = iter(trainer.train_loader)
        batch_a1 = next(iter_a)
        batch_a2 = next(iter_a)

        iter_b = iter(trainer.train_loader)
        batch_b1 = next(iter_b)
        batch_b2 = next(iter_b)

        assert batch_a1.pdb_id == batch_b1.pdb_id
        assert batch_a2.pdb_id == batch_b2.pdb_id
        assert torch.allclose(batch_a1.t, batch_b1.t)
        assert torch.allclose(batch_a2.t, batch_b2.t)
