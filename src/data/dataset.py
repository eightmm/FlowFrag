"""FlowFrag dataset: loads preprocessed complexes and samples flow-matching states."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from src.geometry.flow_matching import (
    compute_flow_matching_targets,
    sample_prior_poses,
)
from src.geometry.se3 import (
    quaternion_inverse,
    quaternion_to_matrix,
    sample_uniform_quaternion,
)


class FlowFragDataset(Dataset):
    """Dataset for fragment-based flow matching on SE(3).

    Each sample returns a ``HeteroData`` graph with protein residues, ligand
    atoms, fragment poses at time ``t``, and ground-truth velocity targets.

    Args:
        root: Directory containing per-complex subdirectories, each with
            ``protein.pt``, ``ligand.pt``, ``meta.pt``.
        split_file: Path to a text file (one PDB ID per line) or a JSON
            file with ``{"train": [...], "val": [...]}``.  When JSON,
            *split_key* selects the list to use.
            If ``None``, all subdirectories in *root* are used.
        split_key: Key to use when *split_file* is JSON (default ``"train"``).
        translation_sigma: Gaussian std for the translation prior (Angstroms).
        max_atoms: Discard complexes with more heavy atoms.
        max_frags: Discard complexes with more fragments.
        min_atoms: Discard complexes with fewer heavy atoms.
        rotation_augmentation: One of ``"none"``, ``"ligand_uniform"``, or
            ``"per_fragment"``. Controls how fragment local coordinates are
            rotated before defining the target pose.
        deterministic: Default deterministic setting for augmentation, prior,
            and time sampling. Individual controls below override it.
        deterministic_augmentation: Whether target-gauge augmentation should be
            fixed per sample.
        deterministic_prior: Whether the prior pose ``(T_0, q_0)`` should be
            fixed per sample.
        deterministic_time: Whether the interpolation time ``t`` should be
            fixed per sample.
        prior_bank_size: Number of fixed prior variants to cache per complex.
            Requires ``deterministic_prior=True`` when greater than 1.
        time_bank_size: Number of fixed time samples to cache per complex.
            Requires ``deterministic_time=True`` when greater than 1.
        seed: Base seed used when ``deterministic=True``.
    """

    def __init__(
        self,
        root: str | Path,
        split_file: str | Path | None = None,
        split_key: str = "train",
        translation_sigma: float = 10.0,
        max_atoms: int = 80,
        max_frags: int = 20,
        min_atoms: int = 5,
        rotation_augmentation: str = "none",
        deterministic: bool = False,
        deterministic_augmentation: bool | None = None,
        deterministic_prior: bool | None = None,
        deterministic_time: bool | None = None,
        prior_bank_size: int = 1,
        time_bank_size: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.translation_sigma = translation_sigma
        self.rotation_augmentation = rotation_augmentation
        self.deterministic = deterministic
        self.deterministic_augmentation = (
            deterministic if deterministic_augmentation is None else deterministic_augmentation
        )
        self.deterministic_prior = (
            deterministic if deterministic_prior is None else deterministic_prior
        )
        self.deterministic_time = (
            deterministic if deterministic_time is None else deterministic_time
        )
        self.prior_bank_size = prior_bank_size
        self.time_bank_size = time_bank_size
        self.seed = seed

        valid_aug = {"none", "ligand_uniform", "per_fragment"}
        if self.rotation_augmentation not in valid_aug:
            raise ValueError(
                f"rotation_augmentation must be one of {sorted(valid_aug)}, "
                f"got {self.rotation_augmentation!r}."
            )
        if self.prior_bank_size < 1:
            raise ValueError(f"prior_bank_size must be >= 1, got {self.prior_bank_size}.")
        if self.time_bank_size < 1:
            raise ValueError(f"time_bank_size must be >= 1, got {self.time_bank_size}.")
        if self.prior_bank_size > 1 and not self.deterministic_prior:
            raise ValueError(
                "prior_bank_size > 1 requires deterministic_prior=True so the bank is fixed."
            )
        if self.time_bank_size > 1 and not self.deterministic_time:
            raise ValueError(
                "time_bank_size > 1 requires deterministic_time=True so the bank is fixed."
            )

        # Collect PDB IDs
        if split_file is not None:
            sf = Path(split_file)
            if sf.suffix == ".json":
                with open(sf) as f:
                    split_data = json.load(f)
                pdb_ids = split_data[split_key]
            else:
                with open(sf) as f:
                    pdb_ids = [line.strip() for line in f if line.strip()]
        else:
            pdb_ids = sorted(d.name for d in self.root.iterdir() if d.is_dir())

        # Filter by size limits using meta.pt
        self.pdb_ids: list[str] = []
        for pid in pdb_ids:
            meta_path = self.root / pid / "meta.pt"
            if not meta_path.exists():
                continue
            meta = torch.load(meta_path, weights_only=True)
            n_atom = meta["num_atom"].item()
            n_frag = meta["num_frag"].item()
            if n_atom < min_atoms or n_atom > max_atoms or n_frag > max_frags:
                continue
            self.pdb_ids.append(pid)

    def __len__(self) -> int:
        return len(self.pdb_ids) * self.samples_per_complex

    @property
    def samples_per_complex(self) -> int:
        return self.prior_bank_size * self.time_bank_size

    def _decode_sample_index(self, idx: int) -> tuple[int, int, int]:
        complex_idx = idx // self.samples_per_complex
        variant_idx = idx % self.samples_per_complex
        prior_variant_idx = variant_idx % self.prior_bank_size
        time_variant_idx = variant_idx // self.prior_bank_size
        return complex_idx, prior_variant_idx, time_variant_idx

    def _make_generator(
        self,
        complex_idx: int,
        *,
        variant_idx: int = 0,
        stream_offset: int,
        deterministic: bool,
    ) -> torch.Generator | None:
        if not deterministic:
            return None

        generator = torch.Generator()
        generator.manual_seed(
            self.seed
            + complex_idx * 9_973
            + variant_idx * 1_000_003
            + stream_offset * 104_729
        )
        return generator

    @staticmethod
    def _identity_quaternion(
        num_fragments: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        q = torch.zeros(num_fragments, 4, dtype=dtype)
        q[:, 0] = 1.0
        return q

    def _sample_target_rotation(
        self,
        complex_idx: int,
        n_frags: int,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample augmentation quaternions and resulting target rotations."""
        identity_q = self._identity_quaternion(n_frags, dtype=dtype)

        if self.rotation_augmentation == "none":
            return identity_q, identity_q

        generator = self._make_generator(
            complex_idx,
            stream_offset=1,
            deterministic=self.deterministic_augmentation,
        )
        if self.rotation_augmentation == "ligand_uniform":
            aug_q = sample_uniform_quaternion(1, dtype=dtype, generator=generator)
            aug_q = aug_q.expand(n_frags, -1).clone()
        else:
            aug_q = sample_uniform_quaternion(n_frags, dtype=dtype, generator=generator)

        q_target = quaternion_inverse(aug_q)
        return aug_q, q_target

    def __getitem__(self, idx: int) -> HeteroData:
        complex_idx, prior_variant_idx, time_variant_idx = self._decode_sample_index(idx)
        pdb_id = self.pdb_ids[complex_idx]
        data_dir = self.root / pdb_id

        protein = torch.load(data_dir / "protein.pt", weights_only=True)
        ligand = torch.load(data_dir / "ligand.pt", weights_only=True)
        meta = torch.load(data_dir / "meta.pt", weights_only=True)

        pocket_center = meta["pocket_center"]  # [3]

        data = HeteroData()
        data.pdb_id = pdb_id

        # --- Protein (static context) ---
        data["protein"].pos = protein["res_coords"] - pocket_center
        data["protein"].x = protein["res_type"]
        data["protein"].num_nodes = protein["res_coords"].shape[0]

        # --- Ligand atoms (static features) ---
        data["atom"].x = ligand["atom_element"]
        data["atom"].charge = ligand["atom_charge"]
        data["atom"].aromatic = ligand["atom_aromatic"]
        data["atom"].hybridization = ligand["atom_hybridization"]
        data["atom"].in_ring = ligand["atom_in_ring"]
        data["atom"].fragment_id = ligand["fragment_id"]
        data["atom"].num_nodes = ligand["atom_element"].shape[0]

        # --- Bonds ---
        data["atom", "bond", "atom"].edge_index = ligand["bond_index"]
        data["atom", "bond", "atom"].edge_attr = torch.stack(
            [
                ligand["bond_type"].to(torch.float32),
                ligand["bond_conjugated"].to(torch.float32),
                ligand["bond_in_ring"].to(torch.float32),
            ],
            dim=-1,
        )

        # --- Triangulation edges (cross-fragment, backward compatible) ---
        tri_idx = ligand.get("tri_edge_index", torch.zeros(2, 0, dtype=torch.int64))
        tri_ref = ligand.get("tri_edge_ref_dist", torch.zeros(0, dtype=torch.float32))
        data["atom", "tri", "atom"].edge_index = tri_idx
        data["atom", "tri", "atom"].ref_dist = tri_ref

        # --- Explicit cut-bond edges (rotatable bonds across fragments) ---
        cut_idx = ligand.get("cut_bond_index", torch.zeros(2, 0, dtype=torch.int64))
        if cut_idx.numel() > 0:
            cut_idx = torch.cat([cut_idx, cut_idx.flip(0)], dim=1)
        data["atom", "cut", "atom"].edge_index = cut_idx

        # --- Fragment adjacency (topological, from cut bonds) ---
        frag_adj = ligand.get("fragment_adj_index", torch.zeros(2, 0, dtype=torch.int64))
        data["fragment", "adj", "fragment"].edge_index = frag_adj

        # --- Fragment target (crystal pose, pocket-centered) ---
        T_1 = ligand["frag_centers"] - pocket_center  # [N_frag, 3]
        frag_sizes = ligand["frag_sizes"]  # [N_frag]
        n_frags = T_1.shape[0]
        identity_q = self._identity_quaternion(n_frags, dtype=T_1.dtype)
        prior_generator = self._make_generator(
            complex_idx,
            variant_idx=prior_variant_idx,
            stream_offset=2,
            deterministic=self.deterministic_prior,
        )
        time_generator = self._make_generator(
            complex_idx,
            variant_idx=time_variant_idx,
            stream_offset=3,
            deterministic=self.deterministic_time,
        )

        # Optional gauge augmentation for rotation experiments.
        R_aug_q, q_1 = self._sample_target_rotation(complex_idx, n_frags, dtype=T_1.dtype)
        R_aug = quaternion_to_matrix(R_aug_q)  # [N_frag, 3, 3]

        # Rotate local_coords: local_aug = R_aug @ local
        frag_id = ligand["fragment_id"]
        local_pos_orig = ligand["frag_local_coords"]
        local_pos = torch.einsum("nij,nj->ni", R_aug[frag_id], local_pos_orig)

        # Mask single-atom fragments to identity (rotation irrelevant)
        single_mask = frag_sizes <= 1
        q_1[single_mask] = identity_q[single_mask]

        # --- Flow matching sampling ---
        t = torch.rand(1, generator=time_generator, dtype=T_1.dtype).item()

        # Sample prior (t=0)
        T_0, q_0 = sample_prior_poses(
            n_frags,
            pocket_center=torch.zeros(3, dtype=T_1.dtype),  # already centered
            translation_sigma=self.translation_sigma,
            frag_sizes=frag_sizes,
            generator=prior_generator,
        )

        # Interpolated state + velocity targets
        targets = compute_flow_matching_targets(
            T_0, q_0, T_1, q_1, t, frag_sizes=frag_sizes,
        )

        data["fragment"].num_nodes = n_frags
        data["fragment"].T_frag = targets["T_t"]
        data["fragment"].q_frag = targets["q_t"]
        data["fragment"].v_target = targets["v_t"]
        data["fragment"].omega_target = targets["omega_t"]
        data["fragment"].size = frag_sizes
        data["fragment"].T_target = T_1
        data["fragment"].q_target = q_1
        data["fragment"].T_prior = T_0
        data["fragment"].q_prior = q_0
        data.prior_variant_idx = prior_variant_idx
        data.time_variant_idx = time_variant_idx
        data.t = torch.tensor([t], dtype=T_1.dtype)

        # --- Atom positions at time t (for spatial edge construction) ---
        R_t = quaternion_to_matrix(targets["q_t"])  # [N_frag, 3, 3]
        atom_pos_t = (
            torch.einsum("nij,nj->ni", R_t[frag_id], local_pos)
            + targets["T_t"][frag_id]
        )
        data["atom"].pos_t = atom_pos_t
        data["atom"].local_pos = local_pos  # augmented local coords

        # --- Per-atom velocity target: v_atom = v_frag + omega × r ---
        r = atom_pos_t - targets["T_t"][frag_id]
        v_atom = targets["v_t"][frag_id] + torch.cross(
            targets["omega_t"][frag_id], r, dim=-1,
        )
        data["atom"].v_target = v_atom

        return data


__all__ = ["FlowFragDataset"]
