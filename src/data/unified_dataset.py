"""Unified dataset: loads graph.pt (static topology) + ligand.pt + meta.pt.

Returns a flat dict (not HeteroData) with all node/edge tensors from graph.pt,
plus flow matching state (T_frag, q_frag at time t) and velocity targets.
Fragment and atom node coordinates are updated to reflect the interpolated pose.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.geometry.flow_matching import (
    compute_flow_matching_targets,
    sample_prior_poses,
)
from src.geometry.se3 import (
    quaternion_inverse,
    quaternion_to_matrix,
    sample_uniform_quaternion,
)


class UnifiedDataset(Dataset):
    """Dataset for unified equivariant model.

    Loads the precomputed unified graph (graph.pt) which contains all node
    types and edge types in a single flat structure.  At runtime, applies
    flow matching sampling and updates dynamic coordinates.

    Args:
        root: Directory with per-complex subdirs containing graph.pt, ligand.pt, meta.pt.
        split_file: JSON or text file with PDB IDs.
        split_key: Key for JSON split files.
        translation_sigma: Gaussian std for translation prior (Angstroms).
        max_atoms: Max ligand heavy atoms.
        max_frags: Max fragments.
        min_atoms: Min ligand heavy atoms.
        rotation_augmentation: "none", "ligand_uniform", or "per_fragment".
        deterministic: Fix all random sampling per sample.
        seed: Base seed for deterministic mode.
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

        # Filter by size + require graph.pt
        self.pdb_ids: list[str] = []
        for pid in pdb_ids:
            meta_path = self.root / pid / "meta.pt"
            graph_path = self.root / pid / "graph.pt"
            if not meta_path.exists() or not graph_path.exists():
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

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        complex_idx, prior_variant_idx, time_variant_idx = self._decode_sample_index(idx)
        pdb_id = self.pdb_ids[complex_idx]
        data_dir = self.root / pdb_id

        graph = torch.load(data_dir / "graph.pt", weights_only=True)
        ligand = torch.load(data_dir / "ligand.pt", weights_only=True)
        meta = torch.load(data_dir / "meta.pt", weights_only=True)

        pocket_center = meta["pocket_center"]
        n_frags = meta["num_frag"].item()

        # --- Fragment target (crystal pose, pocket-centered) ---
        T_1 = ligand["frag_centers"] - pocket_center  # [N_frag, 3]

        # Fragment sizes from ligand (real atoms only, excluding dummies)
        is_dummy = ligand.get("is_dummy")
        if is_dummy is not None and is_dummy.any():
            frag_id_real = ligand["fragment_id"][~is_dummy]
        else:
            frag_id_real = ligand["fragment_id"]
        frag_sizes = torch.zeros(n_frags, dtype=torch.int64)
        frag_sizes.scatter_add_(0, frag_id_real, torch.ones_like(frag_id_real))

        # Local coords for atom position reconstruction
        if is_dummy is not None and is_dummy.any():
            local_pos_orig = ligand["frag_local_coords"][~is_dummy]
            frag_id_for_atoms = frag_id_real
        else:
            local_pos_orig = ligand["frag_local_coords"]
            frag_id_for_atoms = ligand["fragment_id"]

        # --- Rotation augmentation ---
        identity_q = torch.zeros(n_frags, 4, dtype=T_1.dtype)
        identity_q[:, 0] = 1.0

        if self.rotation_augmentation == "none":
            R_aug_q, q_1 = identity_q, identity_q.clone()
        else:
            aug_gen = self._make_generator(
                complex_idx, stream_offset=1, deterministic=self.deterministic_augmentation,
            )
            if self.rotation_augmentation == "ligand_uniform":
                R_aug_q = sample_uniform_quaternion(1, dtype=T_1.dtype, generator=aug_gen)
                R_aug_q = R_aug_q.expand(n_frags, -1).clone()
            else:
                R_aug_q = sample_uniform_quaternion(n_frags, dtype=T_1.dtype, generator=aug_gen)
            q_1 = quaternion_inverse(R_aug_q)

        # Mask single-atom fragments
        single_mask = frag_sizes <= 1
        q_1[single_mask] = identity_q[single_mask]

        # Augment local coords
        R_aug = quaternion_to_matrix(R_aug_q)
        local_pos = torch.einsum("nij,nj->ni", R_aug[frag_id_for_atoms], local_pos_orig)

        # --- Flow matching sampling ---
        prior_gen = self._make_generator(
            complex_idx, variant_idx=prior_variant_idx,
            stream_offset=2, deterministic=self.deterministic_prior,
        )
        time_gen = self._make_generator(
            complex_idx, variant_idx=time_variant_idx,
            stream_offset=3, deterministic=self.deterministic_time,
        )

        t = torch.rand(1, generator=time_gen, dtype=T_1.dtype).item()

        T_0, q_0 = sample_prior_poses(
            n_frags,
            pocket_center=torch.zeros(3, dtype=T_1.dtype),
            translation_sigma=self.translation_sigma,
            frag_sizes=frag_sizes,
            generator=prior_gen,
        )

        targets = compute_flow_matching_targets(T_0, q_0, T_1, q_1, t, frag_sizes=frag_sizes)

        # --- Update node coordinates for flow matching state ---
        # graph.pt has static crystal coords (pocket-centered).
        # We replace fragment node coords with T_t, and atom node coords with R_t@local+T_t.
        node_coords = graph["node_coords"].clone()

        # Fragment nodes: use T_t
        frag_slice = graph["lig_frag_slice"]  # [start, end]
        frag_start, frag_end = frag_slice[0].item(), frag_slice[1].item()
        node_coords[frag_start:frag_end] = targets["T_t"]

        # Ligand atom nodes: use R_t @ local_pos + T_t
        atom_slice = graph["lig_atom_slice"]
        atom_start = atom_slice[0].item()
        R_t = quaternion_to_matrix(targets["q_t"])  # [N_frag, 3, 3]
        atom_pos_t = (
            torch.einsum("nij,nj->ni", R_t[frag_id_for_atoms], local_pos)
            + targets["T_t"][frag_id_for_atoms]
        )
        # atom_slice covers all ligand atoms (real + dummy in graph.pt).
        # atom_pos_t is real-atom-only; dummy atoms stay at crystal coords.
        n_real_atoms = atom_pos_t.shape[0]
        node_coords[atom_start:atom_start + n_real_atoms] = atom_pos_t

        # --- Build output dict ---
        # Pass through all graph.pt tensors
        out: dict[str, Tensor] = {}
        for k, v in graph.items():
            if isinstance(v, Tensor):
                out[k] = v
        out["node_coords"] = node_coords

        # Flow matching state
        out["T_frag"] = targets["T_t"]
        out["q_frag"] = targets["q_t"]
        out["v_target"] = targets["v_t"]
        out["omega_target"] = targets["omega_t"]
        out["frag_sizes"] = frag_sizes
        out["T_target"] = T_1
        out["q_target"] = q_1
        out["t"] = torch.tensor([t], dtype=T_1.dtype)
        out["pdb_id"] = pdb_id

        # For atom-level velocity target: v_atom = v_frag + omega × r
        r = atom_pos_t - targets["T_t"][frag_id_for_atoms]
        v_atom = targets["v_t"][frag_id_for_atoms] + torch.cross(
            targets["omega_t"][frag_id_for_atoms], r, dim=-1,
        )
        out["atom_pos_t"] = atom_pos_t
        out["v_atom_target"] = v_atom
        out["local_pos"] = local_pos
        out["frag_id_for_atoms"] = frag_id_for_atoms

        return out


def unified_collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate a list of unified dataset samples into a batched dict.

    Concatenates node/edge tensors and adjusts indices (edge_index, slices)
    with appropriate offsets per sample.
    """
    out: dict[str, Tensor] = {}
    keys = [k for k in batch[0] if isinstance(batch[0][k], Tensor)]
    str_keys = [k for k in batch[0] if isinstance(batch[0][k], str)]

    # Categorize keys
    node_keys = [k for k in keys if k.startswith("node_") or k == "node_coords"]
    edge_keys = [k for k in keys if k.startswith("edge_")]
    frag_keys = ["T_frag", "q_frag", "v_target", "omega_target", "frag_sizes",
                 "T_target", "q_target"]
    atom_keys = ["atom_pos_t", "v_atom_target", "local_pos", "frag_id_for_atoms"]
    scalar_keys = ["t"]
    count_keys = [k for k in keys if k.startswith("num_")]
    slice_keys = [k for k in keys if k.endswith("_slice")]

    # Node tensors: concatenate along dim 0
    for k in node_keys:
        out[k] = torch.cat([b[k] for b in batch], dim=0)

    # Edge tensors: concatenate, offset edge_index
    node_offsets = [0]
    for b in batch[:-1]:
        node_offsets.append(node_offsets[-1] + b["num_nodes"].item())

    if "edge_index" in keys:
        edge_indices = []
        for i, b in enumerate(batch):
            edge_indices.append(b["edge_index"] + node_offsets[i])
        out["edge_index"] = torch.cat(edge_indices, dim=1)

    for k in edge_keys:
        if k == "edge_index":
            continue
        out[k] = torch.cat([b[k] for b in batch], dim=0)

    # Fragment tensors: concatenate, offset frag_id_for_atoms
    frag_offsets = [0]
    for b in batch[:-1]:
        frag_offsets.append(frag_offsets[-1] + b["num_lig_frag"].item())

    for k in frag_keys:
        if k in keys:
            out[k] = torch.cat([b[k] for b in batch], dim=0)

    for k in atom_keys:
        if k not in keys:
            continue
        if k == "frag_id_for_atoms":
            parts = []
            for i, b in enumerate(batch):
                parts.append(b[k] + frag_offsets[i])
            out[k] = torch.cat(parts, dim=0)
        else:
            out[k] = torch.cat([b[k] for b in batch], dim=0)

    # Scalar: stack
    for k in scalar_keys:
        if k in keys:
            out[k] = torch.stack([b[k] for b in batch], dim=0)

    # Counts: stack
    for k in count_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)

    # Slices: offset and stack
    for k in slice_keys:
        parts = []
        for i, b in enumerate(batch):
            parts.append(b[k] + node_offsets[i])
        out[k] = torch.stack(parts, dim=0)

    # Batch index for nodes (which sample each node belongs to)
    batch_idx = []
    for i, b in enumerate(batch):
        batch_idx.append(torch.full((b["num_nodes"].item(),), i, dtype=torch.long))
    out["batch"] = torch.cat(batch_idx, dim=0)

    # Batch index for fragments
    frag_batch_idx = []
    for i, b in enumerate(batch):
        frag_batch_idx.append(torch.full((b["num_lig_frag"].item(),), i, dtype=torch.long))
    out["frag_batch"] = torch.cat(frag_batch_idx, dim=0)

    # String keys
    for k in str_keys:
        out[k] = [b[k] for b in batch]

    return out


__all__ = ["UnifiedDataset", "unified_collate"]
