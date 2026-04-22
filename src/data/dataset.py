"""Dataset: loads protein.pt + ligand.pt + meta.pt, crops pocket at runtime,
builds the unified graph on-the-fly, and samples flow matching state.

Returns a flat dict with all node/edge tensors from the graph plus flow
matching targets (T_frag, q_frag at time t, velocity targets).  Fragment
and atom node coordinates are updated to reflect the interpolated pose.
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
from src.preprocess.graph import build_static_complex_graph


def crop_to_pocket(
    prot_data: dict[str, Tensor],
    ref_coords: Tensor,
    cutoff: float = 8.0,
) -> dict[str, Tensor] | None:
    """Crop full protein tensors to pocket around reference coordinates.

    Residue-aware: if any atom of a residue is within *cutoff* of any
    reference point, the entire residue is kept.  Atom/bond/virtual-node
    indices are compacted.

    Args:
        prot_data: Full protein dict (from protein.pt / parse_pocket_atoms).
        ref_coords: [N_ref, 3] reference coordinates (e.g. crystal ligand atoms)
                    or [3] single point (e.g. predicted pocket center).
        cutoff: Distance cutoff in Angstroms.

    Returns cropped dict with the same keys, or None if nothing survives.
    """
    if ref_coords.ndim == 1:
        ref_coords = ref_coords.unsqueeze(0)

    patom_coords = prot_data["patom_coords"]
    patom_residue_id = prot_data["patom_residue_id"]

    # Atom-level distance filter
    dmat = torch.cdist(patom_coords, ref_coords)
    in_range = dmat.min(dim=1).values <= cutoff

    # Residue-aware: keep every atom whose residue has ≥1 atom in range
    active_res = patom_residue_id[in_range].unique()
    atom_mask = torch.isin(patom_residue_id, active_res)
    if not atom_mask.any():
        return None

    # Old → new atom index mapping
    old_indices = atom_mask.nonzero(as_tuple=True)[0]
    remap = torch.full((patom_coords.shape[0],), -1, dtype=torch.int64)
    remap[old_indices] = torch.arange(old_indices.shape[0], dtype=torch.int64)

    # Atom tensors
    new_residue_id = patom_residue_id[atom_mask]
    _, new_residue_id = torch.unique(new_residue_id, return_inverse=True)

    # Bond filter + remap
    pbond = prot_data["pbond_index"]
    if pbond.numel() > 0:
        keep = atom_mask[pbond[0]] & atom_mask[pbond[1]]
        new_pbond = torch.stack([remap[pbond[0][keep]], remap[pbond[1][keep]]])
    else:
        new_pbond = torch.zeros(2, 0, dtype=torch.int64)

    # Virtual-node filter (anchor atom must be kept)
    pres_mask = atom_mask[prot_data["pres_atom_index"]]

    return {
        "patom_coords": patom_coords[atom_mask],
        "patom_token": prot_data["patom_token"][atom_mask],
        "patom_residue_id": new_residue_id,
        "patom_is_backbone": prot_data["patom_is_backbone"][atom_mask],
        "patom_is_metal": prot_data["patom_is_metal"][atom_mask],
        "pbond_index": new_pbond,
        "pres_coords": prot_data["pres_coords"][pres_mask],
        "pres_residue_type": prot_data["pres_residue_type"][pres_mask],
        "pres_atom_index": remap[prot_data["pres_atom_index"][pres_mask]],
        "pres_is_pseudo": prot_data["pres_is_pseudo"][pres_mask],
    }


class UnifiedDataset(Dataset):
    """Dataset for the unified equivariant model.

    Loads per-complex protein.pt (full protein) and ligand.pt, crops the
    protein to a pocket at runtime, builds the unified graph, then applies
    flow matching sampling.

    Args:
        root: Directory with per-complex subdirs containing protein.pt, ligand.pt, meta.pt.
        split_file: JSON or text file with PDB IDs.
        split_key: Key for JSON split files.
        pocket_cutoff: Residue-aware distance cutoff for pocket cropping (Å).
        pocket_jitter_sigma: Gaussian jitter on pocket center (training augmentation).
        pocket_cutoff_noise: Uniform noise on cutoff (training augmentation).
        translation_sigma: Gaussian std for translation prior (Å).
        max_atoms / max_frags / min_atoms / min_protein_res: size filters.
        rotation_augmentation: "none", "ligand_uniform", or "per_fragment".
        deterministic: Fix all random sampling per sample (reproducible eval).
        seed: Base seed for deterministic mode.
    """

    def __init__(
        self,
        root: str | Path,
        split_file: str | Path | None = None,
        split_key: str = "train",
        pocket_cutoff: float = 8.0,
        pocket_jitter_sigma: float = 2.0,
        pocket_cutoff_noise: float = 2.0,
        translation_sigma: float = 10.0,
        max_atoms: int = 80,
        max_frags: int = 20,
        min_atoms: int = 5,
        min_protein_res: int = 50,
        rotation_augmentation: str = "none",
        deterministic: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.pocket_cutoff = pocket_cutoff
        self.pocket_jitter_sigma = pocket_jitter_sigma
        self.pocket_cutoff_noise = pocket_cutoff_noise
        self.translation_sigma = translation_sigma
        self.rotation_augmentation = rotation_augmentation
        self.deterministic = deterministic
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

        # Filter by size + require all files
        self.pdb_ids: list[str] = []
        for pid in pdb_ids:
            d = self.root / pid
            if not all((d / f).exists() for f in ("protein.pt", "ligand.pt", "meta.pt")):
                continue
            meta = torch.load(d / "meta.pt", weights_only=True)
            n_atom = meta["num_atom"].item()
            n_frag = meta["num_frag"].item()
            n_res = meta["num_res"].item()
            if n_atom < min_atoms or n_atom > max_atoms or n_frag > max_frags:
                continue
            if n_res < min_protein_res:
                continue
            self.pdb_ids.append(pid)

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def _make_generator(self, idx: int, stream_offset: int) -> torch.Generator | None:
        """Return a seeded generator for deterministic mode, else None (global RNG)."""
        if not self.deterministic:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.seed + idx * 9_973 + stream_offset * 104_729)
        return generator

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        pdb_id = self.pdb_ids[idx]
        data_dir = self.root / pdb_id

        prot_data = torch.load(data_dir / "protein.pt", weights_only=True)
        ligand = torch.load(data_dir / "ligand.pt", weights_only=True)
        meta = torch.load(data_dir / "meta.pt", weights_only=True)

        pocket_center = meta["pocket_center"]
        n_frags = meta["num_frag"].item()

        # --- Crop protein to pocket & build graph -------------------------
        # Always use pocket_center (protein residue centroid) as reference,
        # never ligand coords — matches inference where ligand position is unknown.
        ref_center = pocket_center.clone()
        cutoff = self.pocket_cutoff
        if self.pocket_jitter_sigma > 0:
            ref_center = ref_center + torch.randn(3) * self.pocket_jitter_sigma
        if self.pocket_cutoff_noise > 0:
            cutoff = cutoff + (torch.rand(1).item() * 2 - 0.5) * self.pocket_cutoff_noise * 2
            cutoff = max(cutoff, 4.0)
        cropped_prot = crop_to_pocket(prot_data, ref_center, cutoff=cutoff)
        if cropped_prot is None:
            cropped_prot = crop_to_pocket(prot_data, pocket_center, cutoff=self.pocket_cutoff + 5.0)
        assert cropped_prot is not None, f"No pocket residues found for {pdb_id}"
        graph = build_static_complex_graph(ligand, cropped_prot)

        # --- Fragment target (crystal pose, pocket-centered) --------------
        T_1 = ligand["frag_centers"] - pocket_center  # [N_frag, 3]

        # Fragment sizes + atom→fragment assignment
        frag_id_for_atoms = ligand["fragment_id"]
        frag_sizes = torch.zeros(n_frags, dtype=torch.int64)
        frag_sizes.scatter_add_(0, frag_id_for_atoms, torch.ones_like(frag_id_for_atoms))
        local_pos_orig = ligand["frag_local_coords"]

        # --- Rotation augmentation ----------------------------------------
        identity_q = torch.zeros(n_frags, 4, dtype=T_1.dtype)
        identity_q[:, 0] = 1.0

        if self.rotation_augmentation == "none":
            R_aug_q, q_1 = identity_q, identity_q.clone()
        else:
            aug_gen = self._make_generator(idx, stream_offset=1)
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

        # --- Flow matching sampling ---------------------------------------
        prior_gen = self._make_generator(idx, stream_offset=2)
        time_gen = self._make_generator(idx, stream_offset=3)

        # Logit-normal time sampling (SD3 / FlowDock): t = sigmoid(N(0, 1)).
        # Concentrates training on t ≈ 0.5 where flow matching is hardest.
        import math
        z = torch.randn(1, generator=time_gen, dtype=T_1.dtype).item()
        t = 1.0 / (1.0 + math.exp(-z))

        T_0, q_0 = sample_prior_poses(
            n_frags,
            pocket_center=torch.zeros(3, dtype=T_1.dtype),
            translation_sigma=self.translation_sigma,
            frag_sizes=frag_sizes,
            generator=prior_gen,
        )

        targets = compute_flow_matching_targets(T_0, q_0, T_1, q_1, t, frag_sizes=frag_sizes)

        # --- Update node coordinates for flow matching state --------------
        node_coords = graph["node_coords"].clone()

        # Pocket-center the static protein/ligand coords
        node_coords -= pocket_center

        # Fragment nodes: use T_t
        frag_slice = graph["lig_frag_slice"]
        frag_start, frag_end = frag_slice[0].item(), frag_slice[1].item()
        node_coords[frag_start:frag_end] = targets["T_t"]

        # Ligand atom nodes: use R_t @ local_pos + T_t
        atom_slice = graph["lig_atom_slice"]
        atom_start = atom_slice[0].item()
        R_t = quaternion_to_matrix(targets["q_t"])
        atom_pos_t = (
            torch.einsum("nij,nj->ni", R_t[frag_id_for_atoms], local_pos)
            + targets["T_t"][frag_id_for_atoms]
        )
        node_coords[atom_start:atom_start + atom_pos_t.shape[0]] = atom_pos_t

        # --- Build output dict --------------------------------------------
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

        # Atom-level velocity target: v_atom = v_frag + omega × r
        r = atom_pos_t - targets["T_t"][frag_id_for_atoms]
        v_atom = targets["v_t"][frag_id_for_atoms] + torch.cross(
            targets["omega_t"][frag_id_for_atoms], r, dim=-1,
        )
        out["atom_pos_t"] = atom_pos_t
        out["v_atom_target"] = v_atom
        out["local_pos"] = local_pos
        out["frag_id_for_atoms"] = frag_id_for_atoms

        # Cut bond indices for boundary alignment loss
        cut_bond = ligand.get("cut_bond_index")
        if cut_bond is not None and cut_bond.numel() > 0:
            out["cut_bond_src"] = cut_bond[0]
            out["cut_bond_dst"] = cut_bond[1]
        else:
            out["cut_bond_src"] = torch.zeros(0, dtype=torch.int64)
            out["cut_bond_dst"] = torch.zeros(0, dtype=torch.int64)

        return out


def unified_collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate a list of dataset samples into a batched dict.

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

    # Batch index for ligand atoms (frag_id_for_atoms has one entry per atom)
    atom_batch_idx = []
    atom_offsets = [0]
    for i, b in enumerate(batch):
        n_atoms = b["frag_id_for_atoms"].shape[0]
        atom_batch_idx.append(torch.full((n_atoms,), i, dtype=torch.long))
        if i < len(batch) - 1:
            atom_offsets.append(atom_offsets[-1] + n_atoms)
    out["atom_batch"] = torch.cat(atom_batch_idx, dim=0)

    # Cut bond indices for boundary alignment loss (offset by atom counts)
    if "cut_bond_src" in keys:
        cut_src_parts, cut_dst_parts = [], []
        for i, b in enumerate(batch):
            cut_src_parts.append(b["cut_bond_src"] + atom_offsets[i])
            cut_dst_parts.append(b["cut_bond_dst"] + atom_offsets[i])
        out["cut_bond_src"] = torch.cat(cut_src_parts, dim=0)
        out["cut_bond_dst"] = torch.cat(cut_dst_parts, dim=0)

    # String keys
    for k in str_keys:
        out[k] = [b[k] for b in batch]

    return out


__all__ = ["UnifiedDataset", "unified_collate"]
