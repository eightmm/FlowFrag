"""Training utilities for the per-atom confidence head.

Loads sharded per-atom features + labels (produced by
``scripts/gen_conf_train_data.py``) and provides complex-level batching
helpers used by the confidence trainer.

Shards are numpy npz files with keys:
  atom_scalar   [N_total, D_scalar]
  atom_norms    [N_total, D_norms]
  atom_disp     [N_total]          Å  (per-atom displacement label)
  atom_pose_ptr [P+1]              CSR pointer into atom arrays
  pose_pid      [P]                PDB id per pose
  pose_rmsd     [P]                sqrt(mean(atom_disp^2)) per pose
  pose_n_atoms, pose_n_frags       [P]
  pose_v_mean/max/p95, pose_w_*    [P]   scalar stats at t=1

Per-complex atom_frag_id (local 0..n_frag-1) is reconstructed from
``ligand.pt`` on demand — it is complex-invariant so we cache once per pid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.confidence import ConfidenceHead
from src.models.layers import scatter_mean


POSE_STATS_KEYS = [
    "pose_v_mean", "pose_v_max", "pose_v_p95",
    "pose_w_mean", "pose_w_max", "pose_w_p95",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_shards(shards_dir: Path) -> dict:
    """Concatenate all ``shard_*.npz`` under ``shards_dir`` with rebased pointers."""
    shards = sorted(shards_dir.glob("shard_*.npz"))
    assert shards, f"No shards in {shards_dir}"
    print(f"Loading {len(shards)} shards ...")

    parts: dict[str, list] = {
        "atom_scalar": [], "atom_norms": [], "atom_disp": [],
        "pose_pid": [], "pose_rmsd": [],
        "pose_n_atoms": [], "pose_n_frags": [],
    }
    for k in POSE_STATS_KEYS:
        parts[k] = []
    ptrs: list[np.ndarray] = []
    atom_offset = 0

    for sh in shards:
        d = np.load(sh)
        parts["atom_scalar"].append(d["atom_scalar"])
        parts["atom_norms"].append(d["atom_norms"])
        parts["atom_disp"].append(d["atom_disp"])
        parts["pose_pid"].append(d["pose_pid"])
        parts["pose_rmsd"].append(d["pose_rmsd"])
        parts["pose_n_atoms"].append(d["pose_n_atoms"])
        parts["pose_n_frags"].append(d["pose_n_frags"])
        for k in POSE_STATS_KEYS:
            parts[k].append(d[k])
        p = d["atom_pose_ptr"]
        ptrs.append(p[:-1] + atom_offset)
        atom_offset += int(p[-1])

    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["atom_pose_ptr"] = np.concatenate(
        ptrs + [np.asarray([atom_offset], dtype=np.int64)]
    )
    print(f"  poses: {len(out['pose_pid'])}  atoms: {atom_offset}  "
          f"unique complexes: {len(np.unique(out['pose_pid']))}")
    return out


def load_atom_frag_id_cache(
    pids: list[str], processed_dir: Path,
) -> dict[str, np.ndarray]:
    """Per-complex local ``atom_frag_id`` (0..n_frag-1) from ``ligand.pt``."""
    cache: dict[str, np.ndarray] = {}
    missing = 0
    for pid in set(pids):
        p = processed_dir / pid / "ligand.pt"
        if not p.exists():
            missing += 1
            continue
        lig = torch.load(p, map_location="cpu", weights_only=False)
        cache[pid] = lig["fragment_id"].numpy().astype(np.int32)
    print(f"atom_frag_id cache: {len(cache)}/{len(set(pids))} complexes "
          f"(missing: {missing})")
    return cache


# ---------------------------------------------------------------------------
# Splitting + batch assembly (complex-level)
# ---------------------------------------------------------------------------
def split_by_complex(
    pids: np.ndarray, val_frac: float, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_pids, val_pids).  Splits are by unique complex, not pose."""
    unique = np.unique(pids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(int(len(unique) * val_frac), 100)
    return unique[n_val:], unique[:n_val]


def build_batch(
    complex_ids: np.ndarray,
    pid_to_pose_indices: dict[str, np.ndarray],
    atom_pose_ptr: np.ndarray,
    atom_scalar_t: torch.Tensor,
    atom_norms_t: torch.Tensor,
    atom_disp_t: torch.Tensor,
    pose_rmsd_t: torch.Tensor,
    pose_stats_t: torch.Tensor,
    frag_id_by_pid: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Gather K complexes worth of poses into one flat batch (complex-level batching).

    Produces all tensors needed by :class:`ConfidenceHead` plus the matching
    per-atom / per-frag / per-pose labels (including per-fragment RMSD derived
    from per-atom displacements and ``atom_frag_id``).
    """
    pose_indices_list: list[np.ndarray] = []
    pose_cx_id_list: list[np.ndarray] = []
    pid_per_pose: list[str] = []
    for cx_id, pid in enumerate(complex_ids):
        pid_str = str(pid)
        idx = pid_to_pose_indices[pid_str]
        pose_indices_list.append(idx)
        pose_cx_id_list.append(np.full(len(idx), cx_id, dtype=np.int64))
        pid_per_pose.extend([pid_str] * len(idx))
    pose_indices = np.concatenate(pose_indices_list)
    pose_cx_id = np.concatenate(pose_cx_id_list)

    lo = atom_pose_ptr[pose_indices]
    hi = atom_pose_ptr[pose_indices + 1]
    sizes = hi - lo

    atom_idx = np.concatenate([np.arange(l, h, dtype=np.int64) for l, h in zip(lo, hi)])
    new_ptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    frag_ids_list: list[np.ndarray] = []
    frag_offset = 0
    for pid, sz in zip(pid_per_pose, sizes):
        local = frag_id_by_pid[pid]
        assert len(local) == sz, f"size mismatch for {pid}: {len(local)} vs {sz}"
        frag_ids_list.append(local.astype(np.int64) + frag_offset)
        n_frag = int(local.max()) + 1
        frag_offset += n_frag
    atom_frag_id = np.concatenate(frag_ids_list)
    total_frags = frag_offset

    atom_idx_t = torch.from_numpy(atom_idx).to(device)
    new_ptr_t = torch.from_numpy(new_ptr).to(device)
    atom_frag_id_t = torch.from_numpy(atom_frag_id).to(device)
    pose_cx_id_t = torch.from_numpy(pose_cx_id).to(device)
    pose_idx_t = torch.from_numpy(pose_indices).to(device)

    atom_scalar = atom_scalar_t.index_select(0, atom_idx_t)
    atom_norms = atom_norms_t.index_select(0, atom_idx_t)
    atom_disp = atom_disp_t.index_select(0, atom_idx_t)
    pose_rmsd = pose_rmsd_t.index_select(0, pose_idx_t)
    pose_stats = pose_stats_t.index_select(0, pose_idx_t)

    # Per-fragment RMSD target: sqrt(mean(atom_disp^2 per fragment))
    disp_sq = atom_disp.pow(2).unsqueeze(-1)
    frag_mean_sq = scatter_mean(disp_sq, atom_frag_id_t, total_frags).squeeze(-1)
    frag_rmsd = frag_mean_sq.sqrt()

    return {
        "atom_scalar": atom_scalar,
        "atom_norms": atom_norms,
        "atom_pose_ptr": new_ptr_t,
        "atom_frag_id": atom_frag_id_t,
        "atom_disp": atom_disp,
        "pose_stats": pose_stats,
        "pose_rmsd": pose_rmsd,
        "frag_rmsd": frag_rmsd,
        "pose_cx_id": pose_cx_id_t,
    }


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------
def eval_val(
    model: ConfidenceHead,
    val_pids: np.ndarray,
    pid_to_pose_indices: dict[str, np.ndarray],
    atom_pose_ptr: np.ndarray,
    atom_scalar_t: torch.Tensor,
    atom_norms_t: torch.Tensor,
    atom_disp_t: torch.Tensor,
    pose_rmsd_t: torch.Tensor,
    pose_stats_t: torch.Tensor,
    frag_id_by_pid: dict[str, np.ndarray],
    device: torch.device,
    batch_complexes: int = 32,
) -> dict:
    model.train(False)
    sel_rmsds: list[float] = []
    pose_pred_list, pose_true_list = [], []
    atom_pred_list, atom_true_list = [], []

    with torch.no_grad():
        for start in range(0, len(val_pids), batch_complexes):
            cx_batch = val_pids[start : start + batch_complexes]
            batch = build_batch(
                cx_batch, pid_to_pose_indices, atom_pose_ptr,
                atom_scalar_t, atom_norms_t, atom_disp_t,
                pose_rmsd_t, pose_stats_t, frag_id_by_pid, device,
            )
            out = model(
                batch["atom_scalar"], batch["atom_norms"],
                batch["atom_pose_ptr"], batch["atom_frag_id"],
                pose_stats=batch["pose_stats"],
            )
            pose_pred = out["pose_rmsd"].cpu().numpy()
            pose_true = batch["pose_rmsd"].cpu().numpy()
            atom_pred = out["atom_disp"].cpu().numpy()
            atom_true = batch["atom_disp"].cpu().numpy()
            pose_cx = batch["pose_cx_id"].cpu().numpy()

            pose_pred_list.append(pose_pred)
            pose_true_list.append(pose_true)
            atom_pred_list.append(atom_pred)
            atom_true_list.append(atom_true)

            for cx in np.unique(pose_cx):
                m = pose_cx == cx
                i = int(np.argmin(pose_pred[m]))
                sel_rmsds.append(pose_true[m][i])

    sel = np.asarray(sel_rmsds)
    pp = np.concatenate(pose_pred_list)
    pt = np.concatenate(pose_true_list)
    ap = np.concatenate(atom_pred_list)
    at = np.concatenate(atom_true_list)

    return {
        "n_cx": len(sel),
        "sel_mean": float(sel.mean()),
        "sel_med": float(np.median(sel)),
        "sel_lt1": float((sel < 1).mean() * 100),
        "sel_lt2": float((sel < 2).mean() * 100),
        "sel_lt5": float((sel < 5).mean() * 100),
        "pose_mse_log1p": float(((np.log1p(pp) - np.log1p(pt)) ** 2).mean()),
        "atom_mse_log1p": float(((np.log1p(ap) - np.log1p(at)) ** 2).mean()),
    }


def fmt_val(r: dict) -> str:
    return (f"mean={r['sel_mean']:.3f}  med={r['sel_med']:.3f}  "
            f"<1={r['sel_lt1']:5.1f}%  <2={r['sel_lt2']:5.1f}%  <5={r['sel_lt5']:5.1f}%  "
            f"atom_mse={r['atom_mse_log1p']:.4f}  pose_mse={r['pose_mse_log1p']:.4f}")


__all__ = [
    "POSE_STATS_KEYS",
    "load_all_shards", "load_atom_frag_id_cache",
    "split_by_complex", "build_batch", "eval_val", "fmt_val",
]
