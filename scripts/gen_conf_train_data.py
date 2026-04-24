#!/usr/bin/env python
"""Generate per-atom confidence training data from PDBbind train set.

For each PDBbind train complex (excluding those overlapping Astex):
  1. Load processed protein.pt + ligand.pt
  2. Build runtime graph (same as training)
  3. Sample N poses via sample_unified (stochastic γ for RMSD diversity)
  4. Run one extra model forward at t=1 with final poses
  5. Extract per-atom features + per-atom displacement labels
  6. Append flat per-atom arrays + per-pose scalars to output shard

Output shard format (npz per shard):
  atom_scalar  : [total_atoms, D_scalar]  raw l=0 hidden per atom
  atom_norms   : [total_atoms, D_norms]   per-atom non-scalar channel norms
  atom_disp    : [total_atoms]            per-atom Euclidean displacement (Å)
  atom_pose_ptr: [n_poses + 1]            CSR pointers (pose i → atoms[ptr[i]:ptr[i+1]])
  pose_pid     : [n_poses]                PDB ID per pose
  pose_rmsd    : [n_poses]                RMSD (redundant with atom_disp aggregate)
  pose_n_atoms : [n_poses]
  pose_n_frags : [n_poses]
  pose_v_mean/max/p95, pose_w_*           [n_poses] scalar stats
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.confidence_features import extract_per_atom_features
from src.inference.preprocess import build_inference_bundle, load_processed
from src.inference.sampler import sample_unified


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--split_json", default="data/splits/pdbbind2020.json")
    ap.add_argument("--split_key", default="train",
                    help="Which split to iterate: 'train' or 'val'")
    ap.add_argument("--astex_smiles", default="data/astex_smiles.json")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", default="data/conf_train",
                    help="Output directory for shard_*.npz files")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--n_steps", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=0.4)
    ap.add_argument("--shard_size", type=int, default=500, help="Complexes per shard")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(Path(args.split_json).read_text())
    assert args.split_key in split, (
        f"--split_key {args.split_key!r} not in {list(split.keys())}"
    )
    train_ids = list(split[args.split_key])
    astex_ids = set(json.loads(Path(args.astex_smiles).read_text()).keys())
    train_ids = [pid for pid in train_ids if pid not in astex_ids]
    print(f"{args.split_key} complexes (after Astex filter): {len(train_ids)}")

    if args.offset:
        train_ids = train_ids[args.offset:]
    if args.limit:
        train_ids = train_ids[: args.limit]
    print(f"Processing {len(train_ids)}, shard_size={args.shard_size} complexes")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = dict(cfg["model"]); model_cfg.pop("model_type", None)
    from src.models.unified import UnifiedFlowFrag
    model = UnifiedFlowFrag(**model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)
    print(f"Model loaded: step={ckpt.get('step','?')}")

    # Accumulators (per shard). Per-atom arrays concatenate flat; per-pose arrays grow.
    cur_complexes = 0
    shard_idx = 0
    buf_atom_scalar: list[np.ndarray] = []
    buf_atom_norms: list[np.ndarray] = []
    buf_atom_disp: list[np.ndarray] = []
    buf_atom_pose_ptr: list[int] = [0]
    buf_pose_pid: list[str] = []
    buf_pose_rmsd: list[float] = []
    buf_pose_n_atoms: list[int] = []
    buf_pose_n_frags: list[int] = []
    buf_pose_scalars: dict[str, list[float]] = {
        k: [] for k in ("v_mean", "v_max", "v_p95", "w_mean", "w_max", "w_p95")
    }
    failures = 0
    processed = 0
    t0 = time.time()

    def flush_shard():
        nonlocal cur_complexes, shard_idx, buf_atom_scalar, buf_atom_norms, buf_atom_disp
        nonlocal buf_atom_pose_ptr, buf_pose_pid, buf_pose_rmsd, buf_pose_n_atoms
        nonlocal buf_pose_n_frags, buf_pose_scalars
        if not buf_pose_pid: return
        out_path = out_dir / f"shard_{shard_idx:04d}.npz"
        pack = {
            "atom_scalar": np.concatenate(buf_atom_scalar).astype(np.float32),
            "atom_norms": np.concatenate(buf_atom_norms).astype(np.float32),
            "atom_disp": np.concatenate(buf_atom_disp).astype(np.float32),
            "atom_pose_ptr": np.asarray(buf_atom_pose_ptr, dtype=np.int64),
            "pose_pid": np.asarray(buf_pose_pid),
            "pose_rmsd": np.asarray(buf_pose_rmsd, dtype=np.float32),
            "pose_n_atoms": np.asarray(buf_pose_n_atoms, dtype=np.int32),
            "pose_n_frags": np.asarray(buf_pose_n_frags, dtype=np.int32),
        }
        for k, vlist in buf_pose_scalars.items():
            pack[f"pose_{k}"] = np.asarray(vlist, dtype=np.float32)
        np.savez_compressed(out_path, **pack)
        n_poses = len(buf_pose_pid)
        total_atoms = int(pack["atom_scalar"].shape[0])
        print(f"  [shard {shard_idx}] {cur_complexes} cxs, {n_poses} poses, "
              f"{total_atoms} atoms → {out_path.name}")
        # reset
        buf_atom_scalar = []
        buf_atom_norms = []
        buf_atom_disp = []
        buf_atom_pose_ptr = [0]
        buf_pose_pid = []
        buf_pose_rmsd = []
        buf_pose_n_atoms = []
        buf_pose_n_frags = []
        buf_pose_scalars = {k: [] for k in buf_pose_scalars}
        cur_complexes = 0
        shard_idx += 1

    for pi, pid in enumerate(train_ids):
        pdb_dir = Path(args.processed_dir) / pid
        loaded = load_processed(pdb_dir)
        if loaded is None:
            failures += 1; continue
        prot, lig, meta = loaded

        bundle = build_inference_bundle(prot, lig, meta)
        if bundle is None:
            failures += 1; continue
        graph, lig_data, inf_meta = bundle
        pocket_center = inf_meta["pocket_center"]
        crystal_centered = lig["atom_coords"] - pocket_center

        try:
            results = sample_unified(
                model, graph, lig_data, inf_meta,
                num_samples=args.n_samples, num_steps=args.n_steps,
                translation_sigma=args.sigma, time_schedule="late",
                schedule_power=3.0, device=device, stochastic_gamma=args.gamma,
            )
            raw_poses = [r["atom_pos_pred"].cpu() for r in results]
        except Exception as e:
            failures += 1
            if failures <= 5: print(f"  sample fail {pid}: {e}")
            continue

        try:
            feats = extract_per_atom_features(
                model, graph, lig_data, inf_meta, raw_poses,
                crystal_centered, device, t_eval=1.0,
            )
        except Exception as e:
            failures += 1
            if failures <= 5: print(f"  extract fail {pid}: {e}")
            continue

        B = args.n_samples
        n_atoms = feats["pose_n_atoms"]
        buf_atom_scalar.append(feats["atom_scalar"])
        buf_atom_norms.append(feats["atom_norms"])
        buf_atom_disp.append(feats["atom_disp"])
        for b in range(B):
            buf_atom_pose_ptr.append(buf_atom_pose_ptr[-1] + n_atoms)
            buf_pose_pid.append(pid)
            buf_pose_rmsd.append(float(feats["pose_rmsd"][b]))
            buf_pose_n_atoms.append(int(n_atoms))
            buf_pose_n_frags.append(int(feats["pose_n_frags"]))
            for k in buf_pose_scalars:
                buf_pose_scalars[k].append(float(feats[f"pose_{k}"][b]))

        processed += 1
        cur_complexes += 1

        if cur_complexes >= args.shard_size:
            flush_shard()

        if (pi + 1) % 50 == 0:
            dt = time.time() - t0
            rate = processed / dt
            remain = (len(train_ids) - pi - 1) / max(rate, 1e-9)
            print(f"  {pi+1}/{len(train_ids)}  elapsed={dt/60:.1f}min  "
                  f"rate={rate*60:.1f}/min  ETA={remain/3600:.1f}h  fails={failures}")

    flush_shard()
    elapsed = time.time() - t0
    print(f"\nDone: processed={processed} failures={failures} time={elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
