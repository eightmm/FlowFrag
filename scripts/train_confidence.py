#!/usr/bin/env python
"""Train multi-head confidence model on per-atom shards.

Uses Muon + AdamW hybrid optimizer and trapezoidal (WSD) LR schedule
matching the main-model training setup.  Step-based loop with periodic
validation + checkpoint snapshots.

Input shards  (scripts/gen_conf_train_data.py output):
    atom_scalar   [N, 512]
    atom_norms    [N, 192]
    atom_disp     [N]            per-atom Å displacement label
    atom_pose_ptr [P+1]          CSR pointers
    pose_pid      [P]            PDB id per pose
    pose_rmsd     [P]
    pose_v_mean/max/p95, pose_w_mean/max/p95 [P]
    pose_n_atoms, pose_n_frags   [P]

Per-complex ``atom_frag_id`` (local 0..n_frag-1) is reconstructed from
``data/processed/{pid}/ligand.pt``; per-fragment RMSD targets are derived
from per-atom displacements.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.models.confidence import ConfidenceHead
from src.training.confidence import (
    POSE_STATS_KEYS,
    build_batch, eval_val, fmt_val,
    load_all_shards, load_atom_frag_id_cache,
)
from src.training.losses import confidence_multitask_loss
from src.training.trainer import configure_optimizers, get_trapezoidal_scheduler


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path, model: ConfidenceHead,
    scalar_mu, scalar_sd, norms_mu, norms_sd, ps_mu, ps_sd,
    step: int, val_metrics: dict | None, args, lw: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "scalar_mu": scalar_mu, "scalar_sd": scalar_sd,
        "norms_mu": norms_mu, "norms_sd": norms_sd,
        "pose_stats_mu": ps_mu, "pose_stats_sd": ps_sd,
        "pose_stats_keys": POSE_STATS_KEYS,
        "scalar_dim": model.scalar_dim,
        "norms_dim": model.norms_dim,
        "pose_stats_dim": model.pose_stats_dim,
        "hidden": args.hidden,
        "trunk_depth": args.trunk_depth,
        "head_depth": args.head_depth,
        "dropout": args.dropout,
        "pool_mode": args.pool_mode,
        "n_pool_queries": args.n_pool_queries,
        "step": step,
        "val_metrics": val_metrics,
        "loss_weights": lw,
        "total_steps": args.total_steps,
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_shards_dir", default="data/conf_train",
                    help="Training shards (PDBbind train split)")
    ap.add_argument("--val_shards_dir", default="data/conf_val",
                    help="Validation shards (CASF-2016 val split, disjoint from train)")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--out_dir", default="outputs/conf_models",
                    help="Directory to save step checkpoints")
    ap.add_argument("--run_name", default="head_16k")

    # Model
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--trunk_depth", type=int, default=3)
    ap.add_argument("--head_depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--pool_mode", type=str, default="mean_max",
                    choices=("mean_max", "attention", "both"))
    ap.add_argument("--n_pool_queries", type=int, default=4)

    # Training
    ap.add_argument("--total_steps", type=int, default=20000)
    ap.add_argument("--batch_complexes", type=int, default=32)
    ap.add_argument("--val_every", type=int, default=1000,
                    help="Validate and save a checkpoint every N steps")
    ap.add_argument("--seed", type=int, default=42)

    # Optimizer (Muon + AdamW hybrid, matching main model defaults)
    ap.add_argument("--lr", type=float, default=3e-4, help="AdamW LR (1D params)")
    ap.add_argument("--muon_lr", type=float, default=0.02, help="Muon LR (2D+ params)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--cooldown_ratio", type=float, default=0.3)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # Loss weights
    ap.add_argument("--w_atom", type=float, default=0.3)
    ap.add_argument("--w_plddt", type=float, default=0.5)
    ap.add_argument("--w_frag", type=float, default=0.5)
    ap.add_argument("--w_frag_bad", type=float, default=0.3)
    ap.add_argument("--w_pose", type=float, default=1.0)
    ap.add_argument("--w_pose_prob", type=float, default=0.3)
    ap.add_argument("--w_rank", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # --- Data: load train + val shards separately, then concatenate --------
    print("[train]")
    data_tr = load_all_shards(Path(args.train_shards_dir))
    print("[val]")
    data_va = load_all_shards(Path(args.val_shards_dir))

    n_atoms_tr = data_tr["atom_scalar"].shape[0]
    # Concatenate atom-level arrays; offset val pose_ptr by train atom count.
    atom_scalar_all = np.concatenate([data_tr["atom_scalar"], data_va["atom_scalar"]])
    atom_norms_all = np.concatenate([data_tr["atom_norms"], data_va["atom_norms"]])
    atom_disp_all = np.concatenate([data_tr["atom_disp"], data_va["atom_disp"]])
    pose_pid_all = np.concatenate([data_tr["pose_pid"], data_va["pose_pid"]])
    pose_rmsd_all = np.concatenate([data_tr["pose_rmsd"], data_va["pose_rmsd"]])
    pose_stats_all = np.stack(
        [np.concatenate([data_tr[k], data_va[k]]) for k in POSE_STATS_KEYS], axis=1,
    ).astype(np.float32)
    atom_pose_ptr = np.concatenate([
        data_tr["atom_pose_ptr"][:-1],
        data_va["atom_pose_ptr"] + n_atoms_tr,
    ])

    # PIDs present in each split (train / val come from disjoint shard dirs)
    train_pid_set = set(data_tr["pose_pid"].tolist())
    val_pid_set = set(data_va["pose_pid"].tolist())

    all_pids_list = list(pose_pid_all)
    frag_id_by_pid = load_atom_frag_id_cache(all_pids_list, Path(args.processed_dir))

    pid_to_pose_indices: dict[str, np.ndarray] = {}
    for i, pid in enumerate(pose_pid_all):
        pid_to_pose_indices.setdefault(str(pid), []).append(i)
    for k in pid_to_pose_indices:
        pid_to_pose_indices[k] = np.asarray(pid_to_pose_indices[k], dtype=np.int64)

    available = set(pid_to_pose_indices.keys()) & set(frag_id_by_pid.keys())
    train_pids = np.asarray(sorted(train_pid_set & available))
    val_pids = np.asarray(sorted(val_pid_set & available))
    print(f"Split: {len(train_pids)} train / {len(val_pids)} val complexes "
          f"(intersected with ligand.pt availability)")

    # Feature z-score (train atoms only, random sample for speed)
    tr_atom_idx_pose = np.concatenate([pid_to_pose_indices[pid] for pid in train_pids])
    sample_poses = rng.choice(
        len(tr_atom_idx_pose), min(5000, len(tr_atom_idx_pose)), replace=False,
    )
    tr_sample_poses = tr_atom_idx_pose[sample_poses]
    tr_atom_mask_idx = np.concatenate([
        np.arange(atom_pose_ptr[p_idx], atom_pose_ptr[p_idx + 1], dtype=np.int64)
        for p_idx in tr_sample_poses
    ])
    scalar_mu = atom_scalar_all[tr_atom_mask_idx].mean(axis=0).astype(np.float32)
    scalar_sd = atom_scalar_all[tr_atom_mask_idx].std(axis=0).astype(np.float32) + 1e-6
    norms_mu = atom_norms_all[tr_atom_mask_idx].mean(axis=0).astype(np.float32)
    norms_sd = atom_norms_all[tr_atom_mask_idx].std(axis=0).astype(np.float32) + 1e-6

    atom_scalar_t = torch.from_numpy(
        ((atom_scalar_all - scalar_mu) / scalar_sd).astype(np.float32)
    ).to(device)
    atom_norms_t = torch.from_numpy(
        ((atom_norms_all - norms_mu) / norms_sd).astype(np.float32)
    ).to(device)
    atom_disp_t = torch.from_numpy(atom_disp_all.astype(np.float32)).to(device)
    pose_rmsd_t = torch.from_numpy(pose_rmsd_all.astype(np.float32)).to(device)
    ps_mu = pose_stats_all.mean(axis=0)
    ps_sd = pose_stats_all.std(axis=0) + 1e-6
    pose_stats_arr = (pose_stats_all - ps_mu) / ps_sd
    pose_stats_t = torch.from_numpy(pose_stats_arr).to(device)
    print(f"Loaded to GPU: scalar={atom_scalar_t.shape}, norms={atom_norms_t.shape}")

    # --- Model --------------------------------------------------------------
    model = ConfidenceHead(
        scalar_dim=atom_scalar_t.shape[1],
        norms_dim=atom_norms_t.shape[1],
        pose_stats_dim=pose_stats_t.shape[1],
        hidden=args.hidden, trunk_depth=args.trunk_depth,
        head_depth=args.head_depth, dropout=args.dropout,
        pool_mode=args.pool_mode, n_pool_queries=args.n_pool_queries,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ConfidenceHead, params={n_params:,}")

    # --- Muon + AdamW + trapezoidal LR --------------------------------------
    optimizers = configure_optimizers(
        model, lr=args.lr, muon_lr=args.muon_lr,
        weight_decay=args.weight_decay, use_muon=True,
    )
    schedulers = [
        get_trapezoidal_scheduler(
            opt, total_steps=args.total_steps,
            warmup_ratio=args.warmup_ratio, cooldown_ratio=args.cooldown_ratio,
        )
        for opt in optimizers
    ]
    print(f"Optimizers: {[type(o).__name__ for o in optimizers]}")
    print(f"LR schedule: trapezoidal  warmup={int(args.total_steps * args.warmup_ratio)}"
          f"  stable={int(args.total_steps * (1 - args.warmup_ratio - args.cooldown_ratio))}"
          f"  cooldown={int(args.total_steps * args.cooldown_ratio)}")

    lw = dict(
        w_atom=args.w_atom, w_plddt=args.w_plddt, w_frag=args.w_frag,
        w_frag_bad=args.w_frag_bad, w_pose=args.w_pose,
        w_pose_prob=args.w_pose_prob, w_rank=args.w_rank,
    )

    # --- Step-based train loop ----------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train = len(train_pids)
    running: dict[str, float] = {}
    running_n = 0
    ckpt_history: list[dict] = []
    t0 = time.time()

    step = 0
    epoch = 0
    while step < args.total_steps:
        epoch += 1
        perm = np.random.permutation(n_train)
        for b_start in range(0, n_train, args.batch_complexes):
            if step >= args.total_steps:
                break
            cx_batch_ids = train_pids[perm[b_start : b_start + args.batch_complexes]]

            model.train(True)
            batch = build_batch(
                cx_batch_ids, pid_to_pose_indices, atom_pose_ptr,
                atom_scalar_t, atom_norms_t, atom_disp_t,
                pose_rmsd_t, pose_stats_t, frag_id_by_pid, device,
            )
            out = model(
                batch["atom_scalar"], batch["atom_norms"],
                batch["atom_pose_ptr"], batch["atom_frag_id"],
                pose_stats=batch["pose_stats"],
            )
            losses = confidence_multitask_loss(
                out, batch["atom_disp"], batch["frag_rmsd"], batch["pose_rmsd"],
                pose_cx_id=batch["pose_cx_id"], **lw,
            )

            for opt in optimizers:
                opt.zero_grad()
            losses["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            for opt in optimizers:
                opt.step()
            for sch in schedulers:
                sch.step()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach() if torch.is_tensor(v) else v)
            running_n += 1
            step += 1

            # Validation + checkpoint snapshot (same cadence)
            if step % args.val_every == 0 or step == args.total_steps:
                v = eval_val(
                    model, val_pids, pid_to_pose_indices, atom_pose_ptr,
                    atom_scalar_t, atom_norms_t, atom_disp_t,
                    pose_rmsd_t, pose_stats_t, frag_id_by_pid, device,
                    batch_complexes=args.batch_complexes,
                )
                dt = time.time() - t0
                avg = {k: running[k] / max(running_n, 1) for k in running}
                lr_main = optimizers[-1].param_groups[0]["lr"]
                print(f"step {step:>6d}/{args.total_steps}  ep{epoch:>3d}  [{dt/60:.1f}min]  "
                      f"lr={lr_main:.4g}  tr_loss={avg.get('loss', 0):.4f}  "
                      f"atom={avg.get('loss_atom',0):.3f} plddt={avg.get('loss_plddt',0):.3f} "
                      f"frag={avg.get('loss_frag',0):.3f} pose={avg.get('loss_pose',0):.3f} "
                      f"rank={avg.get('loss_rank',0):.3f}  ||  val: {fmt_val(v)}")
                running = {}
                running_n = 0

                ckpt_path = out_dir / f"{args.run_name}_step{step:06d}.pt"
                save_checkpoint(
                    ckpt_path, model,
                    scalar_mu, scalar_sd, norms_mu, norms_sd, ps_mu, ps_sd,
                    step=step, val_metrics=v, args=args, lw=lw,
                )
                ckpt_history.append({
                    "step": step, "path": str(ckpt_path), "val": v,
                })
                print(f"  → checkpoint saved: {ckpt_path.name}")

    # --- Final summary ------------------------------------------------------
    print("\n=== Checkpoint summary (val) ===")
    for c in ckpt_history:
        v = c["val"]
        print(f"step {c['step']:>6d}  sel_mean={v['sel_mean']:.3f}  "
              f"<1={v['sel_lt1']:5.1f}%  <2={v['sel_lt2']:5.1f}%  <5={v['sel_lt5']:5.1f}%  "
              f"({Path(c['path']).name})")
    best = min(ckpt_history, key=lambda c: c["val"]["sel_mean"])
    print(f"\nBest: step {best['step']} mean={best['val']['sel_mean']:.3f} "
          f"→ {best['path']}")


if __name__ == "__main__":
    main()
