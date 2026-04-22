"""Multi-prior rollout: for each val sample, draw N independent priors and
report the oracle top-1 RMSD (min RMSD across N candidates per sample) — i.e.
assuming we pick the best-scoring pose with access to the crystal structure.

Standard docking benchmark metric. Compare against single-prior rollout from
`rollout.py` to see how much benefit sampling diversity gives.

Usage:
    torchrun --standalone --nproc_per_node=8 scripts/rollout_topN.py \\
        --config configs/train_v3_b200.yaml \\
        --checkpoint outputs/v3_b200/checkpoints/best.pt \\
        --num_priors 10 --num_steps 25 \\
        --output outputs/v3_b200/rollout_top10.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_WS_HOME = "/NHNHOME/WORKSPACE/0526040024_A/jaemin"
if os.path.isdir(_WS_HOME):
    os.environ.setdefault("NETRC", f"{_WS_HOME}/.netrc")
    os.environ.setdefault("WANDB_CONFIG_DIR", f"{_WS_HOME}/.config/wandb")
    os.environ.setdefault("WANDB_DIR", f"{_WS_HOME}/.cache/wandb")
    os.environ.setdefault("XDG_CACHE_HOME", f"{_WS_HOME}/.cache")
    os.environ.setdefault("XDG_CONFIG_HOME", f"{_WS_HOME}/.config")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_lr = os.environ.get("LOCAL_RANK")
if _lr is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _lr

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_priors", type=int, default=10, help="priors per sample")
    ap.add_argument("--num_steps", type=int, default=25, help="ODE steps")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = full val")
    ap.add_argument("--time_schedule", type=str, default=None)
    ap.add_argument("--schedule_power", type=float, default=None)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["logging"]["use_wandb"] = False
    cfg["logging"]["rollout_max_samples"] = args.max_samples
    cfg["logging"]["rollout_steps"] = args.num_steps
    if args.time_schedule is not None:
        cfg["logging"]["rollout_time_schedule"] = args.time_schedule
    if args.schedule_power is not None:
        cfg["logging"]["rollout_schedule_power"] = args.schedule_power

    from src.training.trainer import Trainer, cleanup_ddp
    from src.inference.metrics import ligand_rmsd, centroid_distance, frag_centroid_rmsd

    trainer = Trainer(cfg)
    trainer.load_checkpoint(args.checkpoint)

    raw_model = trainer._eval_model()
    raw_model.train(False)

    dcfg = cfg["data"]
    lcfg = cfg["logging"]
    num_steps = args.num_steps
    time_schedule = lcfg.get("rollout_time_schedule", "uniform")
    schedule_power = lcfg.get("rollout_schedule_power", 3.0)
    sigma = dcfg.get("prior_sigma", 5.0)
    seed_base = cfg["training"].get("seed", 42)

    val_ds = trainer.val_loader.dataset
    n_val = len(val_ds) if args.max_samples <= 0 else min(len(val_ds), args.max_samples)

    # rank-strided partition over samples; N priors done serially per sample
    best_rmsds, all_rmsds = [], []   # best_rmsds: [n_local], all_rmsds: [n_local, N]
    best_cent, best_frag = [], []

    for i in range(trainer.rank, n_val, trainer.world_size):
        data = val_ds[i]
        sample_rmsds = []
        sample_cent = []
        sample_frag = []
        for n in range(args.num_priors):
            seed = seed_base + i * args.num_priors + n
            T_pred, atom_pos_pred, true_pos = trainer._rollout_single_unified(
                raw_model, data,
                sigma=sigma,
                num_steps=num_steps,
                time_schedule=time_schedule,
                schedule_power=schedule_power,
                seed=seed,
            )
            T_target = data["T_target"].to(trainer.device)
            sample_rmsds.append(ligand_rmsd(atom_pos_pred, true_pos).item())
            sample_cent.append(centroid_distance(atom_pos_pred, true_pos).item())
            sample_frag.append(frag_centroid_rmsd(T_pred, T_target).item())
        # Oracle: pick min-RMSD pose for this sample
        best_idx = min(range(args.num_priors), key=lambda k: sample_rmsds[k])
        best_rmsds.append(sample_rmsds[best_idx])
        best_cent.append(sample_cent[best_idx])
        best_frag.append(sample_frag[best_idx])
        all_rmsds.append(sample_rmsds)

    # All-gather across ranks
    if trainer.world_size > 1:
        gathered: list = [None] * trainer.world_size  # type: ignore
        dist.all_gather_object(
            gathered,
            (best_rmsds, best_cent, best_frag, all_rmsds),
        )
        best_rmsds = [x for shard in gathered for x in shard[0]]
        best_cent = [x for shard in gathered for x in shard[1]]
        best_frag = [x for shard in gathered for x in shard[2]]
        all_rmsds = [x for shard in gathered for x in shard[3]]

    if trainer.is_main:
        best_t = torch.tensor(best_rmsds)
        all_t = torch.tensor(all_rmsds)       # [n_samples, N]
        first_t = all_t[:, 0]                 # first-prior only (baseline)
        mean_t = all_t.mean(dim=1)            # mean across priors
        worst_t = all_t.max(dim=1).values

        metrics = {
            "n_samples": int(best_t.numel()),
            "num_priors": args.num_priors,
            "num_steps": args.num_steps,
            # Oracle top-1
            "oracle/rmsd_median": best_t.median().item(),
            "oracle/rmsd_mean": best_t.mean().item(),
            "oracle/success_2A": (best_t < 2.0).float().mean().item(),
            "oracle/success_5A": (best_t < 5.0).float().mean().item(),
            "oracle/centroid_dist": torch.tensor(best_cent).mean().item(),
            "oracle/frag_rmsd": torch.tensor(best_frag).mean().item(),
            # Reference: single prior (first draw)
            "single/rmsd_median": first_t.median().item(),
            "single/success_2A": (first_t < 2.0).float().mean().item(),
            "single/success_5A": (first_t < 5.0).float().mean().item(),
            # Per-sample mean across N priors (stability indicator)
            "prior_mean/rmsd_median": mean_t.median().item(),
            # Worst-of-N (gives the sampling variance upper bound)
            "worst/rmsd_median": worst_t.median().item(),
        }

        print(f"\n==== Oracle top-1 of {args.num_priors} priors ({args.num_steps} steps) ====")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<28s} {v:.4f}")
            else:
                print(f"  {k:<28s} {v}")

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "checkpoint": args.checkpoint,
                        "metrics": metrics,
                        "per_sample_rmsds": all_rmsds,
                    },
                    f, indent=2,
                )
            print(f"\nSaved to {out_path}")

    cleanup_ddp(trainer.world_size)


if __name__ == "__main__":
    main()
