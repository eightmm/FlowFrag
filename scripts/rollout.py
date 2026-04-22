"""Standalone rollout evaluation on a checkpoint.

Loads a saved checkpoint and runs `_validate_rollout` on the val split,
optionally sharded across ranks via torchrun. Use this to:

- Evaluate the final (S=max_steps) checkpoint after training (training's built-in
  rollout already fires at the last step, but running this lets you vary
  num_steps / max_samples without retraining).
- Compare intermediate `rollout_step*.pt` checkpoints against each other.
- Sanity-check `best.pt` on the full val set before deploying.

Usage:
    # Full val set, 25 ODE steps, 8 GPU DDP (fastest):
    torchrun --standalone --nproc_per_node=8 scripts/rollout.py \\
        --config configs/train_v3_b200.yaml \\
        --checkpoint outputs/v3_b200/checkpoints/best.pt

    # Single-GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/rollout.py \\
        --config configs/train_v3_b200.yaml \\
        --checkpoint outputs/v3_b200/checkpoints/latest.pt \\
        --num_steps 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Workspace home redirection (mirrors scripts/train.py).
_WS_HOME = "/NHNHOME/WORKSPACE/0526040024_A/jaemin"
if os.path.isdir(_WS_HOME):
    os.environ.setdefault("NETRC", f"{_WS_HOME}/.netrc")
    os.environ.setdefault("WANDB_CONFIG_DIR", f"{_WS_HOME}/.config/wandb")
    os.environ.setdefault("WANDB_DIR", f"{_WS_HOME}/.cache/wandb")
    os.environ.setdefault("XDG_CACHE_HOME", f"{_WS_HOME}/.cache")
    os.environ.setdefault("XDG_CONFIG_HOME", f"{_WS_HOME}/.config")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# cuEquivariance DDP workaround — mask CUDA_VISIBLE_DEVICES per rank before torch import.
_lr = os.environ.get("LOCAL_RANK")
if _lr is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _lr

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main() -> None:
    ap = argparse.ArgumentParser(description="Rollout evaluation of a FlowFrag checkpoint")
    ap.add_argument("--config", required=True, help="Training YAML config (for model/data setup)")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint .pt to evaluate")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = full val set")
    ap.add_argument("--num_steps", type=int, default=25, help="ODE integration steps")
    ap.add_argument("--time_schedule", type=str, default=None,
                    help="Override rollout_time_schedule (uniform / late / early)")
    ap.add_argument("--schedule_power", type=float, default=None,
                    help="Override rollout_schedule_power (for late/early schedule)")
    ap.add_argument("--output", type=str, default=None, help="JSON output path (optional)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override rollout knobs
    cfg["logging"]["rollout_max_samples"] = args.max_samples
    cfg["logging"]["rollout_steps"] = args.num_steps
    if args.time_schedule is not None:
        cfg["logging"]["rollout_time_schedule"] = args.time_schedule
    if args.schedule_power is not None:
        cfg["logging"]["rollout_schedule_power"] = args.schedule_power
    cfg["logging"]["use_wandb"] = False

    from src.training.trainer import Trainer, cleanup_ddp

    trainer = Trainer(cfg)
    trainer.load_checkpoint(args.checkpoint)

    metrics = trainer._validate_rollout(epoch=0)

    if trainer.is_main:
        print("\n==== Rollout metrics ====")
        for k, v in sorted(metrics.items()):
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
                        "num_steps": args.num_steps,
                        "max_samples": args.max_samples,
                        "metrics": metrics,
                    },
                    f, indent=2,
                )
            print(f"\nSaved to {out_path}")

    cleanup_ddp(trainer.world_size)


if __name__ == "__main__":
    main()
