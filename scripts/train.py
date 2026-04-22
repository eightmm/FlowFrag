"""FlowFrag training entrypoint.

Usage:
    # 8-GPU DDP (standard):
    torchrun --standalone --nproc_per_node=8 scripts/train.py \\
        --config configs/train_v3_b200.yaml
    # Resume from checkpoint:
    torchrun --standalone --nproc_per_node=8 scripts/train.py \\
        --config configs/train_v3_b200.yaml \\
        --resume outputs/v3_b200/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Redirect wandb/netrc/cache to the workspace-local "home" shared with other
# projects in this account (flowfrag, mambafold). Real $HOME may be on
# network storage that gets wiped between sessions; the workspace path is
# persistent.
_WS_HOME = "/NHNHOME/WORKSPACE/0526040024_A/jaemin"
if os.path.isdir(_WS_HOME):
    os.environ.setdefault("NETRC", f"{_WS_HOME}/.netrc")
    os.environ.setdefault("WANDB_CONFIG_DIR", f"{_WS_HOME}/.config/wandb")
    os.environ.setdefault("WANDB_DIR", f"{_WS_HOME}/.cache/wandb")
    os.environ.setdefault("XDG_CACHE_HOME", f"{_WS_HOME}/.cache")
    os.environ.setdefault("XDG_CONFIG_HOME", f"{_WS_HOME}/.config")

# Reduce memory fragmentation from variable-size pocket batches
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# cuEquivariance kernel handles don't survive cross-device dispatch under DDP
# (CUDA_ERROR_INVALID_HANDLE on segmented_polynomial).  Mask each rank to its
# own GPU so local_rank == 0 inside the process. Must run before torch import.
_lr = os.environ.get("LOCAL_RANK")
if _lr is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _lr

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Enable TF32 for FP32 matmuls (safe 3-5% speedup on Ampere/Ada GPUs)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FlowFrag")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Validate required sections
    for section in ("model", "data", "training", "logging"):
        assert section in cfg, f"Missing config section: {section}"

    from src.training.trainer import Trainer

    trainer = Trainer(cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
