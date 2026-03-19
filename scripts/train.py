"""FlowFrag training entrypoint.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/overfit.yaml
    torchrun --nproc_per_node=1 scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --resume outputs/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
