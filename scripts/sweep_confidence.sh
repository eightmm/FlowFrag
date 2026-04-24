#!/bin/bash
# Quick model-capacity sweep for confidence head.
# Runs 4 configs sequentially, each for 8000 steps with val every 500.
# All share the same train/val shards + loss weights; only architecture varies.
set -e

SHARDS_TRAIN="data/conf_train"
SHARDS_VAL="data/conf_val"
OUT="outputs/conf_sweep"
STEPS=8000
VAL_EVERY=500

configs=(
    "base:256:3:2:0.2"
    "wide:512:3:2:0.2"
    "big:512:4:3:0.2"
    "xlarge:768:4:3:0.2"
)

mkdir -p "$OUT"

for cfg in "${configs[@]}"; do
    IFS=':' read -r name hidden trunk head dropout <<< "$cfg"
    echo "=========================================="
    echo "Config: $name  hidden=$hidden trunk=$trunk head=$head dropout=$dropout"
    echo "=========================================="
    ~/.local/bin/uv run python scripts/train_confidence.py \
        --train_shards_dir "$SHARDS_TRAIN" \
        --val_shards_dir "$SHARDS_VAL" \
        --out_dir "$OUT/$name" \
        --run_name "$name" \
        --hidden "$hidden" --trunk_depth "$trunk" --head_depth "$head" --dropout "$dropout" \
        --total_steps "$STEPS" --val_every "$VAL_EVERY" \
        --batch_complexes 32 \
        --lr 3e-4 --muon_lr 0.02 --weight_decay 0.01 \
        --warmup_ratio 0.1 --cooldown_ratio 0.3 \
        2>&1 | tee "$OUT/$name.log"
done

echo "=========================================="
echo "SWEEP DONE. Best checkpoint per config:"
echo "=========================================="
for cfg in "${configs[@]}"; do
    IFS=':' read -r name _ _ _ _ <<< "$cfg"
    grep "Best:" "$OUT/$name.log" | tail -1 | sed "s|^|[$name] |"
done
