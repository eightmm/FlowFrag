#!/usr/bin/env bash
# v4 PLINDER end-to-end training pipeline.
#
# Three stages, each idempotent (resume-safe):
#   STAGE 1  Train the main flow-matching model on PLINDER 130k samples
#   STAGE 2  Generate confidence training data via multi-σ ODE rollout
#   STAGE 3  Train the confidence head on the rollout poses
#
# Run from the project root after `rsync`-ing the active tree to B200:
#     bash scripts/train_v4_runbook.sh
#
# To run only one stage, set STAGE in the env:
#     STAGE=2 bash scripts/train_v4_runbook.sh
#
set -euo pipefail

PROJ=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJ"
UV="${UV:-$HOME/.local/bin/uv}"
NPROC="${NPROC:-8}"

CONFIG=configs/train_v4_plinder.yaml
MAIN_CKPT_DIR=outputs/v4_plinder
MAIN_BEST=$MAIN_CKPT_DIR/best.pt

CONF_DATA=data/conf_train_v4
CONF_OUT=outputs/conf_v4
CONF_BEST=$CONF_OUT/confidence_best.pt

STAGE="${STAGE:-all}"
echo "=== FlowFrag v4 runbook | STAGE=$STAGE | NPROC=$NPROC ==="
date

# -----------------------------------------------------------------------------
# STAGE 1 — Main FM training (single GPU loop or torchrun multi-GPU)
# -----------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
    echo
    echo ">>> STAGE 1 — main FM training ($CONFIG)"
    if [[ -f "$MAIN_BEST" && "${SKIP_MAIN_IF_DONE:-1}" == "1" ]]; then
        echo "  $MAIN_BEST exists, skipping (set SKIP_MAIN_IF_DONE=0 to retrain)."
    else
        $UV run torchrun --standalone --nproc_per_node="$NPROC" \
            scripts/train.py --config "$CONFIG"
    fi
fi

# -----------------------------------------------------------------------------
# STAGE 2 — Multi-σ confidence pose generation
#   Default: σ ∈ {2, 3, 4, 5}, 5 poses each (n_samples=20), 10 ODE steps.
# -----------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
    echo
    echo ">>> STAGE 2 — confidence pose generation (multi-σ)"
    if [[ ! -f "$MAIN_BEST" ]]; then
        echo "  ERROR: $MAIN_BEST not found. Run STAGE 1 first." >&2
        exit 1
    fi
    mkdir -p "$CONF_DATA"

    # Train shards
    $UV run python scripts/gen_conf_train_data.py \
        --processed_dir data/plinder_processed \
        --split_json   data/splits/plinder_v4.json \
        --split_key    train \
        --config       "$CONFIG" \
        --checkpoint   "$MAIN_BEST" \
        --out_dir      "$CONF_DATA" \
        --n_samples    20 \
        --n_steps      10 \
        --sigma_list   "2,3,4,5" \
        --gamma        0.0

    # Val shards (separate dir)
    $UV run python scripts/gen_conf_train_data.py \
        --processed_dir data/plinder_processed \
        --split_json   data/splits/plinder_v4.json \
        --split_key    val \
        --config       "$CONFIG" \
        --checkpoint   "$MAIN_BEST" \
        --out_dir      "${CONF_DATA}_val" \
        --n_samples    20 \
        --n_steps      10 \
        --sigma_list   "2,3,4,5" \
        --gamma        0.0
fi

# -----------------------------------------------------------------------------
# STAGE 3 — Confidence head training
# -----------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "3" ]]; then
    echo
    echo ">>> STAGE 3 — confidence head training"
    if [[ ! -d "$CONF_DATA" ]]; then
        echo "  ERROR: $CONF_DATA missing. Run STAGE 2 first." >&2
        exit 1
    fi
    mkdir -p "$CONF_OUT"

    $UV run python scripts/train_confidence.py \
        --train_shards_dir "$CONF_DATA"         \
        --val_shards_dir   "${CONF_DATA}_val"   \
        --processed_dir    data/plinder_processed \
        --out_dir          "$CONF_OUT"          \
        --run_name         confidence_v4        \
        --pool_mode        attention            \
        --hidden           512                  \
        --trunk_depth      4                    \
        --head_depth       2                    \
        --total_steps      20000
fi

# -----------------------------------------------------------------------------
# Optional STAGE 4 — Smoke evaluation on PoseBusters / Astex (multi-σ)
# -----------------------------------------------------------------------------
if [[ "$STAGE" == "eval" ]]; then
    echo
    echo ">>> STAGE 4 — external benchmarks"
    PB_DIR=/mnt/data/PLI/PoseBusters/posebusters_benchmark_set
    AS_DIR=/mnt/data/PLI/Astex-diverse-set

    for DSET_DIR in "$PB_DIR" "$AS_DIR"; do
        $UV run python scripts/eval_benchmark.py \
            --data_dir "$DSET_DIR" \
            --checkpoint "$MAIN_BEST" \
            --config "$CONFIG" \
            --num_steps 25 \
            --num_samples 40 \
            --sigma_list "2,3,4,5" \
            --confidence_ckpt "$CONF_BEST" \
            --smiles_init
    done
fi

echo
echo "=== done at $(date) ==="
