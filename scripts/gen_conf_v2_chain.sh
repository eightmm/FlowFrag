#!/bin/bash
# Multi-config confidence training data generation chain.
# Reads spec from data/conf_train_v2/manifest.json (configs array).
# Writes per-config shards under data/conf_{train,val}_v2/cfg_{tag}/, then
# symlinks them into ./merged/ with unique names for single-dir loaders.
#
# Resumable: each cfg dir gets a .done flag — re-running skips finished ones.
set -e
LOG=outputs/logs/conf_v2_gen.log
mkdir -p outputs/logs

# tag sigma gamma n_steps n_samples
configs=(
  "s3_g0_s25  3.0 0.0 25 5"
  "s5_g04_s12 5.0 0.4 12 5"
  "s5_g04_s25 5.0 0.4 25 5"
  "s7_g0_s25  7.0 0.0 25 5"
)

CKPT=weights/best.pt
CFG=configs/train_v3_b200.yaml

for split_key in train val; do
  out_root="data/conf_${split_key}_v2"
  for cfg in "${configs[@]}"; do
    read tag sigma gamma steps n <<< "$cfg"
    out_dir="${out_root}/cfg_${tag}"
    if [ -f "${out_dir}/.done" ]; then
      echo "[$(date +%H:%M:%S)] SKIP ${split_key}/${tag}" | tee -a $LOG
      continue
    fi
    mkdir -p "$out_dir"
    echo "[$(date +%H:%M:%S)] START ${split_key}/${tag} sigma=${sigma} gamma=${gamma} steps=${steps} n=${n}" | tee -a $LOG
    ~/.local/bin/uv run python scripts/gen_conf_train_data.py \
      --config $CFG --checkpoint $CKPT \
      --split_key $split_key \
      --out_dir "$out_dir" \
      --sigma "$sigma" --gamma "$gamma" \
      --n_steps "$steps" --n_samples "$n" \
      >> outputs/logs/conf_v2_${split_key}_${tag}.log 2>&1
    touch "${out_dir}/.done"
    echo "[$(date +%H:%M:%S)] DONE ${split_key}/${tag}" | tee -a $LOG
  done
done

# Merge — symlink shards with unique names so existing single-dir loader works.
for split_key in train val; do
  out_root="data/conf_${split_key}_v2"
  merge_dir="${out_root}/merged"
  mkdir -p "$merge_dir"
  for cfg in "${configs[@]}"; do
    tag=$(echo $cfg | awk '{print $1}')
    src="${out_root}/cfg_${tag}"
    for shard in "$src"/shard_*.npz; do
      [ -f "$shard" ] || continue
      base=$(basename "$shard")
      ln -sf "../cfg_${tag}/${base}" "${merge_dir}/${tag}_${base}"
    done
  done
  echo "[$(date +%H:%M:%S)] merged ${split_key} -> ${merge_dir}" | tee -a $LOG
done
echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a $LOG
