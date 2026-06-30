#!/usr/bin/env bash
# Multi-seed paired check: is the Method-A effect (multitask - vanilla) real or seed noise?
# Fixes data/recipe/aux_lambda; varies ONLY --seed. Trains vanilla & multitask per seed,
# records val metrics to results/seeds/, deletes the 1.3GB checkpoint after (disk is ~97% full).
set -euo pipefail

ROOT="/workspace/esg_contest"
HERE="$ROOT/exp/integrated_stage_predictions/0617/test_add_feature"
PY="$ROOT/.venv/bin/python"
DEVICE="${DEVICE:-cuda:0}"
AUX_LAMBDA="${AUX_LAMBDA:-0.5}"
SEEDS="${SEEDS:-7 13 42 123 2024}"

TRAIN="$ROOT/data/raw_data/vpesg_4k_train_1000.json"
VAL="$ROOT/data/raw_data/vpesg4k_val_1000.json"
OUTDIR="$HERE/results/seeds"
TMP="/models/test_add_feature_0617/_seed_tmp"
mkdir -p "$OUTDIR"
cd "$ROOT"

for seed in $SEEDS; do
  for mode in vanilla multitask; do
    out="$OUTDIR/train_${mode}_s${seed}.json"
    if [[ -f "$out" ]]; then echo "[skip] $out exists"; continue; fi
    mdir="$TMP/${mode}_s${seed}"
    echo "########## seed=$seed mode=$mode ##########"
    "$PY" "$HERE/train_st3_feature.py" --mode "$mode" \
      --train "$TRAIN" --val "$VAL" --model-dir "$mdir" \
      --aux-lambda "$AUX_LAMBDA" --seed "$seed" --device "$DEVICE" \
      --output "$out"
    rm -f "$mdir/best_st3.pt"   # keep only the metrics json (disk pressure)
  done
done
rm -rf "$TMP"

echo "########## summarize ##########"
"$PY" "$HERE/summarize_seeds.py" --dir "$OUTDIR"
echo "########## seeds-done ##########"
