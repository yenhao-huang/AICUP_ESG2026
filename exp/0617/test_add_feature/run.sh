#!/usr/bin/env bash
# Method A (verifiability-aware ST3) — train vanilla + multitask, then compare to exp23.
# Checkpoints go under /models/ (CLAUDE.md storage rule); small JSON results stay here.
set -euo pipefail

ROOT="/workspace/esg_contest"
HERE="$ROOT/exp/integrated_stage_predictions/0617/test_add_feature"
PY="$ROOT/.venv/bin/python"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
AUX_LAMBDA="${AUX_LAMBDA:-0.5}"

TRAIN="$ROOT/data/raw_data/vpesg_4k_train_1000.json"
VAL="$ROOT/data/raw_data/vpesg4k_val_1000.json"
REF="/models/esg_contest/exp23_train_json_st2_st3_st4_large/best_st3.pt"
MDIR_V="/models/test_add_feature_0617/vanilla"
MDIR_M="/models/test_add_feature_0617/multitask"

cd "$ROOT"

echo "########## [1/3] aux-feature separation sanity ##########"
"$PY" "$HERE/verifiability_features.py" "$TRAIN"

echo "########## [2/3] train vanilla (control) ##########"
"$PY" "$HERE/train_st3_feature.py" --mode vanilla \
  --train "$TRAIN" --val "$VAL" --model-dir "$MDIR_V" \
  --seed "$SEED" --device "$DEVICE"

echo "########## [2/3] train multitask (Method A) ##########"
"$PY" "$HERE/train_st3_feature.py" --mode multitask \
  --train "$TRAIN" --val "$VAL" --model-dir "$MDIR_M" \
  --aux-lambda "$AUX_LAMBDA" --seed "$SEED" --device "$DEVICE"

echo "########## [3/3] compare vs exp23 ##########"
"$PY" "$HERE/compare_st3.py" \
  --val "$VAL" --ref "$REF" \
  --vanilla "$MDIR_V/best_st3.pt" --multitask "$MDIR_M/best_st3.pt" \
  --device "$DEVICE"

echo "########## done ##########"
