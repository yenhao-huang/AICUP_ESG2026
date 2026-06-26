#!/usr/bin/env bash
# Generate Stage 3 multitask train/val data from synthesis dataset.
#
# Run from anywhere:
#   bash scripts/data/get_multitask_model_for_stage3.sh
#
# Pipeline:
#   data/synthesis_data/stage3/vpesg_4k_train_1000_add_val.json
#     -> core/service/data/split_train_val_by_class_ignore_misleading.py
#     -> data/multitask_data/stage3/<dataset>/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ===== PARAMETER SPACE (override with env vars) ===============================
VAL_RATIO="${VAL_RATIO:-0.2}"
SEED="${SEED:-42}"
LABEL_KEY="${LABEL_KEY:-evidence_quality}"
IGNORE_LABELS="${IGNORE_LABELS:-Misleading}"
INPUT_DIR="${INPUT_DIR:-data/synthesis_data/stage3}"
OUTPUT_DIR="${OUTPUT_DIR:-data/multitask_data/stage3}"
INPUT_FILE="${INPUT_FILE:-vpesg_4k_train_1000_add_val.json}"
PYTHON="${PYTHON:-.venv/bin/python}"
SPLITTER="${SPLITTER:-core/service/data/split_train_val_by_class_ignore_misleading.py}"
# =============================================================================

if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

[ -d "$INPUT_DIR" ] || { echo "[error] missing input dir: $INPUT_DIR" >&2; exit 1; }
[ -f "$SPLITTER" ] || { echo "[error] missing splitter: $SPLITTER" >&2; exit 1; }

INPUT="$INPUT_DIR/$INPUT_FILE"

[ -f "$INPUT" ] || { echo "[error] missing input file: $INPUT" >&2; exit 1; }

echo "### stage3 multitask data"
echo "input=$INPUT output_dir=$OUTPUT_DIR splitter=$SPLITTER"
echo "seed=$SEED val_ratio=$VAL_RATIO label=$LABEL_KEY ignore_labels=$IGNORE_LABELS"

dataset="$(basename "$INPUT" .json)"
echo "=== [dataset=$dataset] input=$INPUT ==="

out="$OUTPUT_DIR/$dataset"
mkdir -p "$out"
echo "--- [dataset=$dataset] -> $out ---"

"$PYTHON" "$SPLITTER" \
  --input "$INPUT" \
  --train-out "$out/${dataset}.train.json" \
  --val-out "$out/${dataset}.val.json" \
  --label-key "$LABEL_KEY" \
  --ignore-labels "$IGNORE_LABELS" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED"

echo "### stage3 multitask data DONE"
