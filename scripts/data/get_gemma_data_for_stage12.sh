#!/usr/bin/env bash
# Generate Gemma Stage 1+2 train/val data from synthesis datasets.
#
# Run from anywhere:
#   bash scripts/data/get_gemma_data_for_stage12.sh
#
# Pipeline:
#   data/synthesis_data/stage12/*.json
#     -> core/service/data/split_by_evidence_status.py
#     -> data/gemma_data/stage12/<dataset>.train.json
#     -> data/gemma_data/stage12/<dataset>.val.json
#
# Override examples:
#   INPUT_FILE=vpesg4k_train_val_mix_2000.json \
#   OUTPUT_DIR=exp/eval_train_in_gemma4_st12/data \
#   bash scripts/data/get_gemma_data_for_stage12.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ===== PARAMETER SPACE (override with env vars) ===============================
INPUT_DIR="${INPUT_DIR:-data/synthesis_data/stage12}"
OUTPUT_DIR="${OUTPUT_DIR:-data/gemma_data/stage12}"
INPUT_FILE="${INPUT_FILE:-}"
VAL_RATIO="${VAL_RATIO:-0.2}"
KEY="${KEY:-evidence_status}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-.venv/bin/python}"
SPLITTER="${SPLITTER:-core/service/data/split_by_evidence_status.py}"
# =============================================================================

if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

[ -d "$INPUT_DIR" ] || { echo "[error] missing input dir: $INPUT_DIR" >&2; exit 1; }
[ -f "$SPLITTER" ] || { echo "[error] missing splitter: $SPLITTER" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"

inputs=()
if [ -n "$INPUT_FILE" ]; then
  inputs=("$INPUT_DIR/$INPUT_FILE")
else
  shopt -s nullglob
  inputs=("$INPUT_DIR"/*.json)
  shopt -u nullglob
fi

[ "${#inputs[@]}" -gt 0 ] || { echo "[error] no json inputs found in: $INPUT_DIR" >&2; exit 1; }

echo "### stage12 gemma data"
echo "input_dir=$INPUT_DIR output_dir=$OUTPUT_DIR splitter=$SPLITTER"
echo "key=$KEY val_ratio=$VAL_RATIO seed=$SEED"

for input in "${inputs[@]}"; do
  [ -f "$input" ] || { echo "[error] missing input file: $input" >&2; exit 1; }

  dataset="$(basename "$input" .json)"
  train_out="$OUTPUT_DIR/${dataset}.train.json"
  val_out="$OUTPUT_DIR/${dataset}.val.json"

  echo "=== [dataset=$dataset] input=$input ==="
  "$PYTHON" "$SPLITTER" \
    --input "$input" \
    --out-a "$train_out" \
    --out-b "$val_out" \
    --key "$KEY" \
    --val-ratio "$VAL_RATIO" \
    --seed "$SEED"
done

echo "### stage12 gemma data DONE"
