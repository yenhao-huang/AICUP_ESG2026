#!/usr/bin/env bash
# Generate Stage 2 ensemble train/val data from synthesis dataset.
#
# Run from anywhere:
#   bash scripts/data/get_ensemble_model_data_for_stage2.sh
#
# Pipeline:
#   data/synthesis_data/stage2/mix_a2_b3_add_val.json
#     -> core/service/data/split_train_val_by_class.py per seed
#     -> data/ensemble_data/stage2/<dataset>/seed<S>/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ===== PARAMETER SPACE (override with env vars) ===============================
SEEDS=(${SEEDS:-42 7 123 2024 31337})
VAL_RATIO="${VAL_RATIO:-0.2}"
LABEL_KEY="${LABEL_KEY:-evidence_status}"
INPUT_DIR="${INPUT_DIR:-data/synthesis_data/stage2}"
OUTPUT_DIR="${OUTPUT_DIR:-data/ensemble_data/stage2}"
INPUT_FILE="${INPUT_FILE:-mix_a2_b3_add_val.json}"
PYTHON="${PYTHON:-.venv/bin/python}"
SPLITTER="${SPLITTER:-core/service/data/split_train_val_by_class.py}"
# =============================================================================

if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

[ -d "$INPUT_DIR" ] || { echo "[error] missing input dir: $INPUT_DIR" >&2; exit 1; }
[ -f "$SPLITTER" ] || { echo "[error] missing splitter: $SPLITTER" >&2; exit 1; }

INPUT="$INPUT_DIR/$INPUT_FILE"

[ -f "$INPUT" ] || { echo "[error] missing input file: $INPUT" >&2; exit 1; }

echo "### stage2 ensemble data"
echo "input=$INPUT output_dir=$OUTPUT_DIR splitter=$SPLITTER"
echo "seeds=[${SEEDS[*]}] val_ratio=$VAL_RATIO label=$LABEL_KEY"

dataset="$(basename "$INPUT" .json)"
echo "=== [dataset=$dataset] input=$INPUT ==="

for seed in "${SEEDS[@]}"; do
  out="$OUTPUT_DIR/$dataset/seed${seed}"
  mkdir -p "$out"
  echo "--- [dataset=$dataset seed=$seed] -> $out ---"

  "$PYTHON" "$SPLITTER" \
    --input "$INPUT" \
    --train-out "$out/${dataset}.train.json" \
    --val-out "$out/${dataset}.val.json" \
    --label-key "$LABEL_KEY" \
    --val-ratio "$VAL_RATIO" \
    --seed "$seed"
done

echo "### stage2 ensemble data DONE"
