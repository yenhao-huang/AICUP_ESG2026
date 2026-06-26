#!/usr/bin/env bash
# Predict Stage 3 labels with one multitask BERT checkpoint.
#
# Run from anywhere:
#   bash scripts/predict/predict_multitaskbert_for_stage3.sh
#
# Modes:
#   MODE=submit  reads submission checkpoint under models/submission/stage3 (default)
#   MODE=local   reads locally trained checkpoint under models/multitaskbert/stage3
#
# Defaults:
#   input:  data/raw_data/vpesg4k_test_2000.json
#   output: results/predict/stage3/multitaskbert/<mode>/prediction.csv

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ===== PARAMETER SPACE (override with env vars) ===============================
MODE="${MODE:-submit}"
DATA="${DATA:-data/raw_data/vpesg4k_test_2000.json}"
PREDICTOR="${PREDICTOR:-core/service/predict/stage3/pred_by_multitask.py}"
PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-hfl/chinese-roberta-wwm-ext-large}"
MAX_LEN="${MAX_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-auto}"
LIMIT="${LIMIT:-}"
NC_TAU="${NC_TAU:-}"
DRY_RUN="${DRY_RUN:-0}"
# =============================================================================

case "$MODE" in
  submit)
    DEFAULT_CKPT="models/submission/stage3/w0_2_0_3_0_5/best_multitask_st3.pt"
    DEFAULT_RUN_ID="multitaskbert_st3_submit"
    ;;
  local)
    DEFAULT_CKPT="models/multitaskbert/stage3/vpesg_4k_train_1000_add_val/mt_st123_w1_8_30_mlval_seed42_e10/best_multitask_st3.pt"
    DEFAULT_RUN_ID="multitaskbert_st3_local"
    ;;
  *)
    echo "[error] MODE must be one of: submit, local (got: $MODE)" >&2
    exit 1
    ;;
esac

FINETUNE_PATH="${FINETUNE_PATH:-${CKPT:-$DEFAULT_CKPT}}"
OUTPUT="${OUTPUT:-results/predict/stage3/multitaskbert/${MODE}/prediction.csv}"
RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"

if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

[ -f "$DATA" ] || { echo "[error] missing input data: $DATA" >&2; exit 1; }
[ -f "$PREDICTOR" ] || { echo "[error] missing predictor: $PREDICTOR" >&2; exit 1; }
[ -f "$FINETUNE_PATH" ] || { echo "[error] missing checkpoint: $FINETUNE_PATH" >&2; exit 1; }

mkdir -p "$(dirname "$OUTPUT")"

cmd=(
  "$PYTHON" "$PREDICTOR"
  --data "$DATA"
  --finetune-path "$FINETUNE_PATH"
  --output "$OUTPUT"
  --model "$MODEL"
  --max-len "$MAX_LEN"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
  --run-id "$RUN_ID"
)

if [ -n "$LIMIT" ]; then
  cmd+=(--limit "$LIMIT")
fi

if [ -n "$NC_TAU" ]; then
  cmd+=(--nc-tau "$NC_TAU")
fi

echo "### predict multitaskbert stage3"
echo "mode=$MODE"
echo "data=$DATA"
echo "predictor=$PREDICTOR"
echo "finetune_path=$FINETUNE_PATH"
echo "stage2_gate=disabled"
echo "output=$OUTPUT"
echo "model=$MODEL max_len=$MAX_LEN batch_size=$BATCH_SIZE device=$DEVICE"

if [ "$DRY_RUN" = "1" ]; then
  printf '[dry-run]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
