#!/usr/bin/env bash
# Predict Stage 1 labels with the soft-vote ensemble checkpoints under models/.
#
# Run from anywhere:
#   bash scripts/predict/predict_ensemble_model_for_stage1.sh
#
# Modes:
#   MODE=submit  reads submission checkpoints under models/submission/ (default)
#   MODE=local   reads locally trained checkpoints under models/ensemble_models/
#
# Defaults:
#   input:  data/raw_data/vpesg4k_test_2000.json
#   output: results/predict/stage1/ensemble/<mode>/softvote.csv

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ===== PARAMETER SPACE (override with env vars) ===============================
MODE="${MODE:-submit}"
DATA="${DATA:-data/raw_data/vpesg4k_test_2000.json}"
PREDICTOR="${PREDICTOR:-core/service/predict/stage1/soft_vote.py}"
PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-hfl/chinese-roberta-wwm-ext-large}"
MAX_LEN="${MAX_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-auto}"
LIMIT="${LIMIT:-}"
MEMBERS_OUT="${MEMBERS_OUT:-}"
DRY_RUN="${DRY_RUN:-0}"
# =============================================================================

case "$MODE" in
  submit)
    DEFAULT_MODELS_DIR="models/submission/stage1"
    DEFAULT_CKPT_GLOB="${DEFAULT_MODELS_DIR}/*/best_st1.pt"
    DEFAULT_RUN_ID="softvote_st1_submit"
    ;;
  local)
    DEFAULT_MODELS_DIR="models/ensemble_models/stage1"
    DEFAULT_CKPT_GLOB="${DEFAULT_MODELS_DIR}/*/seed*/*/best_st1.pt"
    DEFAULT_RUN_ID="softvote_st1_local"
    ;;
  *)
    echo "[error] MODE must be one of: submit, local (got: $MODE)" >&2
    exit 1
    ;;
esac

MODELS_DIR="${MODELS_DIR:-$DEFAULT_MODELS_DIR}"
if [ -z "${CKPT_GLOB:-}" ]; then
  case "$MODE" in
    submit) CKPT_GLOB="$MODELS_DIR/*/best_st1.pt" ;;
    local) CKPT_GLOB="$MODELS_DIR/*/seed*/*/best_st1.pt" ;;
  esac
fi
OUTPUT="${OUTPUT:-results/predict/stage1/ensemble/${MODE}/softvote.csv}"
RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"

if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

[ -f "$DATA" ] || { echo "[error] missing input data: $DATA" >&2; exit 1; }
[ -f "$PREDICTOR" ] || { echo "[error] missing predictor: $PREDICTOR" >&2; exit 1; }

shopt -s nullglob
CKPTS=( $CKPT_GLOB )
shopt -u nullglob

[ "${#CKPTS[@]}" -gt 0 ] || {
  echo "[error] no checkpoints matched: $CKPT_GLOB" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"
if [ -n "$MEMBERS_OUT" ]; then
  mkdir -p "$(dirname "$MEMBERS_OUT")"
fi

cmd=(
  "$PYTHON" "$PREDICTOR"
  --data "$DATA"
  --ckpt-glob "$CKPT_GLOB"
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

if [ -n "$MEMBERS_OUT" ]; then
  cmd+=(--members-out "$MEMBERS_OUT")
fi

echo "### predict ensemble stage1"
echo "mode=$MODE"
echo "data=$DATA"
echo "predictor=$PREDICTOR"
echo "models_dir=$MODELS_DIR"
echo "ckpt_glob=$CKPT_GLOB"
echo "matched_checkpoints=${#CKPTS[@]}"
echo "output=$OUTPUT"
echo "model=$MODEL max_len=$MAX_LEN batch_size=$BATCH_SIZE device=$DEVICE"

if [ "$DRY_RUN" = "1" ]; then
  printf '[dry-run]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
