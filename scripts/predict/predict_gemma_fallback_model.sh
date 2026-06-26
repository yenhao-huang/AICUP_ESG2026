#!/usr/bin/env bash
# Predict Stage 1/2 directly with Gemma on every input row.
# No BERT, fallback, confidence gate, or cascade gate is applied in this wrapper.
#
# Run from anywhere:
#   bash scripts/predict/predict_gemma_fallback_model.sh
#
# Modes:
#   MODE=submit  uses submission ensemble checkpoints + submission Gemma adapter (default)
#   MODE=local   uses locally trained ensemble checkpoints + local Gemma adapter
#
# Stages:
#   STAGE=all|stage1|stage2

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ===== PARAMETER SPACE (override with env vars) ===============================
MODE="${MODE:-submit}"
STAGE="${STAGE:-all}"
DATA="${DATA:-data/raw_data/vpesg4k_test_2000.json}"
PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-auto}"
GEMMA_BASE="${GEMMA_BASE:-models/gemma/base/unsloth-gemma-4-12b}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DRY_RUN="${DRY_RUN:-0}"
# =============================================================================

case "$MODE" in
  submit)
    DEFAULT_GEMMA_ADAPTER="models/submission/st12_fallback/gemma4_st12_mix"
    ;;
  local)
    DEFAULT_GEMMA_ADAPTER="models/gemma/gemma4_st12_mix"
    ;;
  *)
    echo "[error] MODE must be one of: submit, local (got: $MODE)" >&2
    exit 1
    ;;
esac

case "$STAGE" in
  all|stage1|stage2) ;;
  *)
    echo "[error] STAGE must be one of: all, stage1, stage2 (got: $STAGE)" >&2
    exit 1
    ;;
esac

GEMMA_ADAPTER="${GEMMA_ADAPTER:-$DEFAULT_GEMMA_ADAPTER}"
OUT_ROOT="${OUT_ROOT:-results/predict/gemma/${MODE}}"
STAGE1_OUTPUT="${STAGE1_OUTPUT:-$OUT_ROOT/stage1/gemma.csv}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-$OUT_ROOT/stage2/gemma.csv}"

if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

[ -f "$DATA" ] || { echo "[error] missing input data: $DATA" >&2; exit 1; }
[ -d "$GEMMA_BASE" ] || { echo "[error] missing Gemma base: $GEMMA_BASE" >&2; exit 1; }
[ -d "$GEMMA_ADAPTER" ] || { echo "[error] missing Gemma adapter: $GEMMA_ADAPTER" >&2; exit 1; }

run_cmd() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

echo "### predict gemma direct"
echo "mode=$MODE stage=$STAGE"
echo "data=$DATA"
echo "gemma_base=$GEMMA_BASE"
echo "gemma_adapter=$GEMMA_ADAPTER"
echo "out_root=$OUT_ROOT"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "stage1" ]; then
  mkdir -p "$(dirname "$STAGE1_OUTPUT")"
  cmd=(
    "$PYTHON" core/service/predict/stage1/pred_by_gemma.py
    --data "$DATA"
    --output "$STAGE1_OUTPUT"
    --gemma-base "$GEMMA_BASE"
    --gemma-adapter "$GEMMA_ADAPTER"
    --device "$DEVICE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --run-id "gemma_direct_${MODE}_stage1"
  )
  if [ -n "${LIMIT:-}" ]; then
    cmd+=(--limit "$LIMIT")
  fi
  echo "predictor=core/service/predict/stage1/pred_by_gemma.py"
  echo "stage1_output=$STAGE1_OUTPUT"
  run_cmd "${cmd[@]}"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "stage2" ]; then
  mkdir -p "$(dirname "$STAGE2_OUTPUT")"
  cmd=(
    "$PYTHON" core/service/predict/stage2/pred_by_gemma.py
    --data "$DATA"
    --output "$STAGE2_OUTPUT"
    --gemma-base "$GEMMA_BASE"
    --gemma-adapter "$GEMMA_ADAPTER"
    --device "$DEVICE"
    --max-new-tokens "$MAX_NEW_TOKENS"
  )
  if [ -n "${LIMIT:-}" ]; then
    cmd+=(--limit "$LIMIT")
  fi
  echo "predictor=core/service/predict/stage2/pred_by_gemma.py"
  echo "stage2_output=$STAGE2_OUTPUT"
  run_cmd "${cmd[@]}"
fi

echo "### predict gemma direct DONE"
