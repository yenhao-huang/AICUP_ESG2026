#!/usr/bin/env bash
# Run Stage 4 Codex prediction on all rows in vpesg4k_test_2000.
# No Stage 1 gate: all rows are sent to Codex.
#
# Run from anywhere:
#   bash scripts/predict/pred_by_codex_for_stage4.sh
#
# Overridable env vars:
#   DATA            Input JSON/CSV rows (default: data/raw_data/vpesg4k_test_2000.json)
#   CODEX_MODEL     Codex model name (default: gpt-5.5)
#   CODEX_TIMEOUT   Per-row timeout in seconds (default: 300)
#   WORKERS         Concurrent Codex predictions (default: 8)
#   STAGE4_PROMPT   Prompt file path
#   START_FROM      Resume from this 1-based row index (default: 1)
#   RUN_ID          Run identifier (default: timestamp)
#   OUT_DIR         Output directory (default: results/predict/stage4/codex/all_rows/<RUN_ID>)
#   LIMIT           Optional row limit for smoke tests
#   DRY_RUN         Print command without running (default: 0)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ===== PARAMETER SPACE (override with env vars) ===============================
PYTHON="${PYTHON:-.venv/bin/python}"
PREDICTOR="${PREDICTOR:-core/service/predict/stage4/pred_by_codex.py}"
DATA="${DATA:-data/raw_data/vpesg4k_test_2000.json}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.5}"
CODEX_TIMEOUT="${CODEX_TIMEOUT:-300}"
WORKERS="${WORKERS:-8}"
STAGE4_PROMPT="${STAGE4_PROMPT:-configs/prompts/stage4/boundary_rules_v4.txt}"
START_FROM="${START_FROM:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-results/predict/stage4/codex/all_rows/${RUN_ID}}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"
# =============================================================================

if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

OUTPUT_CSV="$OUT_DIR/stage4_codex_predictions.csv"
RAW_OUTPUT_DIR="$OUT_DIR/raw"
TOKEN_USAGE_OUTPUT="$OUT_DIR/token_usage.jsonl"

[ -f "$DATA" ] || { echo "[error] missing input data: $DATA" >&2; exit 1; }
[ -f "$PREDICTOR" ] || { echo "[error] missing predictor: $PREDICTOR" >&2; exit 1; }
[ -f "$STAGE4_PROMPT" ] || { echo "[error] missing prompt: $STAGE4_PROMPT" >&2; exit 1; }

mkdir -p "$OUT_DIR" "$RAW_OUTPUT_DIR"

cmd=(
  "$PYTHON" "$PREDICTOR"
  --data "$DATA"
  --output "$OUTPUT_CSV"
  --model "$CODEX_MODEL"
  --prompt-path "$STAGE4_PROMPT"
  --timeout "$CODEX_TIMEOUT"
  --workers "$WORKERS"
  --start-from "$START_FROM"
  --run-id "${RUN_ID}_stage4"
  --raw-output-dir "$RAW_OUTPUT_DIR"
  --token-usage-output "$TOKEN_USAGE_OUTPUT"
)

if [ -n "$LIMIT" ]; then
  cmd+=(--limit "$LIMIT")
fi

echo "### predict stage4 by Codex: all rows"
echo "data=$DATA"
echo "predictor=$PREDICTOR"
echo "codex_model=$CODEX_MODEL timeout=$CODEX_TIMEOUT workers=$WORKERS"
echo "prompt=$STAGE4_PROMPT"
echo "start_from=$START_FROM limit=${LIMIT:-all}"
echo "output_csv=$OUTPUT_CSV"
echo "raw_output_dir=$RAW_OUTPUT_DIR"
echo "token_usage_output=$TOKEN_USAGE_OUTPUT"

if [ "$DRY_RUN" = "1" ]; then
  printf '[dry-run]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
