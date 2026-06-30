#!/usr/bin/env bash
# Stage 3 Qwen reason/thinking run. This script writes separate reason-named
# logs and predictions, so it does not overwrite run_pred.sh artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"

EXP="$ROOT/exp/integrated_stage_predictions/0616/test_add_context/stage3"
DATA="${DATA:-$EXP/data/val_ctxhit.json}"
GATE="${GATE-$EXP/data/val_ctxhit.json}"          # Stage 2 gate; set GATE= or GATE=none for all rows
ENDPOINT="${ENDPOINT:-http://192.168.1.78:3132/v1/chat/completions}"
MODEL="${MODEL:-local-qwen}"
LIMIT="${LIMIT:-}"                                # e.g. LIMIT=20 for smoke
CONC="${CONC:-1}"
MAX_TOKENS="${MAX_TOKENS:-5120}"                   # reasoning needs more room than bare-label mode
TIMEOUT="${TIMEOUT:-180}"
RETRIES="${RETRIES:-4}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"           # set 0 to keep reason output filenames but disable thinking
RUN_ID="${RUN_ID:-reason}"

PRED="$EXP/preds"
LOG="$EXP/logs"
mkdir -p "$PRED" "$LOG"

thinking_args=()
if [[ "$ENABLE_THINKING" == "1" || "$ENABLE_THINKING" == "true" ]]; then
  thinking_args=(--enable-thinking)
fi

stage2_args=()
if [[ -n "$GATE" && "$GATE" != "none" ]]; then
  stage2_args=(--stage2-csv "$GATE" --stage2-gate-col evidence_status)
fi

run () { # <variant> <ctx-flag>
  local variant="$1"
  local ctx_flag="$2"
  local out="$PRED/st3_qwen_${RUN_ID}_${variant}.csv"
  local log="$LOG/pred_${RUN_ID}_${variant}.log"

  echo "[$(date '+%H:%M:%S')] ST3 qwen $RUN_ID/$variant ${LIMIT:+(limit=$LIMIT)} (conc=$CONC, max_tokens=$MAX_TOKENS, thinking=$ENABLE_THINKING)"
  PYTHONUNBUFFERED=1 "$PY" core/human/predict/stage3/pred_by_qwen.py \
    --data "$DATA" "${stage2_args[@]}" \
    --prompt-path configs/prompt/stage3/codex/clear_notclear_with_context.txt \
    --endpoint "$ENDPOINT" --model "$MODEL" \
    --concurrency "$CONC" --max-tokens "$MAX_TOKENS" --timeout "$TIMEOUT" --retries "$RETRIES" \
    ${LIMIT:+--limit "$LIMIT"} "${thinking_args[@]}" \
    "$ctx_flag" --context-budget 800 --context-window after_biased --cross-page \
    --output "$out" \
    2>&1 | tee "$log"
}

run ctx  "--add-context"
run ctrl "--no-add-context"

echo "[$(date '+%H:%M:%S')] done"
echo "preds:"
echo "  $PRED/st3_qwen_${RUN_ID}_ctx.csv"
echo "  $PRED/st3_qwen_${RUN_ID}_ctrl.csv"
echo "logs:"
echo "  $LOG/pred_${RUN_ID}_ctx.log"
echo "  $LOG/pred_${RUN_ID}_ctrl.log"
