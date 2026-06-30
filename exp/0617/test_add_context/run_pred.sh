#!/usr/bin/env bash
# Stage 3 evidence_quality (Clear/Not Clear) via llama-server Qwen (local-qwen),
# add_context vs no-context control. Gated by GT evidence_status (isolates ST3).
# Prompt: configs/prompt/stage3/codex/clear_notclear_with_context.txt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

EXP="$ROOT/exp/integrated_stage_predictions/0616/test_add_context/stage3"
DATA="${DATA:-data/benchmarks/val.json}"
GATE="${GATE-data/benchmarks/val.json}"           # Stage 2 gate; set GATE= or GATE=none for all rows
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${MODEL:-local-qwen}"
LIMIT="${LIMIT:-}"                                 # e.g. LIMIT=100 to cap rows (smoke)
CONC="${CONC:-2}"                                  # concurrent qwen requests (llama-server has 4 slots)
PROMPT_PATH="${PROMPT_PATH:-configs/prompt/stage3/codex/clear_notclear_with_context.txt}"
PRED="$EXP/preds"
mkdir -p "$PRED"

run () { # <variant> <ctx-flag>
  echo "[$(date '+%H:%M:%S')] ST3 qwen $1 ${LIMIT:+(limit=$LIMIT)} (conc=$CONC)"
  STAGE2_ARGS=()
  if [[ -n "$GATE" && "$GATE" != "none" ]]; then
    STAGE2_ARGS=(--stage2-csv "$GATE" --stage2-gate-col evidence_status)
  fi
  PYTHONUNBUFFERED=1 "$PY" core/human/predict/stage3/pred_by_qwen.py \
    --data "$DATA" "${STAGE2_ARGS[@]}" \
    --prompt-path "$PROMPT_PATH" \
    --endpoint "$ENDPOINT" --model "$MODEL" --concurrency "$CONC" ${LIMIT:+--limit "$LIMIT"} \
    $2 --context-budget 800 --context-window after_biased --cross-page \
    --output "$PRED/st3_qwen_$1.csv" \
    2>&1 | tee "$EXP/logs/pred_$1.log"
}
mkdir -p "$EXP/logs"
run ctx  "--add-context"
run ctrl "--no-add-context"

echo "[$(date '+%H:%M:%S')] done. preds: $PRED/st3_qwen_{ctx,ctrl}.csv"
echo "score (4-class full coverage vs GT):"
echo "  $PY core/analysis/score_st3_full_coverage.py <gt.json> $PRED/st3_qwen_ctx.csv"
