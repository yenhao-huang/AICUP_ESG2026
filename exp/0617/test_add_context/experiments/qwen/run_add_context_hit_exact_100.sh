#!/usr/bin/env bash
# add_context_hit_exact on the 100-row set, pinned to the qwen backend; prints the Stage 3 score.
# Inputs: --add-context --context-mode hit_exact_window_norm_window
#   CONC=2 ./run_add_context_hit_exact_100.sh     # codex: lower concurrency to avoid throttling
#   LIMIT=5 ./run_add_context_hit_exact_100.sh    # quick wiring check on 5 rows
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/../.." && pwd)"
STAGE3="$EXP/stage3"
REPO="$(cd "$EXP/../../../.." && pwd)"
PY="${PY:-/workspace/esg_contest/.venv/bin/python}"; [ -x "$PY" ] || PY=python3

NAME=add_context_hit_exact
BACKEND=qwen
CONC="${CONC:-8}"
TIMEOUT="${TIMEOUT:-300}"
MODEL="${MODEL:-gpt-5.5}"
LIMIT="${LIMIT:-}"
PROMPT="$EXP/prompts/codex/add-context.txt"
DATA="${DATA:-$EXP/data/val.100.jsonl}"
BENCH="${BENCH:-$EXP/data/val.100.json}"
OUT="$EXP/preds/$BACKEND/${NAME}_100_${BACKEND}.csv"
LOG="$HERE/logs/${NAME}_100_${BACKEND}.log"
mkdir -p "$EXP/preds/$BACKEND" "$HERE/logs"

INPUTS=(--add-context --context-mode hit_exact_window_norm_window)

extra=()
[ "$BACKEND" = codex ] && extra=(--model "$MODEL")
[ "$BACKEND" = qwen ]  && extra=(--logprobs)

echo "[$(date '+%H:%M:%S')] $NAME/$BACKEND  prompt=$(basename "$PROMPT")  conc=$CONC  ${LIMIT:+limit=$LIMIT}"
PYTHONUNBUFFERED=1 "$PY" "$STAGE3/core/vlm_pred.py" \
  --backend "$BACKEND" --concurrency "$CONC" --timeout "$TIMEOUT" \
  --prompt-path "$PROMPT" --data "$DATA" \
  "${INPUTS[@]}" ${LIMIT:+--limit "$LIMIT"} "${extra[@]}" \
  --output "$OUT" 2>&1 | tee "$LOG"

echo "=== score ($NAME, 100 rows, $BACKEND) ==="
"$PY" "$REPO/core/analysis/score_st3_full_coverage.py" --benchmark "$BENCH" --pred "$OUT"
