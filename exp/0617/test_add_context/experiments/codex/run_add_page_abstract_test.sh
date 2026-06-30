#!/usr/bin/env bash
# add_page_abstract on ALL non-N/A rows (val_nonNA, 398), codex backend; prints the Stage 3 score.
# Inputs: --add-page-abstract --no-add-context  (<data-prompt> + <page-abstract>)
# Output: preds/codex/add_page_abstract_nonNA_codex.csv  (does NOT collide with the _100 run)
#   ./run_add_page_abstract_all.sh          # codex, CONC=4 (throttle-safe)
#   LIMIT=4 ./run_add_page_abstract_all.sh  # quick wiring check
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/../.." && pwd)"
STAGE3="$EXP/stage3"
REPO="$(cd "$EXP/../../../.." && pwd)"
PY="${PY:-/workspace/esg_contest/.venv/bin/python}"; [ -x "$PY" ] || PY=python3

NAME=add_page_abstract
BACKEND=codex
CONC="${CONC:-8}"                          # lower to 2-4 if codex throttles/stalls
TIMEOUT="${TIMEOUT:-300}"
MODEL="${MODEL:-gpt-5.5}"
LIMIT="${LIMIT:-}"
PROMPT="$EXP/prompts/codex/add-page-abstract.txt"
DATA="${DATA:-$EXP/data/val_nonNA.jsonl}"
BENCH="${BENCH:-$EXP/data/val_nonNA.json}"
OUT="$EXP/preds/$BACKEND/${NAME}_nonNA_${BACKEND}.csv"
LOG="$HERE/logs/${NAME}_nonNA_${BACKEND}.log"
mkdir -p "$EXP/preds/$BACKEND" "$HERE/logs"

INPUTS=(--add-page-abstract --no-add-context)

extra=(--model "$MODEL")

echo "[$(date '+%H:%M:%S')] $NAME/$BACKEND  prompt=$(basename "$PROMPT")  data=$(basename "$DATA")  conc=$CONC  ${LIMIT:+limit=$LIMIT}"
PYTHONUNBUFFERED=1 "$PY" "$STAGE3/core/vlm_pred.py" \
  --backend "$BACKEND" --concurrency "$CONC" --timeout "$TIMEOUT" \
  --prompt-path "$PROMPT" --data "$DATA" \
  "${INPUTS[@]}" ${LIMIT:+--limit "$LIMIT"} "${extra[@]}" \
  --output "$OUT" 2>&1 | tee "$LOG"

echo "=== score ($NAME, $(basename "$DATA"), $BACKEND) ==="
"$PY" "$REPO/core/analysis/score_st3_full_coverage.py" --benchmark "$BENCH" --pred "$OUT"
