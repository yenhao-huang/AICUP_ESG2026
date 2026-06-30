#!/usr/bin/env bash
# add_page_abstract Stage 3 prediction over the 2000-row TEST set (vpesg4k_test_2000), codex backend.
# Inputs: --add-page-abstract --no-add-context  (<data-prompt> + <page-abstract>)
# DATA   : test2000 rows that already carry a page_abstract field (qwen build by default, all 2000 filled).
# Output : preds/codex/add_page_abstract_test2000_codex.csv
# NOTE   : the test set has NO labels, so there is no scoring step (set BENCH=... to score a labeled file).
#   ./run_add_page_abstract_all.sh                 # codex, CONC=8 (lower if it throttles/stalls)
#   LIMIT=4 ./run_add_page_abstract_all.sh         # quick wiring check
#   DATA=/path/to/other_with_page_abstract.jsonl ./run_add_page_abstract_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D617="$(cd "$HERE/.." && pwd)"                       # exp/integrated_stage_predictions/0617
EXP="$(cd "$HERE/../test_add_context" && pwd)"       # holds stage3/ data/ prompts/
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
# 2000-row test rows with a page_abstract field (qwen page-abstract build, all 2000 filled).
DATA="${DATA:-$D617/add_page_abstract/test2000/val_with_page_abstract.jsonl}"
BENCH="${BENCH:-}"                         # test has no labels; set to a labeled .json to enable scoring
OUT="$EXP/preds/$BACKEND/${NAME}_test2000_${BACKEND}.csv"
LOG="$HERE/logs/${NAME}_test2000_${BACKEND}.log"
mkdir -p "$EXP/preds/$BACKEND" "$HERE/logs"

if [ ! -f "$DATA" ]; then
  echo "[error] DATA not found: $DATA" >&2
  echo "        先用 build_page_abstract.py 產出 test2000 的 val_with_page_abstract.jsonl（qwen，全 2000 筆）," >&2
  echo "        或用 DATA=<其他帶 page_abstract 的 jsonl> 覆寫。" >&2
  exit 1
fi

INPUTS=(--add-page-abstract --no-add-context)

extra=(--model "$MODEL")

echo "[$(date '+%H:%M:%S')] $NAME/$BACKEND  prompt=$(basename "$PROMPT")  data=$(basename "$DATA")  rows=$(wc -l < "$DATA")  conc=$CONC  ${LIMIT:+limit=$LIMIT}"
PYTHONUNBUFFERED=1 "$PY" "$STAGE3/core/vlm_pred.py" \
  --backend "$BACKEND" --concurrency "$CONC" --timeout "$TIMEOUT" \
  --prompt-path "$PROMPT" --data "$DATA" \
  "${INPUTS[@]}" ${LIMIT:+--limit "$LIMIT"} "${extra[@]}" \
  --output "$OUT" 2>&1 | tee "$LOG"

if [ -n "$BENCH" ] && [ -f "$BENCH" ]; then
  echo "=== score ($NAME, $(basename "$DATA"), $BACKEND) ==="
  "$PY" "$REPO/core/analysis/score_st3_full_coverage.py" --benchmark "$BENCH" --pred "$OUT"
else
  echo "=== no scoring (test set has no labels; pred -> $OUT) ==="
fi
