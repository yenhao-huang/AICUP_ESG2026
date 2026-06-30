#!/usr/bin/env bash
# add_context Stage 3 prediction over the 2000-row TEST set (vpesg4k_test_2000), codex backend.
# Inputs: --add-context --context-mode all  (<data-prompt> + <same-page OCR context>)
# CONTEXT: same-page OCR text is located live by ContextBuilder, which resolves doc_id via the row's
#          `pdf_url` (url2doc) then picks the page from offsets.jsonl (matched_page_no). The row MUST
#          carry pdf_url -- the page_abstract build file drops it, so use the RAW test2000 json here.
# Output : preds/codex/add_context_test2000_codex.csv
# CACHE  : vlm_pred.py appends each done row to <output>.csv.cache.jsonl; reruns skip cached ids,
#          so a killed run never re-spends. Delete that .cache.jsonl to force a clean full rerun.
# NOTE   : the test set has NO labels, so there is no scoring step (set BENCH=... to score a labeled file).
#   ./run_add_context_all.sh                 # codex, CONC=8 (lower if it throttles/stalls)
#   LIMIT=4 ./run_add_context_all.sh         # quick wiring check
#   DATA=/path/to/other_with_id_data.jsonl ./run_add_context_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D617="$(cd "$HERE/.." && pwd)"                       # exp/integrated_stage_predictions/0617
EXP="$(cd "$HERE/../test_add_context" && pwd)"       # holds stage3/ data/ prompts/
STAGE3="$EXP/stage3"
REPO="$(cd "$EXP/../../../.." && pwd)"
PY="${PY:-/workspace/esg_contest/.venv/bin/python}"; [ -x "$PY" ] || PY=python3

NAME=add_context
BACKEND=codex
CONC="${CONC:-8}"                          # lower to 2-4 if codex throttles/stalls
TIMEOUT="${TIMEOUT:-300}"
MODEL="${MODEL:-gpt-5.5}"
LIMIT="${LIMIT:-}"
PROMPT="$EXP/prompts/codex/add-context.txt"
# RAW 2000-row test json: carries pdf_url, which ContextBuilder needs to resolve doc_id -> same-page OCR.
# (Do NOT use the page_abstract build file here -- it has no pdf_url, so every row would be miss_no_doc.)
DATA="${DATA:-$REPO/data/raw_data/vpesg4k_test_2000.json}"
BENCH="${BENCH:-}"                         # test has no labels; set to a labeled .json to enable scoring
OUT="$EXP/preds/$BACKEND/${NAME}_test2000_${BACKEND}.csv"
LOG="$HERE/logs/${NAME}_test2000_${BACKEND}.log"
mkdir -p "$EXP/preds/$BACKEND" "$HERE/logs"

if [ ! -f "$DATA" ]; then
  echo "[error] DATA not found: $DATA" >&2
  echo "        預設用 RAW data/raw_data/vpesg4k_test_2000.json（含 pdf_url，ContextBuilder 解 doc_id 必需）," >&2
  echo "        或用 DATA=<其他帶 id+data+pdf_url 的 json/jsonl> 覆寫。" >&2
  exit 1
fi

INPUTS=(--add-context --context-mode all)

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
