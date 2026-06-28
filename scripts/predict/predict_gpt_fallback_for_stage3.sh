#!/usr/bin/env bash
# add_context Stage 3 prediction over the 2000-row TEST set (vpesg4k_test_2000), codex backend.
# Inputs: --add-context --context-mode all  (<data-prompt> + <same-page OCR context>)
# CONTEXT: same-page OCR text is resolved by ContextBuilder, which maps the row's
#          `pdf_url` to doc_id and then picks the page from offsets.jsonl
#          (matched_page_no). The row MUST carry pdf_url, so use the RAW test2000
#          json here instead of page_abstract-derived files that drop pdf_url.
# Output : results/predict/stage3/gpt_fallback/codex/add_context_test2000_codex.csv
# CACHE  : vlm_pred.py appends each done row to <output>.cache.jsonl; reruns skip
#          cached ids, so a killed run never re-spends. Delete that .cache.jsonl
#          to force a clean full rerun.
# NOTE   : the test set has NO labels, so there is no scoring step (set BENCH=...
#          to score a labeled file).
#   bash scripts/predict/predict_gpt_fallback_for_stage3.sh       # WORKERS=8
#   WORKERS=4 bash scripts/predict/predict_gpt_fallback_for_stage3.sh
#   LIMIT=4 bash scripts/predict/predict_gpt_fallback_for_stage3.sh
#   DATA=/path/to/other_with_id_data_pdf_url.jsonl bash scripts/predict/predict_gpt_fallback_for_stage3.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-.venv/bin/python}"
if [ ! -x "$PY" ]; then
  PY="${PYTHON_FALLBACK:-python3}"
fi

NAME="add_context"
BACKEND="codex"
WORKERS="${WORKERS:-${CONC:-8}}"
TIMEOUT="${TIMEOUT:-300}"
MODEL="${MODEL:-gpt-5.5}"
LIMIT="${LIMIT:-}"
DATA="${DATA:-data/raw_data/vpesg4k_test_2000.json}"
BENCH="${BENCH:-}"
PROMPT="${PROMPT:-configs/prompts/stage3/add-context.txt}"
PREDICTOR="${PREDICTOR:-core/service/predict/stage3/vlm_pred.py}"
DOC_TABLE="${DOC_TABLE:-data/generated/raw_doc_table.jsonl}"
PAGE_TABLE="${PAGE_TABLE:-data/generated/raw_page_table.jsonl}"
OFFSETS_PATH="${OFFSETS_PATH:-data/generated/stage3_offsets.jsonl}"
OUT="${OUT:-results/predict/stage3/gpt_fallback/$BACKEND/${NAME}_test2000_${BACKEND}.csv}"
LOG="${LOG:-logs/predict/stage3/gpt_fallback/${NAME}_test2000_${BACKEND}.log}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

if [ ! -f "$DATA" ]; then
  echo "[error] DATA not found: $DATA" >&2
  echo "        預設用 RAW data/raw_data/vpesg4k_test_2000.json（含 pdf_url，ContextBuilder 解 doc_id 必需），" >&2
  echo "        或用 DATA=<其他帶 id+data+pdf_url 的 json/jsonl/csv> 覆寫。" >&2
  exit 1
fi

[ -f "$PREDICTOR" ] || { echo "[error] predictor not found: $PREDICTOR" >&2; exit 1; }
[ -f "$PROMPT" ] || { echo "[error] prompt not found: $PROMPT" >&2; exit 1; }
[ -f "$DOC_TABLE" ] || { echo "[error] doc table not found: $DOC_TABLE" >&2; exit 1; }
[ -f "$PAGE_TABLE" ] || { echo "[error] page table not found: $PAGE_TABLE" >&2; exit 1; }
[ -f "$OFFSETS_PATH" ] || { echo "[error] offsets not found: $OFFSETS_PATH" >&2; exit 1; }

cmd=(
  "$PY" "$PREDICTOR"
  --backend "$BACKEND"
  --workers "$WORKERS"
  --timeout "$TIMEOUT"
  --model "$MODEL"
  --prompt-path "$PROMPT"
  --data "$DATA"
  --add-context
  --context-mode all
  --doc-table "$DOC_TABLE"
  --page-table "$PAGE_TABLE"
  --offsets-path "$OFFSETS_PATH"
  --output "$OUT"
)

if [ -n "$LIMIT" ]; then
  cmd+=(--limit "$LIMIT")
fi

echo "[$(date '+%H:%M:%S')] $NAME/$BACKEND prompt=$(basename "$PROMPT") data=$(basename "$DATA") workers=$WORKERS ${LIMIT:+limit=$LIMIT}"

if [ "$DRY_RUN" = "1" ]; then
  printf '[dry-run]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$LOG"

if [ -n "$BENCH" ] && [ -f "$BENCH" ]; then
  echo "=== score ($NAME, $(basename "$DATA"), $BACKEND) ==="
  "$PY" core/analysis/score_st3_full_coverage.py --benchmark "$BENCH" --pred "$OUT"
else
  echo "=== no scoring (test set has no labels; pred -> $OUT) ==="
fi
