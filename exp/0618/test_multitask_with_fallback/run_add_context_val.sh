#!/usr/bin/env bash
# add_context Stage 3 prediction over the COPIED VAL set (vpesg_4k_train_1000_add_val.val.json,
# 402 rows, LABELED), codex backend. Output + scoring under test_multitask_with_fallback/.
#
# Differences vs submission_12/run_add_context_all.sh:
#   - DATA   -> this dir's data/vpesg_4k_train_1000_add_val.val.json (val carries pdf_url on all 402
#               rows, which ContextBuilder needs to resolve doc_id -> same-page OCR).
#   - EXP    -> reuses the 0617 test_add_context resources (vlm_pred.py + ContextBuilder + offsets),
#               because 0618 has NO test_add_context. vlm_pred.py self-locates its context assets via
#               _EXP_ROOT = parents[1], so calling the 0617 copy uses the 0617 offsets/url2doc.
#   - BENCH  -> defaults to DATA: val HAS labels, so scoring is ON (score_st3_full_coverage.py,
#               full-coverage {Clear,Not Clear,N/A} + GT-gated 2-class macro-F1).
#   - OUT    -> this dir's preds/codex/.
#
# CONTEXT: same-page OCR text is located live by ContextBuilder, resolving doc_id via the row's
#          pdf_url (url2doc) then the page from offsets.jsonl (matched_page_no).
# CACHE  : vlm_pred.py appends each done row to <output>.csv.cache.jsonl; reruns skip cached ids.
#          Delete that .cache.jsonl to force a clean full rerun.
#
#   bash run_add_context_val.sh                 # codex, CONC=8, scores against val labels
#   LIMIT=4 bash run_add_context_val.sh         # quick wiring/context-hit check first
#   BENCH= bash run_add_context_val.sh          # disable scoring
#   DATA=/path/other_with_id_pdf_url.json bash run_add_context_val.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # .../0618/test_multitask_with_fallback
REPO="$(cd "$HERE/../../../.." && pwd)"                  # /workspace/esg_contest
EXP="$REPO/exp/integrated_stage_predictions/0617/test_add_context"   # vlm_pred.py + prompts + offsets
STAGE3="$EXP/stage3"
PY="${PY:-/workspace/esg_contest/.venv/bin/python}"; [ -x "$PY" ] || PY=python3

NAME=add_context
BACKEND=codex
CONC="${CONC:-8}"                          # lower to 2-4 if codex throttles/stalls
TIMEOUT="${TIMEOUT:-300}"
MODEL="${MODEL:-gpt-5.5}"
LIMIT="${LIMIT:-}"
PROMPT="$EXP/prompts/codex/add-context.txt"
# 預測對象 = 複製到本目錄的 val（含 pdf_url，ContextBuilder 解 same-page OCR 必需）
DATA="${DATA:-$HERE/data/vpesg_4k_train_1000_add_val.val.json}"
# val 有 evidence_quality 標籤 -> 預設啟用評分（BENCH= 可關閉）
BENCH="${BENCH:-$DATA}"
OUT="$HERE/preds/codex/${NAME}_val_${BACKEND}.csv"
LOG="$HERE/logs/${NAME}_val_${BACKEND}.log"
mkdir -p "$HERE/preds/codex" "$HERE/logs"

for f in "$STAGE3/core/vlm_pred.py" "$PROMPT" "$DATA"; do
  [ -f "$f" ] || { echo "[error] not found: $f" >&2; exit 1; }
done

INPUTS=(--add-context --context-mode all)
extra=(--model "$MODEL")

echo "[$(date '+%H:%M:%S')] $NAME/$BACKEND  prompt=$(basename "$PROMPT")  data=$(basename "$DATA")  conc=$CONC  ${LIMIT:+limit=$LIMIT}"
PYTHONUNBUFFERED=1 "$PY" "$STAGE3/core/vlm_pred.py" \
  --backend "$BACKEND" --concurrency "$CONC" --timeout "$TIMEOUT" \
  --prompt-path "$PROMPT" --data "$DATA" \
  "${INPUTS[@]}" ${LIMIT:+--limit "$LIMIT"} "${extra[@]}" \
  --output "$OUT" 2>&1 | tee "$LOG"

if [ -n "$BENCH" ] && [ -f "$BENCH" ]; then
  echo "=== score ($NAME, val, $BACKEND) ==="
  "$PY" "$REPO/core/analysis/score_st3_full_coverage.py" --benchmark "$BENCH" --pred "$OUT"
else
  echo "=== no scoring (set BENCH=<labeled.json> to enable) ==="
fi
