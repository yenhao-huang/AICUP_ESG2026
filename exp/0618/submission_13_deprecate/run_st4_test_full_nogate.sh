#!/usr/bin/env bash
# Predict Stage 4 verification_timeline over the ENTIRE test 2000 set (full, no
# pre-filter at predict time), THEN apply the ST1 gate as a POST-PROCESS.
#
# Parallel to 0617/test_add_context/stage4/experiments/run_add_context2_gated_v4.sh,
# with these changes:
#   1. PREDICT FULL: DATA = full data/raw_data/vpesg4k_test_2000.json (2000 rows).
#      The reference uses val_yes.json (pre-filtered to promise=Yes); here EVERY
#      row gets a codex ST4 prediction — no promise pre-filter during prediction.
#   2. POST GATE: after prediction, apply_stage1_gate_to_stage4.py sets
#      verification_timeline="N/A" for every id whose ST1 promise_status=="No"
#      (uses submission_13/stage1/bert_gemma.csv; covers all 2000 test ids, 374 No).
#   3. Scoring OFF: the test set has no labels (blind), so no score_st4.py step.
# Outputs: raw  ..._nogate_codex.csv   (full prediction, before gate)
#          gated ..._st1gated_codex.csv (after ST1 gate)
# Everything else follows the reference: codex gpt-5.5 backend and
# --add-context --context-mode all. PROMPT is bumped to add-context-2-gated-v6
# (the "gated_v6" in its name is the TIMELINE-RULE version, NOT a cascade gate).
#
# Reuses the 0617 test_add_context/stage4 harness (vlm_pred.py + ContextBuilder +
# prompts + offsets), which self-locates its context assets via _EXP_ROOT.
#
# DATA-USE: --add-context injects same-page OCR (resolved via each row's pdf_url),
# which EXCEEDS the CLAUDE.md `data`-only default. This is a user-approved probe
# and must NOT be promoted as a data-only runtime path.
#
# CACHE: vlm_pred.py appends each done row to <output>.csv.cache.jsonl; reruns skip
# cached ids, so a killed run never re-spends. Delete that .cache.jsonl to force a
# clean full rerun.
#
#   bash run_st4_test_full_nogate.sh                 # codex, CONC=8, full 2000 rows
#   LIMIT=4 bash run_st4_test_full_nogate.sh         # quick wiring / context-hit check first
#   CONC=4 bash run_st4_test_full_nogate.sh          # lower if codex throttles/stalls
#   BACKEND=qwen bash run_st4_test_full_nogate.sh    # local VLM endpoint instead of codex
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # .../0618/submission_13
REPO="$(cd "$HERE/../../../.." && pwd)"                         # /workspace/esg_contest
STAGE4="$REPO/exp/integrated_stage_predictions/0617/test_add_context/stage4"
PY="${PY:-/workspace/esg_contest/.venv/bin/python}"; [ -x "$PY" ] || PY=python3

BACKEND="${BACKEND:-codex}"
MODEL="${MODEL:-gpt-5.5}"
CONC="${CONC:-8}"
TIMEOUT="${TIMEOUT:-300}"
LIMIT="${LIMIT:-}"
PROMPT="$STAGE4/prompts/codex/add-context-2-gated-v6.txt"
# 整個 test 2000（含 pdf_url，ContextBuilder 解 same-page OCR 必需）；不預過濾、不接 ST1 gate
DATA="${DATA:-$REPO/data/raw_data/vpesg4k_test_2000.json}"
OUT="$HERE/stage4/preds/codex/st4_add_context2_v6_test2000_nogate_${BACKEND}.csv"
LOG="$HERE/logs/st4_add_context2_v6_test2000_nogate_${BACKEND}.log"
mkdir -p "$HERE/stage4/preds/codex" "$HERE/logs"

for f in "$STAGE4/core/vlm_pred.py" "$PROMPT" "$DATA"; do
  [ -f "$f" ] || { echo "[error] not found: $f" >&2; exit 1; }
done

extra=()
[ "$BACKEND" = codex ] && extra=(--model "$MODEL")
[ "$BACKEND" = qwen ]  && extra=(--logprobs)

echo "[$(date '+%H:%M:%S')] st4 add_context2_v6 / $BACKEND  data=$(basename "$DATA")  conc=$CONC  ${LIMIT:+limit=$LIMIT}  (NO GATE, full test2000)"
PYTHONUNBUFFERED=1 "$PY" "$STAGE4/core/vlm_pred.py" \
  --backend "$BACKEND" --concurrency "$CONC" --timeout "$TIMEOUT" \
  --prompt-path "$PROMPT" --data "$DATA" \
  --add-context --context-mode all \
  ${LIMIT:+--limit "$LIMIT"} "${extra[@]}" \
  --output "$OUT" 2>&1 | tee "$LOG"

echo "=== raw full (no-gate) ST4 prediction -> $OUT ==="

# === POST-PROCESS: apply ST1 gate (promise_status==No -> verification_timeline=N/A) ===
STAGE1_PRED="${STAGE1_PRED:-$HERE/stage1/bert_gemma.csv}"
GATE_PY="$REPO/exp/integrated_stage_predictions/0617/submit_9/scripts/apply_stage1_gate_to_stage4.py"
GATED_OUT="$HERE/stage4/preds/codex/st4_add_context2_v6_test2000_st1gated_${BACKEND}.csv"
for f in "$STAGE1_PRED" "$GATE_PY"; do
  [ -f "$f" ] || { echo "[error] ST1 gate input not found: $f" >&2; exit 1; }
done
echo "=== apply ST1 gate (ST1=No -> N/A) using $(basename "$STAGE1_PRED") ==="
"$PY" "$GATE_PY" --stage1 "$STAGE1_PRED" --stage4 "$OUT" --output "$GATED_OUT"

echo "=== done (test2000 is blind; no scoring) ==="
echo "raw   -> $OUT"
echo "gated -> $GATED_OUT"
