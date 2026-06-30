#!/usr/bin/env bash
# Compare the top-3 ablations (by 100-row 2-class Macro-F1) on the FULL 600-row val_ctxhit,
# then score all three against the 600-row benchmark.
#   top3: add_evidence_promise, add_evidence_string, add_context
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
STAGE3="$EXP/stage3"
PY="${PY:-/workspace/esg_contest/.venv/bin/python}"; [ -x "$PY" ] || PY=python3
DATA="${DATA:-$EXP/data/val_ctxhit.json}"            # 600 rows
PROMPT="${PROMPT:-$EXP/prompts/clear_notclear_with_context_scoped.txt}"
CONC="${CONC:-4}"
mkdir -p "$EXP/preds" "$HERE/logs"

run () {  # <name> <flags...>
  local name="$1"; shift
  local out="$EXP/preds/full_${name}.csv"
  echo "[$(date '+%H:%M:%S')] full600 $name"
  PYTHONUNBUFFERED=1 "$PY" "$STAGE3/core/vlm_pred.py" \
    --data "$DATA" --output "$out" \
    --backend qwen --prompt-path "$PROMPT" --logprobs --concurrency "$CONC" "$@" \
    > "$HERE/logs/full_${name}.log" 2>&1
  echo "[$(date '+%H:%M:%S')] full600 $name done -> $out"
}

run add_evidence_promise --no-add-context --add-evidence-string --add-promise-string
run add_evidence_string  --no-add-context --add-evidence-string
run add_context          --add-context --context-mode all

echo "=== scoring 600-row comparison ==="
"$PY" "$HERE/score_all.py" \
  --benchmark "$DATA" --preds-dir "$EXP/preds" --pattern 'full_*.csv' \
  --output "$EXP/preds/exp_scores_600.json"
