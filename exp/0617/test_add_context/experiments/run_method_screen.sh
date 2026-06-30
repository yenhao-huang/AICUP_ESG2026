#!/usr/bin/env bash
# Re-run the method ablation UNDER THE GOOD PROMPT (clear_lenient_tagged), 100 rows.
# Answers: does the page image help? does evidence_string help? — vs data-only / context.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
STAGE3="$EXP/stage3"
PY="${PY:-/workspace/esg_contest/.venv/bin/python}"; [ -x "$PY" ] || PY=python3
DATA="${DATA:-$EXP/data/val_ctxhit.100.jsonl}"
PROMPT="${PROMPT:-$EXP/prompts/clear_lenient_tagged.txt}"
BENCH="${BENCH:-$EXP/data/val_ctxhit.100.json}"
CONC="${CONC:-4}"
mkdir -p "$EXP/preds" "$HERE/logs"

run () {  # <name> <flags...>
  local name="$1"; shift
  local out="$EXP/preds/m_${name}.csv"
  echo "[$(date '+%H:%M:%S')] $name"
  PYTHONUNBUFFERED=1 "$PY" "$STAGE3/core/vlm_pred.py" \
    --data "$DATA" --output "$out" \
    --backend qwen --prompt-path "$PROMPT" --logprobs --concurrency "$CONC" "$@" \
    > "$HERE/logs/m_${name}.log" 2>&1
  echo "[$(date '+%H:%M:%S')] $name done"
}

run data_only        --no-add-context
run add_context      --add-context --context-mode all
run add_image        --no-add-context --add-image
run context_image    --add-context --context-mode all --add-image
run add_evidence_str --no-add-context --add-evidence-string

echo "=== scoring method ablation under good prompt (100 rows) ==="
"$PY" "$HERE/score_all.py" --benchmark "$BENCH" --preds-dir "$EXP/preds" \
  --pattern 'm_*.csv' --output "$EXP/preds/method_scores_100.json"
