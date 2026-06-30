#!/usr/bin/env bash
# Validate the method ablation on FULL 600 under the good prompt (clear_lenient_tagged):
# data_only / add_context / add_image / context_image. Settles the "does image help?" question.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
STAGE3="$EXP/stage3"
PY="${PY:-/workspace/esg_contest/.venv/bin/python}"; [ -x "$PY" ] || PY=python3
DATA="${DATA:-$EXP/data/val_ctxhit.json}"
PROMPT="${PROMPT:-$EXP/prompts/clear_lenient_tagged.txt}"
CONC="${CONC:-4}"
mkdir -p "$EXP/preds" "$HERE/logs"

run () {  # <name> <flags...>
  local name="$1"; shift
  echo "[$(date '+%H:%M:%S')] 600 $name"
  PYTHONUNBUFFERED=1 "$PY" "$STAGE3/core/vlm_pred.py" \
    --data "$DATA" --output "$EXP/preds/m6_${name}.csv" \
    --backend qwen --prompt-path "$PROMPT" --logprobs --concurrency "$CONC" "$@" \
    > "$HERE/logs/m6_${name}.log" 2>&1
  echo "[$(date '+%H:%M:%S')] 600 $name done"
}

run data_only     --no-add-context
run add_context   --add-context --context-mode all
run add_image     --no-add-context --add-image
run context_image --add-context --context-mode all --add-image

echo "=== scoring method ablation on 600 (good prompt) ==="
"$PY" "$HERE/score_all.py" --benchmark "$DATA" --preds-dir "$EXP/preds" \
  --pattern 'm6_*.csv' --output "$EXP/preds/method_scores_600.json"
