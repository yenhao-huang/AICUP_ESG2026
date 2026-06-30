#!/usr/bin/env bash
# Validate the top prompts on the FULL 600-row set (add_context, mode=all), then score.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
STAGE3="$EXP/stage3"
PY="${PY:-/workspace/esg_contest/.venv/bin/python}"; [ -x "$PY" ] || PY=python3
DATA="${DATA:-$EXP/data/val_ctxhit.json}"
CONC="${CONC:-4}"
mkdir -p "$EXP/preds" "$HERE/logs"

PROMPTS="${PROMPTS:-clear_lenient_tagged clear_checklist_tagged}"
for name in $PROMPTS; do
  out="$EXP/preds/win_${name}.csv"
  echo "[$(date '+%H:%M:%S')] 600 prompt=$name"
  PYTHONUNBUFFERED=1 "$PY" "$STAGE3/core/vlm_pred.py" \
    --data "$DATA" --output "$out" \
    --backend qwen --prompt-path "$EXP/prompts/${name}.txt" \
    --add-context --context-mode all --logprobs --concurrency "$CONC" \
    > "$HERE/logs/win_${name}.log" 2>&1
  echo "[$(date '+%H:%M:%S')] $name done -> $out"
done

echo "=== scoring prompt validation (600 rows, add_context) ==="
"$PY" "$HERE/score_all.py" \
  --benchmark "$DATA" --preds-dir "$EXP/preds" --pattern 'win_*.csv' \
  --output "$EXP/preds/prompt_scores_600.json"
