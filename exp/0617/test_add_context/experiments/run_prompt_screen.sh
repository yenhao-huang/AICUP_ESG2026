#!/usr/bin/env bash
# Screen the new clarity prompts on the 100-row set, all with add_context (mode=all),
# then score vs the scoped baseline. Goal: fix the Not-Clear over-prediction.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
STAGE3="$EXP/stage3"
PY="${PY:-/workspace/esg_contest/.venv/bin/python}"; [ -x "$PY" ] || PY=python3
DATA="${DATA:-$EXP/data/val_ctxhit.100.jsonl}"
CONC="${CONC:-4}"
mkdir -p "$EXP/preds" "$HERE/logs"

# prompt files to screen (scoped = current baseline)
PROMPTS="scoped clear_lenient_tagged clear_fewshot_tagged clear_checklist_tagged"
declare -A FILE=(
  [scoped]="clear_notclear_with_context_scoped.txt"
  [clear_lenient_tagged]="clear_lenient_tagged.txt"
  [clear_fewshot_tagged]="clear_fewshot_tagged.txt"
  [clear_checklist_tagged]="clear_checklist_tagged.txt"
)

for name in $PROMPTS; do
  out="$EXP/preds/pp_${name}.csv"
  echo "[$(date '+%H:%M:%S')] prompt=$name"
  PYTHONUNBUFFERED=1 "$PY" "$STAGE3/core/vlm_pred.py" \
    --data "$DATA" --output "$out" \
    --backend qwen --prompt-path "$EXP/prompts/${FILE[$name]}" \
    --add-context --context-mode all --logprobs --concurrency "$CONC" \
    > "$HERE/logs/pp_${name}.log" 2>&1
  echo "[$(date '+%H:%M:%S')] $name done -> $out"
done

echo "=== scoring prompt screen (100 rows, add_context) ==="
"$PY" "$HERE/score_all.py" \
  --benchmark "$EXP/data/val_ctxhit.100.json" --preds-dir "$EXP/preds" --pattern 'pp_*.csv' \
  --output "$EXP/preds/prompt_scores_100.json"
