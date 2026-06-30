#!/usr/bin/env bash
# Shared setup for the Stage 3 vlm_pred ablation experiments (100-row smoke).
# Sourced by each run_*.sh. Defines run_exp <name> <extra vlm_pred flags...>.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # experiments/qwen/
EXP="$(cd "$HERE/../.." && pwd)"                         # test_add_context/
STAGE3="$EXP/stage3"

PY="${PY:-/workspace/esg_contest/.venv/bin/python}"
[ -x "$PY" ] || PY=python3
DATA="${DATA:-$EXP/data/val.100.jsonl}"
PROMPT="${PROMPT:-$EXP/prompts/clear_notclear_with_context_scoped.txt}"
LIMIT="${LIMIT:-}"                                       # e.g. LIMIT=3 for a quick wiring check
CONC="${CONC:-4}"

mkdir -p "$EXP/preds" "$HERE/logs"

run_exp () {  # <name> <extra flags...>
  local name="$1"; shift
  local out="$EXP/preds/exp_${name}.csv"
  local log="$HERE/logs/exp_${name}.log"
  echo "[$(date '+%H:%M:%S')] $name  data=$(basename "$DATA") ${LIMIT:+limit=$LIMIT}"
  PYTHONUNBUFFERED=1 "$PY" "$STAGE3/core/vlm_pred.py" \
    --data "$DATA" --output "$out" \
    --backend qwen --prompt-path "$PROMPT" --logprobs --concurrency "$CONC" \
    ${LIMIT:+--limit "$LIMIT"} "$@" \
    2>&1 | tee "$log"
  echo "[$(date '+%H:%M:%S')] $name done -> $out"
}
