#!/usr/bin/env bash
# all on the 100-row set, pinned to the codex backend; prints the Stage 3 score.
# Inputs: --add-page-abstract --add-context --context-mode all --add-evidence-string --add-promise-string --add-image
#   ./run_all_100.sh            # codex, 8 workers
#   CONC=2 ./run_all_100.sh     # codex: lower concurrency to avoid throttling
#   LIMIT=5 ./run_all_100.sh    # quick wiring check on 5 rows
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # experiments/codex/
EXP="$(cd "$HERE/../.." && pwd)"                            # test_add_context/
STAGE3="$EXP/stage3"
REPO="$(cd "$EXP/../../../.." && pwd)"                      # esg_contest repo root
PY="${PY:-/workspace/esg_contest/.venv/bin/python}"; [ -x "$PY" ] || PY=python3

NAME=all
BACKEND=codex
CONC="${CONC:-8}"
TIMEOUT="${TIMEOUT:-300}"
MODEL="${MODEL:-gpt-5.5}"                  # codex only
LIMIT="${LIMIT:-}"
PROMPT="$EXP/prompts/codex/all.txt"
DATA="${DATA:-$EXP/data/val.100.jsonl}"
BENCH="${BENCH:-$EXP/data/val.100.json}"
OUT="$EXP/preds/$BACKEND/${NAME}_100_${BACKEND}.csv"
LOG="$HERE/logs/${NAME}_100_${BACKEND}.log"
mkdir -p "$EXP/preds/$BACKEND" "$HERE/logs"

INPUTS=(--add-page-abstract --add-context --context-mode all --add-evidence-string --add-promise-string --add-image)

extra=()
[ "$BACKEND" = codex ] && extra=(--model "$MODEL")
[ "$BACKEND" = qwen ]  && extra=(--logprobs)

echo "[$(date '+%H:%M:%S')] $NAME/$BACKEND  prompt=$(basename "$PROMPT")  conc=$CONC  ${LIMIT:+limit=$LIMIT}"
PYTHONUNBUFFERED=1 "$PY" "$STAGE3/core/vlm_pred.py" \
  --backend "$BACKEND" --concurrency "$CONC" --timeout "$TIMEOUT" \
  --prompt-path "$PROMPT" --data "$DATA" \
  "${INPUTS[@]}" ${LIMIT:+--limit "$LIMIT"} "${extra[@]}" \
  --output "$OUT" 2>&1 | tee "$LOG"

echo "=== score ($NAME, 100 rows, $BACKEND) ==="
"$PY" "$REPO/core/analysis/score_st3_full_coverage.py" --benchmark "$BENCH" --pred "$OUT"
