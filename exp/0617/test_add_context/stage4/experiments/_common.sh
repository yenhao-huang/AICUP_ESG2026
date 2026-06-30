#!/usr/bin/env bash
# Shared setup for the Stage 4 verification_timeline test_add_context ablations.
# Sourced by each run_*.sh. Defines run_variant <name> <prompt> <input flags...>.
#
# Vanilla baseline  : prompts/codex/vanilla.txt  (= configs/prompt/stage4/codex/boundary_rules_v4.txt), data-only.
# Enrichment probes : add same-page-context / page-image / promise_string on top.
#
# DATA-USE NOTE: same-page-context, the page image, and promise_string all exceed
# the CLAUDE.md `data`-only default. They are enabled here only for this explicit
# user-approved Stage 4 probe and must NOT be promoted as a data-only runtime path.
#
# Backend is Codex gpt-5.5 (multimodal: it honours --image, so add_image / all are
# real, not ignored). Set BACKEND=qwen to use the local VLM endpoint instead.
#
# Knobs:
#   SET=100 | full      which eval subset (default 100)
#   BACKEND=codex|qwen   (default codex)
#   MODEL=gpt-5.5        codex model
#   CONC=8  TIMEOUT=300  LIMIT=<n>   (LIMIT for a quick wiring check)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # experiments/
STAGE4="$(cd "$HERE/.." && pwd)"                          # stage4/
EXP="$(cd "$STAGE4/.." && pwd)"                           # test_add_context/

PY="${PY:-/workspace/esg_contest/.venv/bin/python}"
[ -x "$PY" ] || PY=python3

SET="${SET:-100}"
if [ "$SET" = full ]; then
  DATA="${DATA:-$STAGE4/data/val_yes.json}"
  BENCH="${BENCH:-$STAGE4/data/val_yes.json}"
else
  DATA="${DATA:-$STAGE4/data/val_yes.100.json}"
  BENCH="${BENCH:-$STAGE4/data/val_yes.100.json}"
fi

BACKEND="${BACKEND:-codex}"
MODEL="${MODEL:-gpt-5.5}"
CONC="${CONC:-8}"
TIMEOUT="${TIMEOUT:-300}"
LIMIT="${LIMIT:-}"

mkdir -p "$STAGE4/preds/codex" "$HERE/logs"

run_variant () {  # <name> <prompt-basename> <input flags...>
  local name="$1"; shift
  local prompt_base="$1"; shift
  local prompt="$STAGE4/prompts/codex/${prompt_base}"
  local tag="${SET}_${BACKEND}"
  local out="$STAGE4/preds/codex/${name}_${tag}.csv"
  local log="$HERE/logs/${name}_${tag}.log"
  local score_out="$STAGE4/preds/codex/${name}_${tag}.score.json"

  local extra=()
  [ "$BACKEND" = codex ] && extra=(--model "$MODEL")
  [ "$BACKEND" = qwen ]  && extra=(--logprobs)

  echo "[$(date '+%H:%M:%S')] $name/$BACKEND  prompt=$prompt_base  set=$SET  conc=$CONC  ${LIMIT:+limit=$LIMIT}"
  PYTHONUNBUFFERED=1 "$PY" "$STAGE4/core/vlm_pred.py" \
    --backend "$BACKEND" --concurrency "$CONC" --timeout "$TIMEOUT" \
    --prompt-path "$prompt" --data "$DATA" \
    "$@" ${LIMIT:+--limit "$LIMIT"} "${extra[@]}" \
    --output "$out" 2>&1 | tee "$log"

  echo "=== score ($name, set=$SET, $BACKEND) ==="
  "$PY" "$HERE/score_st4.py" --benchmark "$BENCH" --pred "$out" --output "$score_out"
}
