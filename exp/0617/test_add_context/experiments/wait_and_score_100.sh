#!/usr/bin/env bash
# Wait for all five 100-row ablation CSVs to finish, then score + rank them.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
PY="${PY:-/workspace/esg_contest/.venv/bin/python}"; [ -x "$PY" ] || PY=python3
NAMES="add_evidence_string add_image add_evidence_promise add_context add_context_window"
DEADLINE=$((SECONDS + 3000))   # 50 min cap

while true; do
  done=1
  for n in $NAMES; do
    f="$EXP/preds/exp_${n}.csv"
    if [ ! -f "$f" ] || [ "$(( $(wc -l < "$f") - 1 ))" -lt 100 ]; then done=0; fi
  done
  [ "$done" = 1 ] && break
  if [ $SECONDS -ge $DEADLINE ]; then echo "TIMEOUT: not all 100-row runs finished"; break; fi
  sleep 15
done

echo "=== scoring 100-row ablations ==="
"$PY" "$HERE/score_all.py" \
  --benchmark "$EXP/data/val_ctxhit.100.json" \
  --preds-dir "$EXP/preds" \
  --output "$EXP/preds/exp_scores_100.json"
