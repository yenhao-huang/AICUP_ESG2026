#!/usr/bin/env bash
# Pre-submit INDEPENDENT per-member check for submit_7: for each of the 15
# ensemble members (3 stages x 5 seeds), evaluate the member on ITS OWN held-out
# val split (data/ensemble/seed<S>/...val.json) -- the only leakage-free signal,
# since the global vpesg4k_val_1000 is folded into every member's training set.
# Reports per-member coverage + raw-head Macro-F1. Exits non-zero if any stage
# FAILs. This is a per-model test -- unrelated to the soft vote.
#
#   bash exp/integrated_stage_predictions/0615/submit_7/test_7/run_test.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"   # -> repo root /workspace/esg_contest
cd "$ROOT"
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv/bin/python"

DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"

ST1_GLOB="${ST1_GLOB:-/models/ensemble_st1_a3_b1_focal_g3_w4_seed*/best_st1.pt}"
ST2_GLOB="${ST2_GLOB:-/models/ensemble_st2_mix_a2_b3_seed*/best_st2.pt}"
ST3_GLOB="${ST3_GLOB:-/models/ensemble_mt_st123_w1_8_30_mlval_seed*/best_multitask_st3.pt}"

rc=0
run_stage() { # $1=stage $2=glob
  PYTHONUNBUFFERED=1 "$VENV" "$HERE/check_members.py" \
    --stage "$1" --ckpt-glob "$2" \
    --device "$DEVICE" --batch-size "$BATCH_SIZE" $LIMIT_ARG || rc=1
}

run_stage 1 "$ST1_GLOB"
run_stage 2 "$ST2_GLOB"
run_stage 3 "$ST3_GLOB"

echo
if [ "$rc" -eq 0 ]; then
  echo "############ ALL 15 MEMBERS EVALUATED ON OWN VAL -- no missing rows ############"
else
  echo "############ SOME CHECKS FAILED (see FAIL above) ############"
fi
exit "$rc"
