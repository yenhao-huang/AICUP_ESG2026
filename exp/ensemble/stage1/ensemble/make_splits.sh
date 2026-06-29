# Run from repo root:
#   bash exp/integrated_stage_predictions/0615/ensemble/stage1/ensemble/make_splits.sh
#
# ENSEMBLE split generator for ST1. Takes the ONE raw dataset
# (a3_b1_add_val.json, 2500 rows) and cuts it into 5 DIFFERENT stratified
# train/val partitions -- one per seed -- so each ensemble member trains on its
# own data slice and is selected on its own held-out val. Splits land in
# data/ensemble/seed<S>/. Deterministic per seed; safe to re-run.
#
# ===== PARAMETER SPACE (edit here) ============================================
SEEDS=(42 7 123 2024 31337)        # one train/val split per seed
VAL_RATIO="${VAL_RATIO:-0.2}"      # per-class held-out fraction
LABEL_KEY="${LABEL_KEY:-promise_status}"
# =============================================================================
set -uo pipefail
cd /workspace/esg_contest

STAGE1=exp/integrated_stage_predictions/0615/ensemble/stage1
INPUT="$STAGE1/data/a3_b1_add_val.json"
NAME=a3_b1_add_val

[ -f "$INPUT" ] || { echo "[error] missing raw input: $INPUT" >&2; exit 1; }

echo "### ensemble splits ST1: seeds=[${SEEDS[*]}] val_ratio=$VAL_RATIO label=$LABEL_KEY"
for seed in "${SEEDS[@]}"; do
  OUT="$STAGE1/data/ensemble/seed${seed}"
  mkdir -p "$OUT"
  echo "=== [seed=$seed] -> $OUT ==="
  .venv/bin/python "$STAGE1/split_train_val_by_class.py" \
    --input "$INPUT" \
    --train-out "$OUT/${NAME}.train.json" \
    --val-out   "$OUT/${NAME}.val.json" \
    --label-key "$LABEL_KEY" --val-ratio "$VAL_RATIO" --seed "$seed"
done
echo "### splits DONE"
