# Run from repo root:
#   bash exp/integrated_stage_predictions/0615/ensemble/stage3/ensemble/make_splits.sh
#
# ENSEMBLE split generator for ST3 (multitask). Takes the ONE raw dataset
# (vpesg_4k_train_1000_add_val.json) and cuts it into 5 DIFFERENT stratified
# train/val partitions -- one per seed -- stratified on evidence_quality. The
# scarce Misleading rows are forced ENTIRELY into val (--force-val-classes), so
# every member's ST3 head trains on Clear / Not Clear and keeps Misleading as a
# held-out signal; only the stratified part varies by seed. Splits land in
# data/ensemble/seed<S>/. Deterministic per seed; safe to re-run.
#
# ===== PARAMETER SPACE (edit here) ============================================
SEEDS=(42 7 123 2024 31337)        # one train/val split per seed
VAL_RATIO="${VAL_RATIO:-0.2}"      # per-class held-out fraction
LABEL_KEY="${LABEL_KEY:-evidence_quality}"
FORCE_VAL="${FORCE_VAL:-Misleading}"
# =============================================================================
set -uo pipefail
cd /workspace/esg_contest

STAGE3=exp/integrated_stage_predictions/0615/ensemble/stage3
INPUT="$STAGE3/data/vpesg_4k_train_1000_add_val.json"
NAME=vpesg_4k_train_1000_add_val

[ -f "$INPUT" ] || { echo "[error] missing raw input: $INPUT" >&2; exit 1; }

echo "### ensemble splits ST3: seeds=[${SEEDS[*]}] val_ratio=$VAL_RATIO label=$LABEL_KEY force_val=$FORCE_VAL"
for seed in "${SEEDS[@]}"; do
  OUT="$STAGE3/data/ensemble/seed${seed}"
  mkdir -p "$OUT"
  echo "=== [seed=$seed] -> $OUT ==="
  .venv/bin/python "$STAGE3/split_train_val_by_class.py" \
    --input "$INPUT" \
    --train-out "$OUT/${NAME}.train.json" \
    --val-out   "$OUT/${NAME}.val.json" \
    --label-key "$LABEL_KEY" --force-val-classes "$FORCE_VAL" \
    --val-ratio "$VAL_RATIO" --seed "$seed"
done
echo "### splits DONE"
