# Run from repo root:
#   bash exp/integrated_stage_predictions/0615/ensemble/stage2/ensemble/make_splits.sh
#
# ENSEMBLE split generator for ST2. Takes the ONE raw dataset
# (mix_a2_b3_add_val.json) and cuts it into 5 DIFFERENT stratified train/val
# partitions -- one per seed -- stratified on evidence_status. Splits land in
# data/ensemble/seed<S>/. Deterministic per seed; safe to re-run.
#
# ===== PARAMETER SPACE (edit here) ============================================
SEEDS=(42 7 123 2024 31337)        # one train/val split per seed
VAL_RATIO="${VAL_RATIO:-0.2}"      # per-class held-out fraction
LABEL_KEY="${LABEL_KEY:-evidence_status}"
# =============================================================================
set -uo pipefail
cd /workspace/esg_contest

STAGE2=exp/integrated_stage_predictions/0615/ensemble/stage2
INPUT="$STAGE2/data/mix_a2_b3_add_val.json"
NAME=mix_a2_b3_add_val

[ -f "$INPUT" ] || { echo "[error] missing raw input: $INPUT" >&2; exit 1; }

echo "### ensemble splits ST2: seeds=[${SEEDS[*]}] val_ratio=$VAL_RATIO label=$LABEL_KEY"
for seed in "${SEEDS[@]}"; do
  OUT="$STAGE2/data/ensemble/seed${seed}"
  mkdir -p "$OUT"
  echo "=== [seed=$seed] -> $OUT ==="
  .venv/bin/python "$STAGE2/split_train_val_by_class.py" \
    --input "$INPUT" \
    --train-out "$OUT/${NAME}.train.json" \
    --val-out   "$OUT/${NAME}.val.json" \
    --label-key "$LABEL_KEY" --val-ratio "$VAL_RATIO" --seed "$seed"
done
echo "### splits DONE"
