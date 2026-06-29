# Run from repo root:
#   bash exp/integrated_stage_predictions/0615/ensemble/stage1/ensemble/train.sh
#
# ENSEMBLE trainer for ST1. Trains 5 BERT members: member <seed> on the
# data/ensemble/seed<seed>/ split from make_splits.sh, each with its own --seed.
# True ensemble -- every model sees a DIFFERENT train/val partition AND a
# DIFFERENT RNG seed. Each member keeps ONLY its best checkpoint
# (--no-epoch-saves), in its own /models dir + train_results json.
# Run make_splits.sh FIRST.
#
# ===== PARAMETER SPACE (edit here) ============================================
SEEDS=(42 7 123 2024 31337)        # must match make_splits.sh
# loss held fixed across members: focal_g3_w4 recipe (best ST1 loss from
# custom_loss.sh -> /models/submit_5_a3_b1_add_val_focal_g3_w4): focal, manual
# class weights No=4/Yes=1, gamma 3.0 (harder focusing on the scarce No class).
LOSS="${LOSS:-focal}"
CLASS_WEIGHTS="${CLASS_WEIGHTS:-4.0,1.0}"   # No=0, Yes=1
FOCAL_GAMMA="${FOCAL_GAMMA:-3.0}"
LOSS_TAG="${LOSS_TAG:-focal_g3_w4}"         # goes into model-dir / output names
# =============================================================================
set -uo pipefail
cd /workspace/esg_contest
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE1=exp/integrated_stage_predictions/0615/ensemble/stage1
NAME=a3_b1_add_val
RESULTS="$STAGE1/ensemble/train_results"
mkdir -p "$RESULTS"

echo "### ensemble train ST1: seeds=[${SEEDS[*]}] loss=$LOSS($LOSS_TAG) cw=$CLASS_WEIGHTS gamma=$FOCAL_GAMMA GPU=$CUDA_VISIBLE_DEVICES"
for seed in "${SEEDS[@]}"; do
  SPLIT="$STAGE1/data/ensemble/seed${seed}"
  TRAIN="$SPLIT/${NAME}.train.json"
  VAL="$SPLIT/${NAME}.val.json"
  for f in "$TRAIN" "$VAL"; do
    [ -f "$f" ] || { echo "[error] missing split: $f (run make_splits.sh first)" >&2; exit 1; }
  done
  echo "=== [member seed=$seed] $(date +%T) ==="
  .venv/bin/python core/train/train_bert.py \
    --model large --stage st1 \
    --train-path "$TRAIN" --val-path "$VAL" \
    --loss "$LOSS" --class-weights "$CLASS_WEIGHTS" --focal-gamma "$FOCAL_GAMMA" \
    --seed "$seed" \
    --model-dir "/models/ensemble_st1_a3_b1_${LOSS_TAG}_seed${seed}" \
    --batch-size 4 --no-epoch-saves \
    --output "$RESULTS/${NAME}_${LOSS_TAG}_seed${seed}.json" \
    2>&1 | tee "$RESULTS/${LOSS_TAG}_seed${seed}.log"
  echo "=== [member seed=$seed] rc=${PIPESTATUS[0]} ==="
done
echo "### ensemble train DONE $(date +%T)"
