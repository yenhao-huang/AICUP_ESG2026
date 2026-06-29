# Run from repo root:
#   bash exp/integrated_stage_predictions/0615/ensemble/stage3/ensemble/train.sh
#
# ENSEMBLE trainer for ST3-focused multitask BERT (joint ST1+ST2+ST3, recipe
# mt_st123_w1_8_30_mlval). Trains 5 members: member <seed> on the
# data/ensemble/seed<seed>/ split from make_splits.sh, each with its own --seed.
# True ensemble -- every model sees a DIFFERENT train/val partition AND a
# DIFFERENT RNG seed. best_multitask_st3.pt is picked by ST3 val Macro-F1. Each
# member keeps ONLY its best checkpoint (--no-epoch-saves). Run make_splits.sh FIRST.
#
# ===== PARAMETER SPACE (edit here) ============================================
SEEDS=(42 7 123 2024 31337)        # must match make_splits.sh
EPOCHS="${EPOCHS:-10}"             # epochs held fixed across members
# =============================================================================
set -uo pipefail
cd /workspace/esg_contest
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE3=exp/integrated_stage_predictions/0615/ensemble/stage3
NAME=vpesg_4k_train_1000_add_val
RESULTS="$STAGE3/ensemble/train_results"
mkdir -p "$RESULTS"

echo "### ensemble train ST3(multitask): seeds=[${SEEDS[*]}] epochs=$EPOCHS GPU=$CUDA_VISIBLE_DEVICES"
for seed in "${SEEDS[@]}"; do
  SPLIT="$STAGE3/data/ensemble/seed${seed}"
  TRAIN="$SPLIT/${NAME}.train.json"
  VAL="$SPLIT/${NAME}.val.json"
  for f in "$TRAIN" "$VAL"; do
    [ -f "$f" ] || { echo "[error] missing split: $f (run make_splits.sh first)" >&2; exit 1; }
  done
  echo "=== [member seed=$seed] $(date +%T) ==="
  .venv/bin/python core/train/train_multitaskbert_stage3.py \
    --model large \
    --train-path "$TRAIN" --val-path "$VAL" \
    --st1-loss weighted_ce --st2-loss weighted_ce \
    --st3-loss weighted_ce --st3-class-weights 1,8,30 \
    --task-weights 0.2,0.3,0.5 \
    --batch-size 4 --grad-accum 4 \
    --epochs "$EPOCHS" --no-epoch-saves \
    --seed "$seed" \
    --model-dir "/models/ensemble_mt_st123_w1_8_30_mlval_seed${seed}" \
    --output "$RESULTS/${NAME}_seed${seed}.json" \
    2>&1 | tee "$RESULTS/seed${seed}.log"
  echo "=== [member seed=$seed] rc=${PIPESTATUS[0]} ==="
done
echo "### ensemble train DONE $(date +%T)"
