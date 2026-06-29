# Run from repo root:
#   bash exp/integrated_stage_predictions/0615/ensemble/stage3/ensemble/train_31337_seed.sh
#
# PARALLEL SHARD of train.sh -- trains ONLY the last ensemble member (seed 31337)
# so it can run on a SECOND GPU in parallel while train.sh handles the rest. Same
# recipe / outputs as train.sh; the member writes to its own per-seed /models dir,
# so the shards never collide. (Make the other run skip 31337 so it isn't trained
# twice.)
#
# ===== PARAMETER SPACE (edit here) ============================================
SEEDS=(2024)                      # the last seed only
EPOCHS="${EPOCHS:-10}"             # epochs held fixed across members
# =============================================================================
set -uo pipefail
cd /workspace/esg_contest
export CUDA_VISIBLE_DEVICES="${GPU:-0}"   # default to a different GPU than train.sh (GPU=1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE3=exp/integrated_stage_predictions/0615/ensemble/stage3
NAME=vpesg_4k_train_1000_add_val
RESULTS="$STAGE3/ensemble/train_results"
mkdir -p "$RESULTS"

echo "### ensemble train ST3(multitask) SHARD: seeds=[${SEEDS[*]}] epochs=$EPOCHS GPU=$CUDA_VISIBLE_DEVICES"
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
echo "### ensemble train ST3 SHARD DONE $(date +%T)"
