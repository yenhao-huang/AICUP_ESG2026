# Run from repo root:
#   bash exp/integrated_stage_predictions/0614/submit/submit_5/stage1/retrain/seed.sh
#
# SEED sweep for ST1 on the pre-split a3_b1+val data. Trains the same recipe under
# several seeds to measure run-to-run variance and pick a stable checkpoint. Each
# seed keeps ONLY its best checkpoint (--no-epoch-saves), in its own /models dir +
# train_results json.
#
# ===== PARAMETER SPACE (edit here) ============================================
SEEDS=(42 7 123 2024 31337)        # add/remove seeds here
LOSS="${LOSS:-weighted_ce}"        # loss held fixed across seeds (override via env)
# =============================================================================

set -uo pipefail
cd /workspace/esg_contest
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE1=exp/integrated_stage_predictions/0614/submit/submit_5/stage1
TRAIN="$STAGE1/data/a3_b1_add_val.train.json"
VAL="$STAGE1/data/a3_b1_add_val.val.json"

echo "### seed sweep: seeds=[${SEEDS[*]}] loss=$LOSS GPU=$CUDA_VISIBLE_DEVICES"
for seed in "${SEEDS[@]}"; do
  echo "=== [seed=$seed] $(date +%T) ==="
  .venv/bin/python core/train/train_bert.py \
    --model large --stage st1 \
    --train-path "$TRAIN" --val-path "$VAL" \
    --loss "$LOSS" \
    --seed "$seed" \
    --model-dir "/models/submit_5_a3_b1_add_val_seed${seed}" \
    --batch-size 4 --no-epoch-saves \
    --output "$STAGE1/train_results/a3_b1_add_val_seed${seed}.json" \
    2>&1 | tee "$STAGE1/train_results/seed${seed}.log"
  echo "=== [seed=$seed] rc=${PIPESTATUS[0]} ==="
done
echo "### seed sweep DONE $(date +%T)"
