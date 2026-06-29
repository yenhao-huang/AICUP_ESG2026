# Run from repo root:
#   bash exp/integrated_stage_predictions/0614/submit/submit_5/stage2/retrain/seed.sh
#
# SEED sweep for ST2 on the pre-split mix_a2_b3+val data. Trains the same recipe
# (the loop001_st2_synth_A2_b3_c1_25 recipe) under several seeds to measure
# run-to-run variance and pick a stable checkpoint. Each seed keeps ONLY its best
# checkpoint (--no-epoch-saves), in its own /models dir + train_results json.
#
# Pre-split files come from split_train_val_by_class.py (see custom_loss.sh header).
#
# ===== PARAMETER SPACE (edit here) ============================================
SEEDS=(42 7 123 2024 31337)        # add/remove seeds here
LOSS="${LOSS:-weighted_ce}"        # loss held fixed across seeds (override via env)
EPOCHS="${EPOCHS:-10}"             # epochs held fixed across seeds (override via env)
# =============================================================================

set -uo pipefail
cd /workspace/esg_contest
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE2=exp/integrated_stage_predictions/0614/submit/submit_5/stage2
TRAIN="$STAGE2/data/mix_a2_b3_add_val.train.json"
VAL="$STAGE2/data/mix_a2_b3_add_val.val.json"
mkdir -p "$STAGE2/train_results"

for f in "$TRAIN" "$VAL"; do
  [ -f "$f" ] || { echo "[error] missing split file: $f (run split_train_val_by_class.py first)" >&2; exit 1; }
done

echo "### seed sweep: seeds=[${SEEDS[*]}] loss=$LOSS epochs=$EPOCHS GPU=$CUDA_VISIBLE_DEVICES"
for seed in "${SEEDS[@]}"; do
  echo "=== [seed=$seed] $(date +%T) ==="
  .venv/bin/python core/train/train_bert.py \
    --model large --stage st2 \
    --train-path "$TRAIN" --val-path "$VAL" \
    --epochs "$EPOCHS" --max-len 512 \
    --loss "$LOSS" \
    --seed "$seed" \
    --model-dir "/models/esg_contest/loop001_st2_synth_A2_b3_c1_25_add_val_seed${seed}" \
    --batch-size 4 --no-epoch-saves \
    --output "$STAGE2/train_results/mix_a2_b3_add_val_seed${seed}.json" \
    2>&1 | tee "$STAGE2/train_results/seed${seed}.log"
  echo "=== [seed=$seed] rc=${PIPESTATUS[0]} ==="
done
echo "### seed sweep DONE $(date +%T)"
