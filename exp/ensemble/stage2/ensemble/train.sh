# Run from repo root:
#   bash exp/integrated_stage_predictions/0615/ensemble/stage2/ensemble/train.sh
#
# ENSEMBLE trainer for ST2 (loop001_st2_synth_A2_b3_c1_25 recipe). Trains 5 BERT
# members: member <seed> on the data/ensemble/seed<seed>/ split from
# make_splits.sh, each with its own --seed. True ensemble -- every model sees a
# DIFFERENT train/val partition AND a DIFFERENT RNG seed. Each member keeps ONLY
# its best checkpoint (--no-epoch-saves). Run make_splits.sh FIRST.
#
# ===== PARAMETER SPACE (edit here) ============================================
SEEDS=(42 7 123 2024 31337)        # must match make_splits.sh
LOSS="${LOSS:-ce}"        # loss held fixed across members
EPOCHS="${EPOCHS:-5}"              # epochs held fixed across members
# =============================================================================
set -uo pipefail
cd /workspace/esg_contest
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE2=exp/integrated_stage_predictions/0615/ensemble/stage2
NAME=mix_a2_b3_add_val
RESULTS="$STAGE2/ensemble/train_results"
mkdir -p "$RESULTS"

echo "### ensemble train ST2: seeds=[${SEEDS[*]}] loss=$LOSS epochs=$EPOCHS GPU=$CUDA_VISIBLE_DEVICES"
for seed in "${SEEDS[@]}"; do
  SPLIT="$STAGE2/data/ensemble/seed${seed}"
  TRAIN="$SPLIT/${NAME}.train.json"
  VAL="$SPLIT/${NAME}.val.json"
  for f in "$TRAIN" "$VAL"; do
    [ -f "$f" ] || { echo "[error] missing split: $f (run make_splits.sh first)" >&2; exit 1; }
  done
  echo "=== [member seed=$seed] $(date +%T) ==="
  .venv/bin/python core/train/train_bert.py \
    --model large --stage st2 \
    --train-path "$TRAIN" --val-path "$VAL" \
    --epochs "$EPOCHS" --max-len 512 \
    --loss "$LOSS" \
    --seed "$seed" \
    --model-dir "/models/ensemble_st2_mix_a2_b3_seed${seed}" \
    --batch-size 4 --no-epoch-saves \
    --output "$RESULTS/${NAME}_seed${seed}.json" \
    2>&1 | tee "$RESULTS/seed${seed}.log"
  echo "=== [member seed=$seed] rc=${PIPESTATUS[0]} ==="
done
echo "### ensemble train DONE $(date +%T)"
