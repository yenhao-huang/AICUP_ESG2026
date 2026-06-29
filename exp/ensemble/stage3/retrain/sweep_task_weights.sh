#!/usr/bin/env bash
# Run from repo root:
#   bash exp/integrated_stage_predictions/0614/submit/submit_5/stage3/retrain/sweep_task_weights.sh
#
# Hyperparameter search over ST3 multitask --task-weights (st1,st2,st3). Everything
# else is held fixed (loss modes, --st3-class-weights 1,8,30, epochs, data) so the
# only varying factor is the per-task loss weighting. Each combo trains a fresh
# model selected by ST3 val Macro-F1 on the pre-split val (split_train_val_by_class.py
# with --force-val-classes Misleading; the ST3 head therefore trains on Clear /
# Not Clear only). Checkpoints -> /models/ per CLAUDE.md; per-combo result JSONs
# stay in retrain/results/. Collect with collect_results.py.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

D=exp/integrated_stage_predictions/0614/submit/submit_5/stage3
RETRAIN=$D/retrain
TRAIN=$D/data/vpesg_4k_train_1000_add_val.train.json
VAL=$D/data/vpesg_4k_train_1000_add_val.val.json
MODEL_ROOT=/models/submit_5_mt_st123_twsweep
mkdir -p "$RETRAIN/results" "$RETRAIN/logs"

# st1,st2,st3 task-weight grid (baseline first).
GRID=(
  "0.2,0.3,0.5"
  "0.15,0.25,0.6"
  "0.1,0.2,0.7"
  "0.1,0.1,0.8"
  "0.2,0.35,0.45"
  "0.33,0.33,0.34"
)

for TW in "${GRID[@]}"; do
  TAG="w$(echo "$TW" | tr ',.' '__')"          # e.g. 0.2,0.3,0.5 -> w0_2_0_3_0_5
  MD="$MODEL_ROOT/$TAG"
  OUT="$RETRAIN/results/$TAG.json"
  echo "==================== task-weights=$TW  tag=$TAG ===================="
  PYTHONUNBUFFERED=1 .venv/bin/python core/train/train_multitaskbert_stage3.py \
    --model large \
    --train-path "$TRAIN" \
    --val-path   "$VAL" \
    --model-dir  "$MD" \
    --st1-loss weighted_ce --st2-loss weighted_ce \
    --st3-loss weighted_ce --st3-class-weights 1,8,30 \
    --task-weights "$TW" \
    --batch-size 4 --grad-accum 4 \
    --epochs 10 --no-epoch-saves \
    --output "$OUT" \
    2>&1 | tee "$RETRAIN/logs/$TAG.log"
done

echo "==================== sweep done — collecting ===================="
.venv/bin/python "$RETRAIN/collect_results.py" --results-dir "$RETRAIN/results"
