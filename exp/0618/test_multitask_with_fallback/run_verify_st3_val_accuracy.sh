#!/usr/bin/env bash
# Run from repo root:
#   bash exp/integrated_stage_predictions/0618/test_multitask_with_fallback/run_verify_st3_val_accuracy.sh
#
# Verify the multitask ST3 checkpoint w0_2_0_3_0_5 accuracy on the copied val set.
# Reproduces (純3類有效子集):
#   Accuracy   = 86.0% (233/271)
#   Macro-F1   = 0.5209   (== training selection metric best_val_st3_f1)
#
# data-only compliance: inference reads ONLY the `data` field via
# pred_by_bert_multitask.py; the GT evidence_quality is used ONLY by the offline
# scorer, never as model input. No gate / no fallback here — this measures the
# bare multitask ST3 head.
set -euo pipefail
cd /workspace/esg_contest
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DIR=exp/integrated_stage_predictions/0618/test_multitask_with_fallback
DATA="$DIR/data/vpesg_4k_train_1000_add_val.val.json"
CKPT=/models/submit_5_mt_st123_twsweep/w0_2_0_3_0_5/best_multitask_st3.pt
PRED="$DIR/preds/st3_val_pred.csv"
METRICS="$DIR/preds/st3_val_metrics.json"
mkdir -p "$DIR/preds"

echo "### [1/2] inference (multitask ST3 head, data-only) ###"
.venv/bin/python core/human/predict/stage3/pred_by_bert_multitask.py \
  --data           "$DATA" \
  --output         "$PRED" \
  --finetune-path  "$CKPT" \
  --model          hfl/chinese-roberta-wwm-ext-large \
  --device         cuda

echo "### [2/2] score (accuracy + Macro-F1) ###"
.venv/bin/python "$DIR/score_st3_subset.py" \
  --gold "$DATA" --pred "$PRED" --output "$METRICS"
