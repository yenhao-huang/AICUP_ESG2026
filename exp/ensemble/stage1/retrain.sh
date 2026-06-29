# Run from repo root: bash exp/integrated_stage_predictions/0614/submit/submit_5/stage1/retrain.sh
# Train ST1 BERT on a3_b1 + vpesg4k_val_1000, using the pre-split stratified
# train/val files (split_train_val_by_class.py) so best_st1.pt is picked by a
# fixed, reproducible held-out val Macro-F1. Checkpoint -> /models/ per CLAUDE.md.
.venv/bin/python core/train/train_bert.py \
  --model large --stage st1 \
  --train-path exp/integrated_stage_predictions/0614/submit/submit_5/stage1/data/a3_b1_add_val.train.json \
  --val-path exp/integrated_stage_predictions/0614/submit/submit_5/stage1/data/a3_b1_add_val.val.json \
  --model-dir /models/submit_5_a3_b1_add_val \
  --batch-size 4  \
  --output exp/integrated_stage_predictions/0614/submit/submit_5/stage1/train_results/a3_b1_add_val.json
