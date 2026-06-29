# Run from repo root: bash exp/integrated_stage_predictions/0614/submit/submit_5/stage3/retrain.sh
# ST3-focused multitask BERT (joint ST1+ST2+ST3) on vpesg_4k_train_1000 + vpesg4k_val_1000,
# using the pre-split stratified train/val files (split_train_val_by_class.py with
# --force-val-classes Misleading, so the ST3 head trains on Clear / Not Clear only
# and the scarce Misleading rows are kept as a held-out signal). best_multitask_st3.pt
# is picked by ST3 val Macro-F1. Checkpoint -> /models/ per CLAUDE.md.
.venv/bin/python core/train/train_multitaskbert_stage3.py \
  --model large \
  --train-path exp/integrated_stage_predictions/0614/submit/submit_5/stage3/data/vpesg_4k_train_1000_add_val.train.json \
  --val-path   exp/integrated_stage_predictions/0614/submit/submit_5/stage3/data/vpesg_4k_train_1000_add_val.val.json \
  --model-dir  /models/submit_5_mt_st123_w1_8_30_mlval \
  --st1-loss weighted_ce --st2-loss weighted_ce \
  --st3-loss weighted_ce --st3-class-weights 1,8,30 \
  --task-weights 0.2,0.3,0.5 \
  --batch-size 4 --grad-accum 4 \
  --epochs 10 --no-epoch-saves \
  --output exp/integrated_stage_predictions/0614/submit/submit_5/stage3/data/mt_st123_w1_8_30_mlval_train.json
