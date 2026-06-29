# Run from repo root:
#   bash exp/integrated_stage_predictions/0614/submit/submit_5/stage2/retrain.sh
#
# Retrain ST2 BERT on mix_a2_b3 + vpesg4k_val_1000 (val folded INTO training),
# reproducing the recipe behind /models/esg_contest/loop001_st2_synth_A2_b3_c1_25/best_st2.pt.
#
# Original recipe (results/train/loop001_st2_A2_b3_c1_25.json):
#   --model large (hfl/chinese-roberta-wwm-ext-large, batch 8 / grad_accum 2 / lr 1e-5)
#   --stage st2, epochs 5, seed 42, max_len 512, loss weighted_ce (inverse-freq, default)
#   train = mix_a2_b3.json (1362 st2 rows: No 411 / Yes 951)
#   val   = vpesg4k_val_1000.json (813 st2 rows: No 145 / Yes 668), best by val_macro_f1
#
# The stratified held-out split is produced separately by split_train_val_by_class.py
# (val is part of TRAIN here, so best_st2.pt is selected on that pre-cut slice).
# This script only TRAINS the pre-split files. Run the split first:
#   .venv/bin/python $STAGE2/split_train_val_by_class.py \
#     --input $STAGE2/data/mix_a2_b3_add_val.json \
#     --train-out $STAGE2/data/mix_a2_b3_add_val.train.json \
#     --val-out   $STAGE2/data/mix_a2_b3_add_val.val.json \
#     --label-key evidence_status --val-ratio 0.2 --seed 42
# Checkpoint -> /models/ per CLAUDE.md.
set -uo pipefail
cd /workspace/esg_contest
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE2=exp/integrated_stage_predictions/0614/submit/submit_5/stage2
TRAIN="$STAGE2/data/mix_a2_b3_add_val.train.json"
VAL="$STAGE2/data/mix_a2_b3_add_val.val.json"
MODEL_DIR="/models/esg_contest/loop001_st2_synth_A2_b3_c1_25_add_val"
mkdir -p "$STAGE2/train_results"

for f in "$TRAIN" "$VAL"; do
  [ -f "$f" ] || { echo "[error] missing split file: $f (run split_train_val_by_class.py first)" >&2; exit 1; }
done

# Train ST2 BERT with the loop001 recipe (large / epochs 5 / seed 42 / weighted_ce).
# batch_size, grad_accum, lr inherited from configs/train/bert.yml `large` (8 / 2 / 1e-5).
echo "### train st2 -> $MODEL_DIR/best_st2.pt $(date +%T)"
.venv/bin/python core/train/train_bert.py \
  --model large --stage st2 \
  --train-path "$TRAIN" --val-path "$VAL" \
  --epochs 10 --seed 42 --max-len 512 \
  --loss weighted_ce \
  --no-epoch-saves \
  --model-dir "$MODEL_DIR" \
  --output "$STAGE2/train_results/mix_a2_b3_add_val.json" \
  2>&1 | tee "$STAGE2/train_results/retrain.log"
echo "### DONE rc=${PIPESTATUS[0]} -> $MODEL_DIR/best_st2.pt $(date +%T)"
