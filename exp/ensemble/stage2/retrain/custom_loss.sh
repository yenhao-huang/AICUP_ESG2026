# Run from repo root:
#   bash exp/integrated_stage_predictions/0614/submit/submit_5/stage2/retrain/custom_loss.sh
#
# LOSS-function sweep for ST2 on the pre-split mix_a2_b3+val data. Targets the
# evidence_status Yes:No ~2.91:1 imbalance (No-evidence F1 is the weak spot).
# Trains one model per config below, each keeping ONLY its best checkpoint
# (--no-epoch-saves), into its own /models dir + train_results json so runs never
# overwrite each other or the base /models/esg_contest/loop001_st2_synth_A2_b3_c1_25_add_val.
#
# Pre-split files are produced separately by split_train_val_by_class.py:
#   .venv/bin/python $STAGE2/split_train_val_by_class.py \
#     --input $STAGE2/data/mix_a2_b3_add_val.json \
#     --train-out $STAGE2/data/mix_a2_b3_add_val.train.json \
#     --val-out   $STAGE2/data/mix_a2_b3_add_val.val.json \
#     --label-key evidence_status --val-ratio 0.2 --seed 42
#
# ===== PARAMETER SPACE (edit here) ============================================
# One config per line:  tag | loss | class_weights | focal_gamma | asl_gneg | asl_gpos
#   loss          : weighted_ce | ce | manual_ce | focal | asl
#   class_weights : per-class, index order No=0,Yes=1 (e.g. 2.9,1.0); blank = inverse-freq
#   focal_gamma   : only for loss=focal      (blank -> 2.0)
#   asl_gneg/gpos : only for loss=asl        (blank -> 4.0 / 1.0)
CONFIGS=(
  "wce            | weighted_ce |          |     |     |    "   # base recipe: inverse-freq weighted CE
  "ce             | ce          |          |     |     |    "   # plain CE (no reweighting)
  "mce_2_9_1      | manual_ce   | 2.9,1.0  |     |     |    "   # manual: No x2.9 (matches raw ratio)
  "mce_4_1        | manual_ce   | 4.0,1.0  |     |     |    "   # manual: No x4 (push recall on No)
  "mce_6_1        | manual_ce   | 6.0,1.0  |     |     |    "   # manual: No x6 (harder push)
  "focal_g2       | focal       |          | 2.0 |     |    "   # focal, inv-freq alpha, gamma 2
  "focal_g2_w4    | focal       | 4.0,1.0  | 2.0 |     |    "   # focal + manual alpha
  "focal_g3_w4    | focal       | 4.0,1.0  | 3.0 |     |    "   # harder focusing
  "asl_n4_p1      | asl         |          |     | 4.0 | 1.0"   # asymmetric focal loss
)
EPOCHS="${EPOCHS:-10}"   # held fixed across loss configs (override via env)
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

trim() { echo "$1" | xargs; }   # strip surrounding whitespace

echo "### custom_loss sweep: ${#CONFIGS[@]} configs  epochs=$EPOCHS  GPU=$CUDA_VISIBLE_DEVICES"
for line in "${CONFIGS[@]}"; do
  IFS='|' read -r tag loss cw fg agn agp <<< "$line"
  tag=$(trim "$tag"); loss=$(trim "$loss"); cw=$(trim "$cw")
  fg=$(trim "$fg"); agn=$(trim "$agn"); agp=$(trim "$agp")

  EXTRA=()
  [ -n "$cw" ]            && EXTRA+=(--class-weights "$cw")
  [ "$loss" = "focal" ]  && EXTRA+=(--focal-gamma "${fg:-2.0}")
  [ "$loss" = "asl" ]    && EXTRA+=(--asl-gamma-neg "${agn:-4.0}" --asl-gamma-pos "${agp:-1.0}")

  echo "=== [$tag] loss=$loss cw='${cw:-inv-freq}' ${EXTRA[*]} $(date +%T) ==="
  .venv/bin/python core/train/train_bert.py \
    --model large --stage st2 \
    --train-path "$TRAIN" --val-path "$VAL" \
    --epochs "$EPOCHS" --seed 42 --max-len 512 \
    --loss "$loss" "${EXTRA[@]}" \
    --model-dir "/models/esg_contest/loop001_st2_synth_A2_b3_c1_25_add_val_${tag}" \
    --batch-size 4 --no-epoch-saves \
    --output "$STAGE2/train_results/mix_a2_b3_add_val_${tag}.json" \
    2>&1 | tee "$STAGE2/train_results/${tag}.log"
  echo "=== [$tag] rc=${PIPESTATUS[0]} ==="
done
echo "### custom_loss sweep DONE $(date +%T)"
