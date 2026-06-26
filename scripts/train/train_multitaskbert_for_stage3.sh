#!/usr/bin/env bash
# Train Stage 3 multitask BERT from data/multitask_data/stage3.
#
# Run from anywhere:
#   bash scripts/train/train_multitaskbert_for_stage3.sh
#
# Expected input layout:
#   data/multitask_data/stage3/<dataset>/<dataset>.train.json
#   data/multitask_data/stage3/<dataset>/<dataset>.val.json
#
# Results:
#   results/train/multitaskbert/stage3/<dataset>/<run_name>.json
# Model checkpoints:
#   models/multitaskbert/stage3/<dataset>/<run_name>/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ===== PARAMETER SPACE (override with env vars) ===============================
DATASET="${DATASET:-vpesg_4k_train_1000_add_val}"
INPUT_DIR="${INPUT_DIR:-data/multitask_data/stage3/$DATASET}"
RESULTS_DIR="${RESULTS_DIR:-results/train/multitaskbert/stage3/$DATASET}"
MODELS_DIR="${MODELS_DIR:-models/multitaskbert/stage3/$DATASET}"
TRAINER="${TRAINER:-core/service/train/train_multitaskbert_stage3.py}"
PYTHON="${PYTHON:-.venv/bin/python}"
GPU="${GPU:-0}"
MODEL="${MODEL:-large}"
SEED="${SEED:-42}"

# Same ST3 multitask recipe as the prior mt_st123_w1_8_30_mlval runs.
EPOCHS="${EPOCHS:-10}"
MAX_LEN="${MAX_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
ST1_LOSS="${ST1_LOSS:-weighted_ce}"
ST2_LOSS="${ST2_LOSS:-weighted_ce}"
ST3_LOSS="${ST3_LOSS:-weighted_ce}"
ST3_CLASS_WEIGHTS="${ST3_CLASS_WEIGHTS:-1,8,30}"
TASK_WEIGHTS="${TASK_WEIGHTS:-0.2,0.3,0.5}"
RUN_NAME="${RUN_NAME:-mt_st123_w1_8_30_mlval_seed${SEED}_e${EPOCHS}}"
DRY_RUN="${DRY_RUN:-0}"
# =============================================================================

if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

TRAIN="$INPUT_DIR/${DATASET}.train.json"
VAL="$INPUT_DIR/${DATASET}.val.json"

[ -d "$INPUT_DIR" ] || { echo "[error] missing input dir: $INPUT_DIR" >&2; exit 1; }
[ -f "$TRAINER" ] || { echo "[error] missing trainer: $TRAINER" >&2; exit 1; }
[ -f "$TRAIN" ] || { echo "[error] missing train split: $TRAIN (run scripts/data/get_multitask_model_for_stage3.sh first)" >&2; exit 1; }
[ -f "$VAL" ] || { echo "[error] missing val split: $VAL (run scripts/data/get_multitask_model_for_stage3.sh first)" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_SYMLINKS_WARNING="${HF_HUB_DISABLE_SYMLINKS_WARNING:-1}"

mkdir -p "$RESULTS_DIR" "$MODELS_DIR"

MODEL_DIR="$MODELS_DIR/$RUN_NAME"
OUTPUT="$RESULTS_DIR/$RUN_NAME.json"
LOG="$RESULTS_DIR/$RUN_NAME.log"
mkdir -p "$MODEL_DIR"

cmd=(
  "$PYTHON" "$TRAINER"
  --model "$MODEL"
  --train-path "$TRAIN"
  --val-path "$VAL"
  --model-dir "$MODEL_DIR"
  --st1-loss "$ST1_LOSS"
  --st2-loss "$ST2_LOSS"
  --st3-loss "$ST3_LOSS"
  --st3-class-weights "$ST3_CLASS_WEIGHTS"
  --task-weights "$TASK_WEIGHTS"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --epochs "$EPOCHS"
  --max-len "$MAX_LEN"
  --seed "$SEED"
  --no-epoch-saves
  --output "$OUTPUT"
)

echo "### train multitaskbert stage3"
echo "train=$TRAIN"
echo "val=$VAL"
echo "trainer=$TRAINER"
echo "model_dir=$MODEL_DIR"
echo "output=$OUTPUT"
echo "log=$LOG"
echo "gpu=$CUDA_VISIBLE_DEVICES model=$MODEL seed=$SEED epochs=$EPOCHS max_len=$MAX_LEN batch_size=$BATCH_SIZE grad_accum=$GRAD_ACCUM"
echo "losses=st1:$ST1_LOSS st2:$ST2_LOSS st3:$ST3_LOSS st3_class_weights=$ST3_CLASS_WEIGHTS task_weights=$TASK_WEIGHTS"

if [ "$DRY_RUN" = "1" ]; then
  printf '[dry-run]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "### train multitaskbert stage3 DONE rc=$rc"
exit "$rc"
