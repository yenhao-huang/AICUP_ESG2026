#!/usr/bin/env bash
# Train Stage 1 ensemble BERT members from data/ensemble_data/stage1 splits.
#
# Run from anywhere:
#   bash scripts/train/train_ensemble_models.sh
#
# Expected input layout:
#   data/ensemble_data/stage1/<dataset>/seed<S>/<dataset>.train.json
#   data/ensemble_data/stage1/<dataset>/seed<S>/<dataset>.val.json
#
# Results:
#   results/train/ensemble/<dataset>/seed<S>/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ===== PARAMETER SPACE (override with env vars) ===============================
SEEDS=(${SEEDS:-42 7 123 2024 31337})
INPUT_DIR="${INPUT_DIR:-data/ensemble_data/stage1}"
RESULTS_DIR="${RESULTS_DIR:-results/train/ensemble}"
TRAINER="${TRAINER:-core/service/train/train_bert.py}"
PYTHON="${PYTHON:-.venv/bin/python}"
GPU="${GPU:-1}"
MODEL="${MODEL:-large}"
STAGE="${STAGE:-st1}"

# Same ST1 ensemble loss recipe as exp/ensemble/stage1/ensemble/train.sh.
LOSS="${LOSS:-focal}"
CLASS_WEIGHTS="${CLASS_WEIGHTS:-4.0,1.0}"
FOCAL_GAMMA="${FOCAL_GAMMA:-3.0}"
LOSS_TAG="${LOSS_TAG:-focal_g3_w4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DRY_RUN="${DRY_RUN:-0}"
# =============================================================================

if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

[ -d "$INPUT_DIR" ] || { echo "[error] missing input dir: $INPUT_DIR" >&2; exit 1; }
[ -f "$TRAINER" ] || { echo "[error] missing trainer: $TRAINER" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

shopt -s nullglob
DATASET_DIRS=("$INPUT_DIR"/*)
shopt -u nullglob

[ "${#DATASET_DIRS[@]}" -gt 0 ] || {
  echo "[error] no dataset dirs found under: $INPUT_DIR" >&2
  exit 1
}

echo "### ensemble train $STAGE"
echo "input_dir=$INPUT_DIR results_dir=$RESULTS_DIR trainer=$TRAINER"
echo "datasets=${#DATASET_DIRS[@]} seeds=[${SEEDS[*]}] loss=$LOSS($LOSS_TAG) cw=$CLASS_WEIGHTS gamma=$FOCAL_GAMMA GPU=$CUDA_VISIBLE_DEVICES"

for dataset_dir in "${DATASET_DIRS[@]}"; do
  [ -d "$dataset_dir" ] || continue
  dataset="$(basename "$dataset_dir")"
  echo "=== [dataset=$dataset] $(date +%T) ==="

  for seed in "${SEEDS[@]}"; do
    split="$dataset_dir/seed${seed}"
    train="$split/${dataset}.train.json"
    val="$split/${dataset}.val.json"
    for path in "$train" "$val"; do
      [ -f "$path" ] || {
        echo "[error] missing split: $path (run scripts/data/get_ensemble_model_data_for_stage1.sh first)" >&2
        exit 1
      }
    done

    member_results="$RESULTS_DIR/$dataset/seed${seed}"
    model_dir="$member_results/models/${LOSS_TAG}"
    output="$member_results/${dataset}_${LOSS_TAG}_seed${seed}.json"
    log="$member_results/${LOSS_TAG}_seed${seed}.log"
    mkdir -p "$member_results" "$model_dir"

    cmd=(
      "$PYTHON" "$TRAINER"
      --model "$MODEL"
      --stage "$STAGE"
      --train-path "$train"
      --val-path "$val"
      --loss "$LOSS"
      --class-weights "$CLASS_WEIGHTS"
      --focal-gamma "$FOCAL_GAMMA"
      --seed "$seed"
      --model-dir "$model_dir"
      --batch-size "$BATCH_SIZE"
      --no-epoch-saves
      --output "$output"
    )

    echo "--- [dataset=$dataset seed=$seed] $(date +%T) ---"
    echo "train=$train"
    echo "val=$val"
    echo "output=$output"
    echo "model_dir=$model_dir"

    if [ "$DRY_RUN" = "1" ]; then
      printf '[dry-run]'
      printf ' %q' "${cmd[@]}"
      printf '\n'
      continue
    fi

    "${cmd[@]}" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    echo "--- [dataset=$dataset seed=$seed] rc=$rc ---"
    [ "$rc" -eq 0 ] || exit "$rc"
  done
done

echo "### ensemble train DONE $(date +%T)"
