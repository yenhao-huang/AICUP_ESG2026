#!/usr/bin/env bash
# Train Stage 2 ensemble BERT members from data/ensemble_data/stage2 splits.
#
# Run from anywhere:
#   bash scripts/train/train_ensemble_models_for_stage2.sh
#
# Expected input layout:
#   data/ensemble_data/stage2/<dataset>/seed<S>/<dataset>.train.json
#   data/ensemble_data/stage2/<dataset>/seed<S>/<dataset>.val.json
#
# Results:
#   results/train/ensemble/stage2/<dataset>/seed<S>/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ===== PARAMETER SPACE (override with env vars) ===============================
SEEDS=(${SEEDS:-42 7 123 2024 31337})
INPUT_DIR="${INPUT_DIR:-data/ensemble_data/stage2}"
RESULTS_DIR="${RESULTS_DIR:-results/train/ensemble/stage2}"
TRAINER="${TRAINER:-core/service/train/train_bert.py}"
PYTHON="${PYTHON:-.venv/bin/python}"
GPU="${GPU:-0}"
MODEL="${MODEL:-large}"
STAGE="${STAGE:-st2}"

# Same ST2 ensemble recipe as exp/ensemble/stage2/ensemble/train.sh.
LOSS="${LOSS:-ce}"
EPOCHS="${EPOCHS:-5}"
MAX_LEN="${MAX_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LOSS_TAG="${LOSS_TAG:-${LOSS}_e${EPOCHS}}"
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
echo "datasets=${#DATASET_DIRS[@]} seeds=[${SEEDS[*]}] loss=$LOSS epochs=$EPOCHS max_len=$MAX_LEN GPU=$CUDA_VISIBLE_DEVICES"

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
        echo "[error] missing split: $path (run scripts/data/get_ensemble_model_data_for_stage2.sh first)" >&2
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
      --epochs "$EPOCHS"
      --max-len "$MAX_LEN"
      --loss "$LOSS"
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
