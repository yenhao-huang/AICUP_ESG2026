#!/usr/bin/env bash
# Run the submit prediction pipeline and build a final submission.csv.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-submit}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-results/submit/$RUN_ID}"
DATA="${DATA:-data/raw_data/vpesg4k_test_2000.json}"
PYTHON="${PYTHON:-.venv/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

run_cmd() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

mkdir -p "$OUT_DIR"/stage{1,2,3,4}

STAGE1_BERT="$OUT_DIR/stage1/bert_softvote.csv"
STAGE1_GEMMA="$OUT_DIR/stage1/gemma.csv"
STAGE1_FINAL="$OUT_DIR/stage1/final.csv"

STAGE2_BERT="$OUT_DIR/stage2/bert_softvote.csv"
STAGE2_GEMMA="$OUT_DIR/stage2/gemma.csv"
STAGE2_RAW="$OUT_DIR/stage2/bert_gemma_raw.csv"
STAGE2_FINAL="$OUT_DIR/stage2/final.csv"

STAGE3_RAW="$OUT_DIR/stage3/raw.csv"
STAGE3_FINAL="$OUT_DIR/stage3/final.csv"

STAGE4_RAW_DIR="$OUT_DIR/stage4/codex_raw"
STAGE4_RAW="$STAGE4_RAW_DIR/stage4_codex_predictions.csv"
STAGE4_FINAL="$OUT_DIR/stage4/final.csv"

SUBMISSION="$OUT_DIR/submission.csv"
SUBMISSION_LATEST="${SUBMISSION_LATEST:-results/submit/submission.csv}"

echo "### submit pipeline"
echo "mode=$MODE"
echo "run_id=$RUN_ID"
echo "out_dir=$OUT_DIR"
echo "data=$DATA"

run_cmd env MODE="$MODE" DATA="$DATA" OUTPUT="$STAGE1_BERT" RUN_ID="softvote_st1_submit" bash scripts/predict/predict_ensemble_model_for_stage1.sh
run_cmd env MODE="$MODE" DATA="$DATA" OUTPUT="$STAGE2_BERT" RUN_ID="softvote_st2_submit" bash scripts/predict/predict_ensemble_model_for_stage2.sh
run_cmd env MODE="$MODE" DATA="$DATA" STAGE=all STAGE1_OUTPUT="$STAGE1_GEMMA" STAGE2_OUTPUT="$STAGE2_GEMMA" bash scripts/predict/predict_gemma_fallback_model.sh

run_cmd "$PYTHON" core/service/utils/submit_pipeline.py merge-stage1 \
  --bert "$STAGE1_BERT" \
  --gemma "$STAGE1_GEMMA" \
  --output "$STAGE1_FINAL" \
  --threshold 0.6 \
  --run-id submit_stage1 \
  --bert-source bert

run_cmd "$PYTHON" core/service/utils/submit_pipeline.py merge-stage2 \
  --bert "$STAGE2_BERT" \
  --gemma "$STAGE2_GEMMA" \
  --output "$STAGE2_RAW" \
  --threshold 0.7

run_cmd "$PYTHON" core/service/utils/submit_pipeline.py gate-stage2 \
  --stage1 "$STAGE1_FINAL" \
  --stage2 "$STAGE2_RAW" \
  --output "$STAGE2_FINAL"

run_cmd env MODE="$MODE" DATA="$DATA" OUTPUT="$STAGE3_RAW" bash scripts/predict/predict_multitaskbert_for_stage3.sh
run_cmd "$PYTHON" core/service/utils/submit_pipeline.py gate-stage3 \
  --stage1 "$STAGE1_FINAL" \
  --stage2 "$STAGE2_FINAL" \
  --stage3 "$STAGE3_RAW" \
  --output "$STAGE3_FINAL"

run_cmd env DATA="$DATA" OUT_DIR="$STAGE4_RAW_DIR" bash scripts/predict/predict_codex_for_stage4.sh
run_cmd "$PYTHON" core/service/utils/submit_pipeline.py gate-stage4 \
  --stage1 "$STAGE1_FINAL" \
  --stage4 "$STAGE4_RAW" \
  --output "$STAGE4_FINAL"

run_cmd "$PYTHON" core/service/utils/submit_pipeline.py build-submission \
  --stage1 "$STAGE1_FINAL" \
  --stage2 "$STAGE2_FINAL" \
  --stage3 "$STAGE3_FINAL" \
  --stage4 "$STAGE4_FINAL" \
  --output "$SUBMISSION"

run_cmd cp "$SUBMISSION" "$SUBMISSION_LATEST"

echo "submission=$SUBMISSION"
echo "submission_latest=$SUBMISSION_LATEST"
