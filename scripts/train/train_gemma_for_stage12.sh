#!/usr/bin/env bash
# Train the Gemma ST1+ST2 QLoRA adapter from the stage12 train split.
#
# Run from anywhere:
#   bash scripts/train/train_gemma_for_stage12.sh
#
# Skip training and reuse the existing adapter:
#   SKIP_TRAIN=1 bash scripts/train/train_gemma_for_stage12.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ===== PARAMETER SPACE (override with env vars) ===============================
PY="${PY:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/train/gemma4_st12_mix.yml}"
ADAPTER="${ADAPTER:-models/gemma/gemma4_st12_mix}"
LOG_DIR="${LOG_DIR:-logs/train/gemma/stage12}"
TRAIN_LOG="${TRAIN_LOG:-$LOG_DIR/train.log}"
DRY_RUN="${DRY_RUN:-0}"
# =============================================================================

if [ ! -x "$PY" ]; then
  PY="${PYTHON_FALLBACK:-python3}"
fi

[ -f "$CONFIG" ] || { echo "[error] missing config: $CONFIG" >&2; exit 1; }
mkdir -p "$LOG_DIR"

echo "### train gemma stage12"
echo "config=$CONFIG"
echo "adapter=$ADAPTER"
echo "train_log=$TRAIN_LOG"

if [[ "$DRY_RUN" == "1" ]]; then
  "$PY" core/service/train/train_gemma4.py --config "$CONFIG" --dry-run --num-show "${NUM_SHOW:-3}"
  exit 0
fi

# 1. Train on the .train split.
if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo "[run] SKIP_TRAIN=1 -> evaluate existing adapter at $ADAPTER"
else
  echo "[run] starting QLoRA SFT (logging to $TRAIN_LOG) ..."
  "$PY" core/service/train/train_gemma4.py --config "$CONFIG" 2>&1 | tee "$TRAIN_LOG"
  echo "[run] train done -> $ADAPTER"
fi
