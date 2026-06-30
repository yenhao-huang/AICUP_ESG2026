#!/usr/bin/env bash
# Stage1 + Stage2 ONLY (ST3 / ST4 / merge removed) — for tuning the ST1/ST2
# Gemma fallback rate in isolation.
#
#   ST1 = soft-vote over the 5 ST1 ensemble members (focal_g3_w4)
#         -> Gemma fallback: rows with max-softmax < ST1_CONF (0.70, ~13.9%, ~279 rows)
#            re-predicted by Gemma-4-12B (adapter gemma4_st12_mix)
#   ST2 = soft-vote over the 5 ST2 members
#         -> Gemma fallback: rows with max-softmax < ST2_TAU (0.90, ~12% of active)
#            re-predicted by Gemma (backend=local, --no-rag)
#         -> gated by ST1 (ST1=No -> N/A)
#
# Data-use: both fallbacks read ONLY the raw `data` field (Gemma prompt uses
# `data`; --no-rag so no retrieval). gemma4_st12_mix is a data-only st1+st2
# JSON-generative adapter (input_field="data").
#
# Outputs: stage1/bert_gemma.csv, stage2/softvote_gated.csv. No merge / submission.csv
# here — this script is for ST1/ST2 fallback-rate experiments only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"   # -> repo root /workspace/esg_contest
cd "$ROOT"
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv/bin/python"
SCRIPTS="$HERE/scripts"

DATA="${DATA:-$ROOT/data/raw_data/vpesg4k_test_2000.json}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# ----- ensemble member checkpoints (quoted so bash does NOT expand the glob;
#       soft_vote.py expands it internally) -----
ST1_GLOB="${ST1_GLOB:-/models/ensemble_st1_a3_b1_focal_g3_w4_seed*/best_st1.pt}"
ST2_GLOB="${ST2_GLOB:-/models/ensemble_st2_mix_a2_b3_seed*/best_st2.pt}"

# ----- Gemma fallback config -----
GEMMA_BASE="${GEMMA_BASE:-/models/unsloth-gemma-4-12b}"
GEMMA_ADAPTER="${GEMMA_ADAPTER:-/models/gemma4_st12_mix}"
GEMMA_MAX_NEW_TOKENS="${GEMMA_MAX_NEW_TOKENS:-256}"
DEVICE_GEMMA="${DEVICE_GEMMA:-$DEVICE}"
ST1_CONF="${ST1_CONF:-0.70}"   # ST1 BERT max-softmax below this -> Gemma (~13.9%, ~279 rows)
ST2_TAU="${ST2_TAU:-0.90}"     # ST2 BERT max-softmax below this -> Gemma (~12% of active)
RUN_ID="${RUN_ID:-submit_13}"

mkdir -p "$HERE/stage1/tmp" "$HERE/stage2/tmp" "$HERE/stage2/raw" "$HERE/logs"

ST1_RAW="$HERE/stage1/tmp/softvote_raw.csv"
ST1_OUT="$HERE/stage1/bert_gemma.csv"
ST2_RAW="$HERE/stage2/tmp/softvote_raw.csv"
ST2_HYBRID="$HERE/stage2/tmp/bert_gemma_raw.csv"
ST2_OUT="$HERE/stage2/softvote_gated.csv"

# ---------------- Stage1: soft vote -> Gemma fallback ----------------
echo "[$(date '+%H:%M:%S')] START st1 soft-vote"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage1/soft_vote.py \
    --data "$DATA" --ckpt-glob "$ST1_GLOB" \
    --output "$ST1_RAW" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    2>&1 | tee "$HERE/logs/st1_softvote.log"
echo "[$(date '+%H:%M:%S')] DONE  st1 soft-vote"

echo "[$(date '+%H:%M:%S')] START st1 gemma fallback (conf<$ST1_CONF)"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage1/pred_by_bert_gemma.py \
    --bert-pred "$ST1_RAW" --data "$DATA" --output "$ST1_OUT" \
    --gemma-base "$GEMMA_BASE" --gemma-adapter "$GEMMA_ADAPTER" \
    --device "$DEVICE_GEMMA" --conf-threshold "$ST1_CONF" \
    --max-new-tokens "$GEMMA_MAX_NEW_TOKENS" --run-id "${RUN_ID}_stage1" \
    2>&1 | tee "$HERE/logs/st1_gemma.log"
echo "[$(date '+%H:%M:%S')] DONE  st1 gemma fallback"

# ---------------- Stage2: soft vote -> Gemma fallback -> gate by ST1 ----------------
echo "[$(date '+%H:%M:%S')] START st2 soft-vote"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage2/soft_vote.py \
    --data "$DATA" --ckpt-glob "$ST2_GLOB" \
    --output "$ST2_RAW" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    2>&1 | tee "$HERE/logs/st2_softvote.log"
echo "[$(date '+%H:%M:%S')] DONE  st2 soft-vote"

echo "[$(date '+%H:%M:%S')] START st2 gemma fallback (tau<$ST2_TAU)"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage2/pred_by_bert_gemma.py \
    --bert-pred "$ST2_RAW" --data "$DATA" --output "$ST2_HYBRID" \
    --threshold "$ST2_TAU" --backend local --no-rag \
    --gemma-base "$GEMMA_BASE" --gemma-adapter "$GEMMA_ADAPTER" \
    --device "$DEVICE_GEMMA" --max-new-tokens "$GEMMA_MAX_NEW_TOKENS" \
    --run-id "${RUN_ID}_stage2" --raw-output-dir "$HERE/stage2/raw" \
    --token-usage-output "$HERE/stage2/tmp/token_usage.jsonl" \
    2>&1 | tee "$HERE/logs/st2_gemma.log"
echo "[$(date '+%H:%M:%S')] DONE  st2 gemma fallback"

PYTHONUNBUFFERED=1 "$VENV" "$SCRIPTS/apply_stage1_gate_to_stage2.py" \
    --stage1 "$ST1_OUT" --stage2 "$ST2_HYBRID" --output "$ST2_OUT" \
    2>&1 | tee "$HERE/logs/st2_gate.log"
echo "[$(date '+%H:%M:%S')] DONE  st2 gate"

echo "[$(date '+%H:%M:%S')] DONE  stage1+stage2 only"
echo "  ST1 -> $ST1_OUT"
echo "  ST2 -> $ST2_OUT"
