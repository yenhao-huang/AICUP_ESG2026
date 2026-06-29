#!/usr/bin/env bash
# Loop001 grid re-run on GPU0 ONLY (GPU1 is busy with exp32).
# Forces cuda:0 via CUDA_VISIBLE_DEVICES, batch 4, expandable_segments to fit ~9GB free.
# Skips a0_real_only + a1_b1 (already have valid result json).
set -u
cd /workspace/esg_contest
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
MIX=data/generated/loop001_canonical_mix
RES=exp/agent_loop/claude/20260608T152150/loops/loops001/exp/train_results
LOG=exp/agent_loop/claude/20260608T152150/loops/loops001/exp/logs
mkdir -p "$RES" "$LOG"

train () {
  local arm="$1"; shift
  local mixfile="$1"; shift
  echo "=== START $arm $(date +%T) ==="
  $PY core/train/train_bert.py \
    --model large --stage st1 \
    --train-path "$mixfile" \
    --model-dir "models/loop001_st1_${arm}" \
    --batch-size 4 \
    --no-epoch-saves \
    --output "$RES/${arm}.json" \
    "$@" \
    > "$LOG/${arm}.log" 2>&1
  echo "=== END   $arm $(date +%T) rc=$? ==="
}

# Re-run every arm that lacks a complete result json (a0, a1_b1 kept).
for arm in a1_b2 a1_b3 a2_b1 a2_b2 a2_b3 a3_b1 a3_b2 a3_b3 a4_b1 a4_b2 a4_b3; do
  train "$arm" "$MIX/mix_${arm}.json"
done

echo "GPU0 GRID RERUN DONE $(date +%T)"
