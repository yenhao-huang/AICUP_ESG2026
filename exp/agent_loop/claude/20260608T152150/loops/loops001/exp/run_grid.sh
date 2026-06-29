#!/usr/bin/env bash
# Loop001 train_grid: A1..A4 x B1..B3 (12 arms) + manual_ce ablation.
# A0 already trained separately. ST1 large, --no-epoch-saves, cuda default(cuda:1).
set -u
cd /workspace/esg_contest
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
    --no-epoch-saves \
    --output "$RES/${arm}.json" \
    "$@" \
    > "$LOG/${arm}.log" 2>&1
  echo "=== END   $arm $(date +%T) rc=$? ==="
}

for src in a1 a2 a3 a4; do
  for b in b1 b2 b3; do
    train "${src}_${b}" "$MIX/mix_${src}_${b}.json"
  done
done

echo "ALL GRID ARMS DONE $(date +%T)"
