#!/usr/bin/env bash
set -u
cd /workspace/esg_contest
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
EXP=exp/agent_loop/claude/20260608T152150/loops/loops001/exp
OUT=$EXP/valpublic_eval; LOG=$EXP/logs
mkdir -p "$OUT"
for arm in a0_real_only a1_b1 a1_b2 a1_b3 a2_b1 a2_b2 a2_b3 a3_b1 a3_b2 a3_b3 a4_b1 a4_b2 a4_b3; do
  dir="models/loop001_st1_${arm}"
  [ -f "$dir/best_st1.pt" ] || { echo "SKIP $arm"; continue; }
  $PY core/eval/eval_bert.py --model large --stage st1 --no-cascade \
    --model-dir "$dir" --pretrain-model hfl/chinese-roberta-wwm-ext-large \
    --data-path data/benchmarks/val_public.json --device cuda:0 \
    --output "$OUT/${arm}_valpublic.json" > "$LOG/eval_valpublic_${arm}.log" 2>&1
  echo "DONE $arm"
done
