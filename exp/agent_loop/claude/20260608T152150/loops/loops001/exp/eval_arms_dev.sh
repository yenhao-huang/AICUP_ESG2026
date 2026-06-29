#!/usr/bin/env bash
# Dev-split (test.json) ST1 eval for every trained loop001 arm. Selection ONLY.
set -u
cd /workspace/esg_contest
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
EXP=exp/agent_loop/claude/20260608T152150/loops/loops001/exp
DEVOUT=$EXP/dev_eval
LOG=$EXP/logs
mkdir -p "$DEVOUT"

eval_dev () {
  local arm="$1"
  local dir="models/loop001_st1_${arm}"
  [ -f "$dir/best_st1.pt" ] || { echo "SKIP $arm (no ckpt)"; return; }
  $PY core/eval/eval_bert.py \
    --model large --stage st1 --no-cascade \
    --model-dir "$dir" \
    --pretrain-model hfl/chinese-roberta-wwm-ext-large \
    --data-path data/benchmarks/test.json \
    --device cuda:0 \
    --output "$DEVOUT/${arm}_dev.json" \
    > "$LOG/eval_dev_${arm}.log" 2>&1
  echo "DONE $arm"
}

ARMS="a0_real_only a1_b1 a1_b2 a1_b3 a2_b1 a2_b2 a2_b3 a3_b1 a3_b2 a3_b3 a4_b1 a4_b2 a4_b3"
for a in $ARMS; do eval_dev "$a"; done
echo "ALL DEV EVAL DONE"
