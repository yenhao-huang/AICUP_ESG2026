#!/usr/bin/env bash
# Run ST4 prompt variants on human_check/data/test_preserved.json
# Variants: boundary_rules_v4, A4_rules_only, A3_cot, A2_8shot, stage4_balance_rule_v1, vanilla_old_anchor2024_system

set -euo pipefail

ROOT="/workspace/esg_contest"
SCRIPT="$ROOT/core/human/predict/stage4/pred_by_codex.py"
DATA="$ROOT/exp/agent_loop/claude/20260609T172829/human_check/data/test_preserved.json"
OUT_BASE="$ROOT/exp/agent_loop/claude/20260609T172829/human_check/results"
PROMPT_DIR="$ROOT/configs/prompt/stage4/codex"
MODEL="gpt-5.5"
PYTHON="$ROOT/.venv/bin/python"

declare -A VARIANTS=(
  [boundary_rules_v4]="boundary_rules_v4.txt"
  [A4_rules_only]="boundary_rules_v4.txt"
  [A3_cot]="few_shot_cot_v3.txt"
  [A2_8shot]="few_shot_boundary_v2.txt"
  [stage4_balance_rule_v1]="stage4_balance_rule_v1.txt"
  [vanilla_old_anchor2024_system]="vanilla_old_anchor2024_system.txt"
)

for VARIANT in boundary_rules_v4 A4_rules_only A3_cot A2_8shot stage4_balance_rule_v1 vanilla_old_anchor2024_system; do
  PROMPT_FILE="${VARIANTS[$VARIANT]}"
  OUT_DIR="$OUT_BASE/$VARIANT"
  mkdir -p "$OUT_DIR"

  echo "========================================"
  echo "Running: $VARIANT  ($PROMPT_FILE)"
  echo "========================================"

  $PYTHON "$SCRIPT" \
    --data "$DATA" \
    --stage1-csv "$DATA" \
    --stage1-gate-col promise_status \
    --prompt-path "$PROMPT_DIR/$PROMPT_FILE" \
    --model "$MODEL" \
    --raw-output-dir "$OUT_DIR/raw" \
    --token-usage-output "$OUT_DIR/token_usage.jsonl" \
    --output "$OUT_DIR/predictions.csv" \
    2>&1 | tee "$OUT_DIR/run.log"

  echo "Done: $OUT_DIR/predictions.csv"
  echo ""
done

echo "All variants complete. Results under:"
echo "  $OUT_BASE"
