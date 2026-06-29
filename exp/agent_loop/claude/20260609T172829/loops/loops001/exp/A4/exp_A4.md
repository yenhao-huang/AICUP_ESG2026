# Experiment Record: A4 — boundary_rules_v4.txt (rules-only enhanced)

## Variant
- Prompt: `configs/prompt/stage4/codex/boundary_rules_v4.txt`
- Model: `gpt-5.5`
- Data: `data/benchmarks/val_public.json` (500 rows)
- Stage1 gate: `results/predict/stage1/bert_codex/stage1_bert_codex_rag_val.csv` (column: `promise_status`)

## Command Used (resume run from row 265)

```bash
cd /workspace/esg_contest && .venv/bin/python core/human/predict/stage4/pred_by_codex.py \
  --data data/benchmarks/val_public.json \
  --stage1-csv results/predict/stage1/bert_codex/stage1_bert_codex_rag_val.csv \
  --stage1-gate-col promise_status \
  --prompt-path configs/prompt/stage4/codex/boundary_rules_v4.txt \
  --model gpt-5.5 \
  --raw-output-dir exp/agent_loop/claude/20260609T172829/loops/loops001/exp/A4/raw \
  --token-usage-output exp/agent_loop/claude/20260609T172829/loops/loops001/exp/A4/token_usage.jsonl \
  --output exp/agent_loop/claude/20260609T172829/loops/loops001/exp/A4/val_public_boundary_rules_v4.csv \
  --start-from 265 \
  --timeout 300
```

Rows 1-264 were recovered from existing raw JSON files (213 files pre-existing).
Rows 265-500 were newly predicted.

## Run Summary (from script stdout)

```json
{
  "output": "exp/agent_loop/claude/20260609T172829/loops/loops001/exp/A4/val_public_boundary_rules_v4.csv",
  "rows": 500,
  "filtered_rows": 91,
  "predicted_rows": 404,
  "errors": 5,
  "flow": "filter_pred_by_codex",
  "model": "gpt-5.5",
  "prompt_path": "configs/prompt/stage4/codex/boundary_rules_v4.txt"
}
```

- Total rows: 500
- Filtered (N/A, ST1=No): 91
- Predicted by Codex: 404
- Errors (label parse failure): 5

## Evaluation: val_public ST4 Macro-F1

```
N/A:                  P=0.625  R=0.667  F1=0.645
already:              P=0.515  R=0.497  F1=0.506
between_2_and_5_years:P=0.506  R=0.607  F1=0.552
more_than_5_years:    P=0.741  R=0.460  F1=0.567
within_2_years:       P=0.667  R=0.933  F1=0.778

Macro-F1: 0.6097
```

## Comparison vs Baseline

| Metric                   | Baseline (val_test) | A4 (val_public) | Delta   |
|--------------------------|---------------------|-----------------|---------|
| Macro-F1                 | 0.5109              | 0.6097          | +0.0988 |
| already                  | 0.454               | 0.506           | +0.052  |
| within_2_years           | 0.222               | 0.778           | +0.556  |
| between_2_and_5_years    | 0.575               | 0.552           | -0.023  |
| more_than_5_years        | 0.603               | 0.567           | -0.036  |
| N/A                      | 0.701               | 0.645           | -0.056  |

Note: Baseline is from val_test; A4 is evaluated on val_public. The comparison is indicative only.

## Output Artifacts

- Predictions CSV: `exp/agent_loop/claude/20260609T172829/loops/loops001/exp/A4/val_public_boundary_rules_v4.csv`
- Raw JSON per-row: `exp/agent_loop/claude/20260609T172829/loops/loops001/exp/A4/raw/` (500 files total)
- Token usage: `exp/agent_loop/claude/20260609T172829/loops/loops001/exp/A4/token_usage.jsonl`
