---
name: predict
description: Run, inspect, and evaluate ESG contest prediction pipelines. Use when the user asks to predict a stage, rerun a stage CSV, evaluate predictions, compare prediction artifacts, or use data/predictions/stage<N>/ inputs with core/human/predict/stage<N>/ scripts.
---

# Predict

Use this skill for VeriPromiseESG4K prediction work in `/workspace/esg_contest`.
It is the Codex version of the local `.claude/skills/predict` note, expanded
into an operational workflow.

## Scope

Invoke when the user asks for:

- `predict`, `預測`, `跑 stage<N>`, `rerun stage<N>`
- evaluate prediction CSVs with `core/human/predict/eval.py` or
  `core/human/predict/eval_pipeline.py`
- compare prediction artifacts under `data/predictions/`,
  `results/predict/`, `exp/**/stage<N>/`, or submit/test pipeline dirs
- inspect confidence, class distribution, or changed rows in prediction CSVs

## Repository Conventions

- Root is `/workspace/esg_contest`.
- Stage prediction scripts live under:
  - `core/human/predict/stage1/`
  - `core/human/predict/stage2/`
  - `core/human/predict/stage3/`
  - `core/human/predict/stage4/`
- Historical prediction inputs often live under:
  - `data/predictions/stage1/`
  - `data/predictions/stage2/`
  - `data/predictions/stage3/`
  - `data/predictions/stage4/`
- Pipeline artifacts often live under:
  - `exp/integrated_stage_predictions/**`
  - `exp/exp41/submit/**`
  - `results/predict/**`
- Evaluation labels:
  - validation submit format: `data/benchmarks/val_submit_format.csv`
  - public validation split: `data/benchmarks/val_public_submit_format.csv`
  - gate/test validation split: `data/benchmarks/val_test_submit_format.csv`

## Data-Use Rule

For stage inference, the raw model input must be `data` only unless the user
explicitly changes the problem definition. Do not use `evidence_string`,
`promise_string`, extracted evidence/promise, labels, or other annotation fields
as model/prompt/rule features.

## Workflow

### 1. Resolve Target

Identify:

- stage id: `stage1`, `stage2`, `stage3`, or `stage4`
- input data path
- upstream gate CSV, if required
- output CSV path
- checkpoint/model/prompt path
- whether the requested operation is prediction, evaluation, comparison, or
  analysis only

Prefer existing runner scripts and local patterns over inventing a new command.

### 2. Inspect Existing Artifacts

Use `rg` and small file reads first:

```bash
rg -n "STAGE3|stage3|finetune-path|--stage2-csv" main exp core/human/predict
find <pipeline-dir> -maxdepth 2 -type f -name "*.csv" | sort
```

For CSV schema checks:

```bash
python - <<'PY'
import pandas as pd
p = "<csv>"
df = pd.read_csv(p, dtype=str, keep_default_na=False, nrows=3)
print(df.shape)
print(list(df.columns))
print(df.head().to_string(index=False))
PY
```

### 3. Run Prediction

Use the stage script that matches the existing artifact family. Common examples:

```bash
.venv/bin/python core/human/predict/stage3/pred_by_bert.py \
  --data data/benchmarks/val.json \
  --finetune-path /models/agent_loop_st3_0612/loops005/A5_seed13/epoch_st3_005.pt \
  --stage2-csv <stage2.csv> \
  --output <stage3.csv>
```

```bash
.venv/bin/python core/human/predict/stage2/pred_by_bert.py \
  --data data/benchmarks/val.json \
  --finetune-path <stage2.pt> \
  --stage1-csv <stage1.csv> \
  --stage1-gate-col promise_status \
  --output <stage2_raw.csv> \
  --text-mode data
```

When a runner already exists, edit the runner instead of manually retyping a
long command. Keep output paths inside that runner's artifact directory.

### 4. Evaluate

Single CSV:

```bash
.venv/bin/python core/human/predict/eval.py \
  --pred-file <prediction.csv> \
  --label-file data/benchmarks/val_submit_format.csv
```

Full pipeline:

```bash
.venv/bin/python core/human/predict/eval_pipeline.py \
  --pipeline-dir <pipeline-dir>
```

`eval_pipeline.py` defaults to `data/benchmarks/val_submit_format.csv` unless
`--label-file` is provided.

### 5. Analyze Prediction Quality

Report class distribution, confidence bins, and changed rows when useful.
Use fixed confidence bins:

```text
<0.50 | 0.50-0.60 | 0.60-0.70 | 0.70-0.80 | 0.80-0.90 | >=0.90
```

Score extraction:

- ST1: `max(score_yes, score_no)`
- ST2: parse `score_yes` / `score_no` from `postprocess_reason`
- ST3: parse `score_clear`, `score_not_clear`, `score_misleading` from
  `evidence_quality_reason`
- ST4: usually no score distribution unless a script explicitly emits scores

For before/after comparison, join by `id` and count changed labels per task.

### 6. Validation Before Final

Run the most focused checks available:

- `bash -n <runner.sh>` after editing shell runners
- `test -f <checkpoint>` for model paths
- evaluation command for changed outputs
- schema check for final `submission.csv`

## Output

Final response should include:

- artifact path(s)
- command(s) run
- metric table or key scores
- changed-row summary when comparing artifacts
- any validation that could not be run
