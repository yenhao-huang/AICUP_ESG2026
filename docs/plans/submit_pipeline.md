# Submit Pipeline Plan

## Goal

Align `How to Reproduce` and the executable submit flow with the predict architecture:

```text
Stage 1: BERT ensemble -> Gemma low-confidence fallback -> confidence merge
Stage 2: BERT ensemble -> Gemma low-confidence fallback -> confidence merge -> Stage 1 gate
Stage 3: MultitaskBERT -> GPT fallback -> Stage 1/2 gate
Stage 4: GPT/Codex prediction -> Stage 1 gate
All: stage finals -> submission.csv
```

The final output must be checkable against:

```text
docs/check/check_submit_model_consistent/submit/all/submission.csv
```

## File Placement

Per user direction, reusable merge and gate code goes under:

```text
core/service/utils/
```

Shell orchestration goes under:

```text
scripts/submit.sh
```

## Implementation Steps

1. Create `core/service/utils/submit_pipeline.py`.
   - Merge Stage 1 BERT soft-vote rows with Gemma rows when BERT confidence is below threshold.
   - Merge Stage 2 BERT soft-vote rows with Gemma rows when BERT confidence is below threshold.
   - Apply Stage 1 gate to Stage 2 and Stage 4.
   - Apply Stage 1/2 gate to Stage 3.
   - Build final `submission.csv`.

2. Create `scripts/submit.sh`.
   - Run submit-mode prediction wrappers.
   - Run Gemma fallback prediction.
   - Call `submit_pipeline.py` for merge/gate/build steps.
   - Write outputs under `results/submit/<RUN_ID>/`.
   - Support `DRY_RUN=1`.

3. Update `README.md`.
   - Make `How to Reproduce` pipeline-oriented instead of wrapper-only.
   - Show BERT -> Gemma -> merge -> gate flow for Stage 1 and Stage 2.
   - Show gated final submission build.

4. Extend consistency checks only if needed.
   - Current stage checks compare raw per-stage submit artifacts.
   - The all-submission check target is `docs/check/check_submit_model_consistent/submit/all/submission.csv`.

## Known Rules From Existing Artifacts

- Stage 1 low-confidence threshold: `0.6`.
- Stage 2 low-confidence threshold: `0.7`.
- Stage 1 confidence is `max(score_yes, score_no)`.
- Stage 2 confidence is parsed from `postprocess_reason` scores.
- Stage 2 gate: if Stage 1 `promise_status != Yes`, set `evidence_status = N/A`.
- Stage 3 gate: if Stage 1 `promise_status != Yes` or Stage 2 `evidence_status != Yes`, set `evidence_quality = N/A`.
- Stage 4 gate: if Stage 1 `promise_status != Yes`, set `verification_timeline = N/A`.

## Validation

Use existing artifacts first, without rerunning models:

```bash
python3 core/service/utils/submit_pipeline.py merge-stage1 \
  --bert docs/check/check_submit_model_consistent/submission_12/stage1/tmp/softvote_raw.csv \
  --gemma docs/check/check_submit_model_consistent/submit/gemma/stage1_gemma.csv \
  --output /tmp/stage1.csv \
  --threshold 0.6 \
  --run-id submit_9_stage1 \
  --bert-source bert

python3 core/service/utils/submit_pipeline.py merge-stage2 \
  --bert docs/check/check_submit_model_consistent/submission_12/stage2/tmp/softvote_raw.csv \
  --gemma docs/check/check_submit_model_consistent/submit/gemma/stage2_gemma.csv \
  --output /tmp/stage2_raw.csv \
  --threshold 0.7

python3 core/service/utils/submit_pipeline.py gate-stage2 \
  --stage1 /tmp/stage1.csv \
  --stage2 /tmp/stage2_raw.csv \
  --output /tmp/stage2.csv

python3 core/service/utils/submit_pipeline.py build-submission \
  --stage1 /tmp/stage1.csv \
  --stage2 /tmp/stage2.csv \
  --stage3 docs/check/check_submit_model_consistent/submission_12/stage3/softvote_gated.csv \
  --stage4 docs/check/check_submit_model_consistent/submission_12/stage4/codex_gated.csv \
  --output /tmp/submission.csv
```

Expected result:

```text
/tmp/submission.csv == docs/check/check_submit_model_consistent/submit/all/submission.csv
```
