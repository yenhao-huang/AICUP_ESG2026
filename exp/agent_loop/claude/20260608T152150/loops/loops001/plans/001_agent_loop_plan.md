# Loop 001 — Synthetic-Data Augmentation for the ST1 BERT Classifier (clean BERT + synthetic baseline)

## Method Family

Synthetic-data augmentation for the Stage 1 (Promise Identification, `Yes`/`No`,
Macro-F1) BERT classifier. This loop establishes a **clean BERT + synthetic-data
baseline**: a `hfl/chinese-roberta-wwm-ext-large` classifier (same backbone as
the baseline artifact) trained on real ST1 data plus offline-synthesized
ST1 examples mixed at chosen weights. **No LLM fallback is introduced in this
loop** — confidence-routed LLM-RAG escalation is explicitly deferred to a later
loop so loop 1 isolates the effect of synthetic data alone.

The single varied axis is *what synthetic training data we add and how much of
it we mix in*. Everything downstream of training (runtime inference) stays a
pure `data`-only BERT forward pass, identical in shape to the current Method 1
in `docs/methods.md`.

### Why a real grid (not one point)

The loop runs a 2-dimensional grid:

**Axis A — synthesis source** (how synthetic rows are produced; all offline):

- `A0_real_only`: no synthetic rows. This is the reproduction/control arm and
  also the arm that produces the **gate baseline measurement** (see Baseline).
- `A1_data_only`: synthesize new ST1-labeled sentences from existing real
  `data` text only (paraphrase/segment real `data` sentences; label carried from
  the real row's `promise_status`). Runtime-clean source, offline transform.
- `A2_promise_derived`: synthesize `Yes` rows whose text is built **offline**
  from `promise_string` spans (user-approved offline override). These become
  additional positive examples. `promise_string` is used ONLY to author the
  synthetic training text offline; it never enters any runtime path.
- `A3_data_plus_promise`: synthesize rows offline by combining the real `data`
  context sentence with its `promise_string` span (e.g. promise span embedded in
  surrounding real context) to teach the boundary between committed and
  contextual language.
- `A4_pdf_derived`: mine candidate sentences offline from the report PDF text in
  `data/generated/raw_page_table.jsonl` / `raw_doc_table.jsonl`, and use the
  existing `data/generated/generation_from_data/merged_strict_st1_train.json`
  pool (n=1530, Yes 652 / No 878) as a ready PDF/data-derived synthetic source.
  This source is the strongest candidate for adding hard `No` examples, since
  the real train set is heavily `Yes`-skewed (Yes 814 / No 186).

**Axis B — synthetic:real mix weight** (fraction / multiplier of synthetic rows
relative to the 1000 real rows):

- `B0 = 0.0` (only meaningful for A0),
- `B1 = 0.5×` synthetic:real,
- `B2 = 1.0×`,
- `B3 = 2.0×` (synthetic-heavy).

Mixing is by concatenating synthetic rows into the training JSON before
`build_subtask_samples`. Real rows are always kept at full weight; synthetic
rows are capped by the multiplier.

**Core knobs (held fixed unless an arm is degenerate):** backbone
`hfl/chinese-roberta-wwm-ext-large`, `max_len=512`, `epochs=5`,
`loss=weighted_ce` (inverse-frequency, matches baseline family), `seed=42`,
batch/grad-accum from the `large` config. We additionally probe ONE loss
variant on the best source/mix arm only: `loss=manual_ce` with class weights
favoring the minority `No` class (e.g. `No:Yes = 2.0:1.0`), because the chief
failure mode of the ST1 baseline is `No` recall under class imbalance.

This yields a concrete grid of roughly 1 (A0) + 4 sources × 3 mixes (A1–A4 ×
B1–B3) = 13 training runs, plus 1 loss-ablation run on the best arm = up to 14
candidate checkpoints. Each candidate is selected on the **dev split only**
(`data/benchmarks/test.json`) and the single best candidate is taken to the
blind gate exactly once.

## Novelty Check

- Closest prior loop: **none** — this is loop 1 of this autonomous run; no prior
  `loops/loops*/plans/*.md` exist in this workspace.
- The existing repo artifact `exp4_optimize2_highconf_yes_balanced_no_large` is
  treated as the external baseline, not a prior loop of this run. Its training
  source mix is not a controlled synthetic-augmentation grid; this loop is the
  first to systematically vary synthesis source × mix weight as the method axis.
- Method-level distinction vs. any future loop: this loop varies **training-data
  synthesis** only and forbids any runtime LLM/RAG/keyword postprocess. A later
  loop introducing confidence-routed LLM-RAG escalation would be a different
  method family (runtime routing), not a parameter change of this one.

## Baseline (gate baseline — MEASURE FIRST)

The gate baseline is the existing artifact
`models/exp4_optimize2_highconf_yes_balanced_no_large` (best_st1.pt,
`hfl/chinese-roberta-wwm-ext-large`), ST1 Macro-F1 measured **on the blind gate
`data/benchmarks/val_test.json`**. The historical 0.7950 figure was on the old
`val.json` and is NOT the gate baseline.

**First experiment action (blocking):** run
`core/eval/eval_bert.py --stage st1 --no-cascade` (data-only ST1 path) pointed at
`--model-dir models/exp4_optimize2_highconf_yes_balanced_no_large`,
`--pretrain-model hfl/chinese-roberta-wwm-ext-large`,
`--data-path data/benchmarks/val_test.json`, save to
`exp/agent_loop/claude/20260608T152150/loops/loops001/exp/baseline_val_test_st1.json`.
Record the resulting Macro-F1 as `BASE_GATE`. For reference also record the same
model's ST1 Macro-F1 on the dev split `data/benchmarks/test.json` as `BASE_DEV`
(used to sanity-check that dev-selection tracks the gate). `BASE_GATE` is reused
as the immutable gate baseline by every later loop.

- Primary metric: ST1 Macro-F1 on `data/benchmarks/val_test.json` (blind gate;
  measured exactly once per promoted candidate).
- Secondary/selection metric: ST1 Macro-F1 on `data/benchmarks/test.json` (dev;
  used for ALL model/mix/threshold selection).
- Report-split monitor: ST1 Macro-F1 on `data/benchmarks/val_public.json`
  (reported for context, never tuned on).
- Diagnostics: per-class P/R/F1 for `Yes`/`No`, especially `No` recall.

## score_first_promote gate

- **Baseline artifact:** `models/exp4_optimize2_highconf_yes_balanced_no_large`
  (current best in `docs/loops/agent_loop_state.json` for ST1 / per task spec).
- **Candidate artifact path:**
  `models/loop001_st1_synth_<bestarm>/best_st1.pt` (e.g.
  `models/loop001_st1_synth_A4_pdf_b2`), with its eval JSON at
  `exp/agent_loop/claude/20260608T152150/loops/loops001/exp/candidate_val_test_st1.json`.
- **Primary blind-gate threshold:** candidate ST1 Macro-F1 on
  `data/benchmarks/val_test.json` must be **strictly greater than `BASE_GATE`**
  (the loop-1-measured baseline number). No relative tie counts as a pass.
- **Selection rule:** the candidate taken to the gate is the single arm with the
  highest **dev** Macro-F1 (`data/benchmarks/test.json`). The gate is touched
  exactly once for that candidate; the gate split is never used for selection.
- **Tolerances (unchanged-stage / inherited artifacts):** Stages 2–4 are not
  modified in this loop; their inherited artifacts and the cascade must be
  byte-for-byte unchanged. Required tolerance on every non-ST1 stage Macro-F1:
  delta = 0.000 (no regression permitted because nothing downstream changes).
  The promoted runtime ST1 path must remain a single `data`-only BERT forward
  pass.
- **Data-only runtime compliance:** the promoted checkpoint's inference path
  consumes `data` ONLY. Any candidate whose runtime depends on `promise_string`,
  `evidence_string`, extracted spans, labels, retrieval over annotation fields,
  or keyword postprocess is INVALID and cannot be promoted, regardless of score.
  `promise_string`/PDF use is permitted strictly offline in synthesis (logged in
  `dev/`).
- **Decision logic:** promote only if (primary gate passes) AND (data-only
  compliance verified) AND (no downstream regression) AND (reflect record +
  Claude review completed). Otherwise the decision is `reject` or `defer`. A
  missing/mis-baselined/label-derived metric counts as a fail, not a pass.
- On promotion: update `docs/methods.md` Stage 1 Method 1 (note the synthetic
  augmentation source + mix) and `docs/loops/agent_loop_state.json` in the same
  work session; otherwise leave both unchanged.

## Data-Use Boundaries

- **Runtime / inference input: `data` ONLY.** The deployed ST1 classifier sees
  only raw `data` text at eval and inference. Confirmed by reusing
  `core/eval/eval_bert.py` (its ST1 path reads `d["data"]`) and
  `core/e2e/stage1.py` (`--method bert`, data-only).
- **Offline-only (synthesis):** `promise_string` (user-approved override) and
  report PDF text (`raw_page_table.jsonl`, `raw_doc_table.jsonl`,
  `generation_from_data/*`) may be used ONLY to author synthetic training rows.
  They must never appear in any runtime/eval/RAG/feature path. Any artifact that
  leaks them into runtime is INVALID for promotion.
- **Forbidden as model input always:** `evidence_string`, extracted
  promise/evidence, ground-truth labels, and any other derived annotation field.
- Ground-truth labels are used only for (a) offline scoring and (b) carrying the
  `promise_status` label onto synthetic rows during offline synthesis — never as
  a runtime input.
- Thresholds/hyperparameters tuned ONLY on `data/benchmarks/test.json`; never on
  `val_public.json` or `val_test.json`.

## Failure Modes & Abandonment Criteria

- **Synthetic noise:** paraphrase/PDF synthesis introduces mislabeled rows that
  hurt minority `No` recall. Evidence: dev Macro-F1 of every synthetic arm ≤
  `BASE_DEV` and `No`-recall not improving. Action: abandon that source; keep
  `A0_real_only`/best arm.
- **Distribution shift / overfit to synthetic style:** dev improves but the
  best arm fails the blind gate (gate Macro-F1 ≤ `BASE_GATE`). Decision:
  `reject`; do not promote; record that pure synthetic augmentation did not beat
  baseline and that loop 2 should pursue a different family (e.g. confidence
  routing / LLM-RAG fallback).
- **Mix saturation:** B3 (2.0×) consistently underperforms B1/B2, indicating
  synthetic-heavy mixes wash out real signal. Action: drop B3 from future grids.
- **Compliance leak:** if any candidate is found to require a non-`data` field
  at runtime, it is immediately marked INVALID and excluded from selection.
- **Method-family abandonment:** if NO source × mix arm beats `BASE_GATE` on the
  blind gate after dev-selection, synthetic-data augmentation as the *sole* ST1
  lever is considered exhausted for the baseline backbone; loop 2 must move to a
  materially different family rather than another synthesis grid.

## Parallel Subtasks
- id: baseline_gate  phase: exp  description: Measure baseline exp4 model ST1 Macro-F1 on val_test.json (BASE_GATE) and on test.json (BASE_DEV); write baseline eval JSON. BLOCKING — must finish before candidate selection.
- id: synth_data_only  phase: dev  description: Build A1 (data-only paraphrase) and A3 (data+promise) offline synthetic ST1 row files at B1/B2/B3 mixes.
- id: synth_promise  phase: dev  description: Build A2 (promise_string-derived positives) offline synthetic ST1 row files at B1/B2/B3 mixes.
- id: synth_pdf  phase: dev  description: Build A4 (PDF/data-derived, reuse generation_from_data merged pool) offline synthetic ST1 row files at B1/B2/B3 mixes.
- id: train_grid  phase: exp  description: Train the source×mix grid checkpoints (A0 control + A1–A4 × B1–B3) plus the manual_ce loss ablation on the best arm; one shared train script over the prepared mix files.
- id: select_and_gate  phase: exp  description: Score every checkpoint on dev (test.json), pick best dev arm, run it once on the blind gate (val_test.json) and val_public.json, compare to BASE_GATE for the promotion decision.
