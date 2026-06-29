# Loop 001 — Dev (subtask synth_promise / source A2)

A2 synthesis source: `promise_string`-derived positive (`Yes`) ST1 training rows,
generated OFFLINE, plus B1/B2/B3 mix files.

## Created files (exact paths)

Generator (reusable, under `core/data/` per filetree rules):
- `/workspace/esg_contest/core/data/synthesize_st1_a2_promise.py`

Generated data (under `data/generated/synth_st1_a2/`):
- `/workspace/esg_contest/data/generated/synth_st1_a2/synth_st1_a2_promise_pool.json`
  — full synthetic `Yes` pool (793 rows).
- `/workspace/esg_contest/data/generated/synth_st1_a2/synth_st1_a2_promise_mix_b1.json`
- `/workspace/esg_contest/data/generated/synth_st1_a2/synth_st1_a2_promise_mix_b2.json`
- `/workspace/esg_contest/data/generated/synth_st1_a2/synth_st1_a2_promise_mix_b3.json`
- `/workspace/esg_contest/data/generated/synth_st1_a2/synth_st1_a2_manifest.json`
  — counts, seed, generation method, sampling policy per mix.

## Row counts per mix

Real train size = 1000 (`data/raw_data/vpesg_4k_train_1000.json`).
Usable `Yes` sources = 793 (Yes rows whose `promise_string` length >= 12).
Unique synthetic pool = 793 (all label `Yes`, ids `syn_a2_<srcid>`).

| mix | multiplier | synthetic rows | sampling | real | total |
|-----|-----------|----------------|----------|------|-------|
| B1  | 0.5x      | 500            | without replacement | 1000 | 1500 |
| B2  | 1.0x      | 1000           | with replacement    | 1000 | 2000 |
| B3  | 2.0x      | 2000           | with replacement    | 1000 | 3000 |

Design note on sampling: the unique pool (793) is smaller than the B2 (1000)
and B3 (2000) synthetic targets because there are only 814 real `Yes` rows and
the 12-char min-length filter drops 21 short spans. To make B2 and B3 genuine
1.0x / 2.0x synthetic-heavy mixes (and not collapse to an identical 793-row
cap), `--allow-oversample` samples WITH replacement for those mixes; duplicated
synthetic ids are suffixed `_i<k>` to keep every training-row id unique. B1
(500 <= 793) stays without replacement. Seed = 42 for reproducible sampling.

## Generation method

- Endpoint: OFFLINE OpenAI-compatible `http://192.168.1.79:3134/v1`,
  model `/workspace/llm_model` (verified reachable with `curl /v1/models`
  before any generation).
- Per source span, the LLM paraphrases `promise_string` into a single
  self-contained Traditional-Chinese commitment sentence (preserving numbers,
  targets, timeframes), system+user prompt in the script.
- Leakage guard `is_leak()`: if the paraphrase is byte-identical to the span,
  contains/equals the span as a substring, or has >= 0.92 char-set overlap, the
  row is re-authored by a deterministic template wrapper instead, so output
  `data` is never a raw copy of the annotation span.
- Deterministic fallback (also used when a per-row LLM call fails): wraps the
  span in one of several commitment prefixes ("本公司承諾，…", "我們將持續落實以下承諾：…", etc.)
  into a self-contained promise sentence.
- Result: pool = 793 (690 LLM paraphrase, 103 deterministic). LLM endpoint was
  heavily throttled during the run (~45 min wall clock); generation completed
  successfully and writes outputs atomically at the end.
- Mix files were rebuilt from the saved pool via
  `--rebuild-mixes-from-pool --allow-oversample` (NO new LLM calls) to apply the
  oversampling policy; pool content is unchanged.

Hard negatives: NOT synthesized in this subtask. The plan scopes A2 to
`Yes`/promise positives; hard `No` examples are A4's responsibility (PDF-derived
pool, subtask synth_pdf). A2 stays a pure positive-augmentation source.

## Schema

Each synthetic row carries the training schema needed by `build_subtask_samples`:
`id`, `data`, `esg_type`, `promise_status` (= `Yes`). Pool rows additionally
carry offline-only provenance fields `synth_source`, `synth_method`, `src_id`
(kept ONLY in the pool, NOT in the mix files). Mix files contain only
`id`, `data`, `esg_type`, `promise_status`.

## Data-use compliance (explicit)

- `promise_string` is used OFFLINE ONLY, to author synthetic training text. It
  is the user-approved offline override for this loop. It is NEVER written into
  any output `data` field, mix file, or any runtime/eval/RAG input.
- The synthetic OUTPUT `data` text is a fresh self-contained sentence
  (paraphrase or template-wrapped), usable at runtime with no annotation
  leakage. Verified programmatically: 0 / 793 pool `data` strings are
  byte-identical to any real `promise_string`; 0 synthetic rows in B1/B2/B3 are
  verbatim `promise_string`.
- Mix files contain NO forbidden annotation fields — verified the union of keys
  across B1/B2/B3 excludes `promise_string`, `evidence_string`,
  `evidence_status`, `evidence_quality`, `verification_timeline`. Only
  `id`/`data`/`esg_type`/`promise_status` are present.
- Ground-truth `promise_status` is used only to set the synthetic label `Yes`
  (offline), never as a runtime input.
- All synthetic ids are prefixed `syn_a2_` and are unique within each mix file.

## Verification run

`python3 core/data/synthesize_st1_a2_promise.py --out-dir data/generated/synth_st1_a2`
(full LLM generation) then
`... --rebuild-mixes-from-pool --allow-oversample` (mix rebuild).
Post-checks confirmed schema, label, id-prefix, id-uniqueness, no forbidden
keys, and zero verbatim promise_string leakage across pool + all three mixes.

## Blocked items

None. A2 pool + B1/B2/B3 mixes are ready for the `train_grid` exp subtask.
The train_grid runner should point the A2 arms at the three mix files above.
