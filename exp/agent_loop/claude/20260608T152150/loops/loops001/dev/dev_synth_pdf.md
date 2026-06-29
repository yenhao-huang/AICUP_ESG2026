# Loop 001 — Dev (subtask: synth_pdf / A4 PDF-derived synthetic ST1 rows)

## Summary

Built source **A4 (PDF/data-derived)** offline synthetic ST1 training rows at
mix weights B1=0.5x, B2=1.0x, B3=2.0x relative to the real train set, by reusing
the existing pool `data/generated/generation_from_data/merged_strict_st1_train.json`
exactly as the plan designates. No LLM call was needed — A4 is produced by
deterministic reuse/dedup/sampling of an already-materialised PDF/data-derived
pool.

## Key finding (decomposition of the pool)

`merged_strict_st1_train.json` has n=1530, which decomposes cleanly against the
real train set `data/benchmarks/train.json` (n=800):

- 800 rows have `data` text byte-identical to the real train rows (the pool is a
  superset of real train).
- **730 NET-NEW rows, ALL labelled `No`** (deduped by `data` text).

These 730 are exactly the "strong source of hard `No` examples" the plan cites.
They directly counter the real train Yes-skew (real train: Yes 652 / No 148).
A4 is therefore a pure hard-negative augmentation source.

## Created files (exact paths)

Generator (data-prep code, per `docs/rules/filetree.md` → `core/data/`):
- `/workspace/esg_contest/core/data/build_synth_st1_a4_pdf.py`

Outputs under `data/generated/synth_st1_a4/` (per filetree + named rules):
- `/workspace/esg_contest/data/generated/synth_st1_a4/synth_st1_a4_pdf_pool.json`
  — full A4 pool, n=730 (No=730), runtime-safe schema.
- `/workspace/esg_contest/data/generated/synth_st1_a4/synth_st1_a4_pdf_b1.json`
  — B1 mix (0.5x), n=400.
- `/workspace/esg_contest/data/generated/synth_st1_a4/synth_st1_a4_pdf_b2.json`
  — B2 mix (1.0x), n=800.
- `/workspace/esg_contest/data/generated/synth_st1_a4/synth_st1_a4_pdf_b3.json`
  — B3 mix (2.0x), n=1600.
- `/workspace/esg_contest/data/generated/synth_st1_a4/synth_st1_a4_pdf_manifest.json`
  — sampling manifest (mix multipliers, target/written row counts,
  oversampling flags, label counts, unique-id counts, data-use note).

Reused (read-only, OFFLINE):
- `/workspace/esg_contest/data/generated/generation_from_data/merged_strict_st1_train.json`
- `/workspace/esg_contest/data/benchmarks/train.json` (real-train overlap removal only)

`raw_doc_table.jsonl` / `raw_page_table.jsonl` were inspected to confirm the pool
is PDF/page-derived; the pool already carries that provenance, so the raw tables
were not re-mined (the plan permits reusing the merged pool as the ready
PDF/data-derived source).

## Row counts per mix

| mix | multiplier | target | rows written | unique ids | labels      | oversampled |
|-----|-----------|--------|--------------|------------|-------------|-------------|
| pool| —         | —      | 730          | 730        | No=730      | no          |
| b1  | 0.5x      | 400    | 400          | 400        | No=400      | no          |
| b2  | 1.0x      | 800    | 800          | 483        | No=800      | yes         |
| b3  | 2.0x      | 1600   | 1600         | 663        | No=1600     | yes         |

## Design decisions / trade-offs

- **Mix base = 800 (actual real train size), not 1000.** The plan's "1000 real
  rows" is nominal; `data/benchmarks/train.json` is 800 rows. B1/B2/B3 are
  computed as 0.5/1.0/2.0 × 800 = 400/800/1600 and this is recorded in the
  manifest. Flagged for the train_grid subtask in case it prefers the nominal
  1000 base.
- **A4 pool is only 730 unique rows**, so B2 (800) and B3 (1600) exceed the pool
  and are oversampled WITH REPLACEMENT, deterministically (seed=42). B1 (400) is
  a seeded subsample without replacement. A full pool file plus this documented
  manifest are both provided so train_grid can choose dedup-vs-oversample.
- **Schema stripped to `id`/`data`/`promise_status`** — the only fields
  `build_subtask_samples(..., "st1")` reads (`core/train/train_bert.py` L70–72).
  No other field is emitted, eliminating any chance of annotation leakage when
  these rows are concatenated into the real train JSON.
- **ids prefixed `syn_a4_`** (e.g. `syn_a4_10080`) so synthetic rows never
  collide with real ids and are auditable in any merged training file.

## Dev/blind split discipline

No tuning, selection, or threshold work was done here. This subtask only
materialises training-row files. Real-train overlap removal used only
`data/benchmarks/train.json` (the dev/train side); `test.json`,
`val_public.json`, and `val_test.json` were never read or referenced.

## Data-use compliance (explicit)

- **PDF/page text stayed OFFLINE-only.** It was consumed solely via the
  pre-materialised pool `merged_strict_st1_train.json` to author training rows;
  it never enters any runtime/eval/RAG/feature path.
- **`promise_string` / `evidence_string` did NOT leak.** Verified
  programmatically: the emitted pool and every B-mix file contain exactly the key
  set `{data, id, promise_status}` and zero forbidden fields
  (`promise_string`, `evidence_string`, `evidence_quality`,
  `verification_timeline`, `evidence_status`, spans, etc.).
- The `promise_status` label is carried offline only as the training target; it
  is never a model input. The runtime ST1 path remains a `data`-only BERT
  forward pass.
- **Runtime-safe `data` text:** every emitted `data` value is raw report
  sentence text — the same field the deployed model already consumes — with no
  annotation-derived content.

Validation command (all files pass: keys=`['data','id','promise_status']`,
leak=NONE, bad_id_prefix=0, bad_label=0):
`python3 core/data/build_synth_st1_a4_pdf.py` followed by a key/leak audit over
`data/generated/synth_st1_a4/synth_st1_a4_pdf_*.json`.

## Blocked items

None. Note for downstream `train_grid`: confirm whether the mix base should be
800 (used here) or the nominal 1000; only the B-mix target counts would change.
