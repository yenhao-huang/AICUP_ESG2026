# Loop 001 — Dev

Merged dev record for loop 001 ST1 offline synthetic-data generation. Three
subtasks each produced one synthetic ST1 training source (A1/A3, A2, A4) plus
B1/B2/B3 mixes. All generation is OFFLINE; the runtime/eval ST1 path stays
`data`-only. This record consolidates the per-subtask dev notes and surfaces the
OPEN ISSUES the `train_grid` experiment step must resolve before training.

## Subtask roster

| subtask | sources | status | generator script(s) | LLM used |
|---------|---------|--------|---------------------|----------|
| synth_data_only | A1 (data-only paraphrase), A3 (data + promise) | DONE | `core/data/synthesize_st1_data_only.py` | yes (600/600 rows LLM) |
| synth_promise   | A2 (promise_string-derived `Yes`) | DONE | `core/data/synthesize_st1_a2_promise.py` | yes (690 LLM / 103 deterministic) |
| synth_pdf       | A4 (PDF/data-derived hard `No`) | DONE | `core/data/build_synth_st1_a4_pdf.py` | no (deterministic reuse/dedup/sample) |

All three subtasks report **no blocked items**; all output files were generated
and validated. Remaining work is entirely in the `train_grid` exp step (see OPEN
ISSUES below).

## Generator scripts (exact paths)

- `/workspace/esg_contest/core/data/synthesize_st1_data_only.py` — A1 + A3.
- `/workspace/esg_contest/core/data/synthesize_st1_a2_promise.py` — A2.
- `/workspace/esg_contest/core/data/build_synth_st1_a4_pdf.py` — A4.

## Generated data files (exact paths + row counts per source/mix)

### A1 — data-only paraphrase (synth_data_only)
Directory: `/workspace/esg_contest/data/generated/loop001_synth_st1/`
- `pool_a1_data_only.json` — pool, **300 rows** (Yes 244 / No 56; all LLM; label-stratified cap).
- `mix_a1_b1.json` — **B1 = 500 synthetic rows** (0.5× of nominal 1000).
- `mix_a1_b2.json` — **B2 = 1000 synthetic rows** (1.0×).
- `mix_a1_b3.json` — **B3 = 2000 synthetic rows** (2.0×).
- `sampling_manifest.json` (shared with A3) — n_real=1000, seed=42, pool_cap=300, with-replacement rule, mix paths/counts.

### A3 — data + promise (synth_data_only)
Directory: `/workspace/esg_contest/data/generated/loop001_synth_st1/`
- `pool_a3_data_plus_promise.json` — pool, **300 rows** (all `Yes`; all LLM).
- `mix_a3_b1.json` — **B1 = 500 synthetic rows** (0.5×).
- `mix_a3_b2.json` — **B2 = 1000 synthetic rows** (1.0×).
- `mix_a3_b3.json` — **B3 = 2000 synthetic rows** (2.0×).

A1/A3 mixes are realized by sampling the 300-row pool **WITH replacement** when
the target exceeds the pool (so B2/B3 contain duplicates).

### A2 — promise_string-derived `Yes` (synth_promise)
Directory: `/workspace/esg_contest/data/generated/synth_st1_a2/`
- `synth_st1_a2_promise_pool.json` — pool, **793 rows** (all `Yes`; 690 LLM paraphrase / 103 deterministic).
- `synth_st1_a2_promise_mix_b1.json` — **B1 = 500 synthetic** (0.5×, WITHOUT replacement) + 1000 real = 1500 total.
- `synth_st1_a2_promise_mix_b2.json` — **B2 = 1000 synthetic** (1.0×, WITH replacement) + 1000 real = 2000 total.
- `synth_st1_a2_promise_mix_b3.json` — **B3 = 2000 synthetic** (2.0×, WITH replacement) + 1000 real = 3000 total.
- `synth_st1_a2_manifest.json` — counts, seed=42, method, per-mix sampling policy.

Note: A2 mix files **embed the real train rows** (mix = synthetic + real, e.g.
B1 total=1500). A1/A3/A4 mix files contain **synthetic rows only** (real not
concatenated). This shape mismatch is normalized by the train step (see OPEN
ISSUE c).

### A4 — PDF/data-derived hard `No` (synth_pdf)
Directory: `/workspace/esg_contest/data/generated/synth_st1_a4/`
- `synth_st1_a4_pdf_pool.json` — pool, **730 rows** (all `No`; deterministic, deduped net-new vs real train).
- `synth_st1_a4_pdf_b1.json` — **B1 = 400 synthetic** (0.5× of 800 base; WITHOUT replacement; 400 unique).
- `synth_st1_a4_pdf_b2.json` — **B2 = 800 synthetic** (1.0× of 800; WITH replacement; 483 unique).
- `synth_st1_a4_pdf_b3.json` — **B3 = 1600 synthetic** (2.0× of 800; WITH replacement; 663 unique).
- `synth_st1_a4_pdf_manifest.json` — multipliers, target/written counts, oversample flags, label counts.

A4 reuses (read-only, OFFLINE) `data/generated/generation_from_data/merged_strict_st1_train.json`
(n=1530 = 800 real-overlap + 730 net-new `No`); real-train overlap removed using
`data/benchmarks/train.json` (n=800).

### Consolidated source × mix matrix (synthetic row counts)

| source | base used | pool | B1 (0.5×) | B2 (1.0×) | B3 (2.0×) | mix shape |
|--------|-----------|------|-----------|-----------|-----------|-----------|
| A1 data_only          | 1000 | 300 | 500 | 1000 | 2000 | synth-only |
| A2 promise_string     | 1000 | 793 | 500 | 1000 | 2000 | synth + real embedded |
| A3 data+promise       | 1000 | 300 | 500 | 1000 | 2000 | synth-only |
| A4 pdf hard-`No`      | **800** | 730 | **400** | **800** | **1600** | synth-only |

The base/multiplier inconsistency (A4 used 800, others used 1000) and the B1/B2/B3
absolute counts that follow from it are the core normalization the train step
must perform (OPEN ISSUE a).

## Generation method notes

- A1/A3 and A2 used the OFFLINE OpenAI-compatible endpoint
  `http://192.168.1.79:3134` (model id `/workspace/llm_model`), verified
  reachable before generation. Endpoint is throughput-limited (~0.3–0.5 req/s,
  serialized), so A1/A3 used a label-stratified `--pool-cap 300` + sampling
  manifest rather than a full per-row pool. A2 ran the full 793-span pool
  (~45 min). seed=42 everywhere.
- A2 has a leakage guard `is_leak()` (byte-identical / substring / ≥0.92 char
  overlap → re-author via deterministic template) so output `data` is never a
  raw copy of the span.
- A4 needed no LLM: deterministic dedup/overlap-removal/sampling of an existing
  PDF/data-derived pool, schema stripped to `id`/`data`/`promise_status`.

## Data-use compliance (confirmed across all subtasks)

- **promise_string and PDF/page text were used OFFLINE ONLY**, solely to author
  synthetic training text (A2, A3) or via a pre-materialised PDF-derived pool
  (A4). They are the user-approved offline override for this loop. They never
  enter any runtime / eval / RAG / feature path.
- **No leakage in any output file.** Each subtask verified programmatically that
  output `data` is never a verbatim copy of `promise_string`, and that every
  output/mix file's key set excludes forbidden annotation fields
  (`promise_string`, `evidence_string`, `evidence_status`, `evidence_quality`,
  `verification_timeline`, spans). A1/A3 outputs assert empty
  `promise_string`/`evidence_string`; A2 mix files carry only
  `id`/`data`/`esg_type`/`promise_status`; A4 carries only
  `id`/`data`/`promise_status`. 0/793 A2 `data` strings byte-identical to any
  real `promise_string`.
- **Ground-truth labels carried offline only as the synthetic target**
  (`promise_status` on A1; fixed `Yes` on A2/A3; fixed `No` on A4); never a
  model input.
- **Runtime/eval ST1 path is unaffected and remains `data`-only** — the deployed
  classifier (`core/eval/eval_bert.py`, `core/e2e/stage1.py`) still consumes
  only `data`. ST1 training reads only `id`/`data`/`promise_status`
  (`core/train/train_bert.py::build_subtask_samples`); all extra provenance
  fields (`syn_source`/`synth_source`/`syn_method`/parent-id) are offline-only
  and ignored by training.
- Synthetic ids are uniquely prefixed per source (`syn_a1_`, `syn_a2_`,
  `syn_a3_`, `syn_a4_`) so synthetic rows never collide with real ids and are
  auditable in any merged training file.

## OPEN ISSUES for the train_grid (experiment) step

These MUST be resolved by `train_grid` before/while building training files. They
are not defects in the generated pools — they are cross-subtask inconsistencies
the train step has to normalize.

### (a) Real-train-size discrepancy — pick ONE canonical real-train file
- synth_data_only (A1/A3) and synth_promise (A2) computed B1/B2/B3 base counts
  against `data/raw_data/vpesg_4k_train_1000.json` (**n=1000**) → B1/B2/B3 =
  500/1000/2000.
- synth_pdf (A4) decomposed and computed mixes against
  `data/benchmarks/train.json` (**n=800**) → B1/B2/B3 = 400/800/1600.
- **The task spec designates `data/raw_data/vpesg_4k_train_1000.json` (n=1000) as
  the train source.** train_grid must pick this ONE canonical real-train file and
  **recompute B1/B2/B3 base counts consistently** for every source. A4's mixes
  (400/800/1600) are computed on the wrong base (800) and should be regenerated
  at 500/1000/2000 against the 1000-row train source, OR train_grid must
  explicitly justify a different base. A4's generator is deterministic and
  re-runnable; only the mix target counts change.

### (b) Per-source pool/quality caveats
- **A1:** 48/300 paraphrases were returned identical to the source sentence
  (model echoed short inputs). Still correctly labeled, but lower diversity —
  train_grid may dedup or high-temperature re-roll on the best arm.
- **A2:** unique pool is only **793** rows, yet B2 (1000) and B3 (2000) synthetic
  targets **oversample WITH replacement beyond the 793 pool** (duplicate ids
  suffixed `_i<k>`). B1 (500 ≤ 793) is without replacement.
- **A4:** pool is hard-`No` ONLY (**n=730**, all `No`); B2 (800) and B3 (1600)
  exceed the pool and oversample WITH replacement (483 / 663 unique). A4 adds no
  `Yes` examples — it is purely the hard-negative lever to counter real-train
  Yes-skew. A1/A3 likewise oversample with replacement from a 300-row pool for
  B2/B3.

### (c) Mix-size / mix-shape inconsistencies to normalize
- **Mix shape differs:** A2 mix files concatenate synthetic + real (B1 total
  1500, B2 2000, B3 3000). A1/A3/A4 mix files are synthetic-only. train_grid must
  normalize how real rows are combined (either consume A2's pre-merged mixes
  consistently, or rebuild all mixes from each source's pool with a uniform
  real-concat policy) so all arms train on comparable totals.
- **Absolute synthetic counts differ at the same B multiplier** because of issue
  (a): at B2, A1/A2/A3 = 1000 synthetic but A4 = 800 synthetic. Normalizing the
  base (issue a) fixes this.
- **Provenance fields vary across pools** (A1/A3 keep `syn_*`; A2 keeps
  `synth_source`/`synth_method`/`src_id` in the pool only; A4 strips to 3 keys).
  Training ignores extras, but train_grid should confirm a uniform projection to
  `id`/`data`/`promise_status` (+`esg_type` where present) before concatenation.
