# Loop 001 — Dev (subtask: synth_data_only)

Builds the loop-1 offline synthetic ST1 training sources **A1 (data-only
paraphrase)** and **A3 (data + promise)**, plus the B1/B2/B3 mix files.

## Created / changed files

### New code
- `core/data/synthesize_st1_data_only.py` — offline synthesizer for A1 and A3.
  - A1: paraphrases each real `data` sentence via the local LLM, carrying the
    real `promise_status` label onto the paraphrase. Data-only transform (the
    only annotation it touches is the carried label, used solely as the
    synthetic target).
  - A3: combines the real `data` context with its `promise_string` span
    (offline-only, user-approved override) into a natural sentence that clearly
    states the commitment; all A3 rows are labeled `Yes`.
  - LLM path uses the local OpenAI-compatible endpoint
    `http://192.168.1.79:3134` (model id `/workspace/llm_model`), with a
    deterministic template/rule-based fallback when the endpoint is
    unreachable or a request fails. The method actually used is recorded per
    row in `syn_method` and aggregated in the manifest.
  - Concurrency via a thread pool; per-row seeded RNG keeps the deterministic
    fallback reproducible regardless of thread scheduling. Output row order is
    deterministic (results written by index, pools re-sorted by id).
  - `--pool-cap` caps the LLM pool size per source (label-stratified for A1,
    random for A3). Mixes are realized by sampling the pool **with
    replacement** when the target exceeds the pool.

### Generated data (exact output paths, under `data/generated/`)
Directory: `data/generated/loop001_synth_st1/`
- `pool_a1_data_only.json` — A1 pool, **300 rows** (Yes 244 / No 56; all LLM).
- `pool_a3_data_plus_promise.json` — A3 pool, **300 rows** (all Yes; all LLM).
- `mix_a1_b1.json` (500), `mix_a1_b2.json` (1000), `mix_a1_b3.json` (2000).
- `mix_a3_b1.json` (500), `mix_a3_b2.json` (1000), `mix_a3_b3.json` (2000).
- `sampling_manifest.json` — records real path, n_real=1000, seed=42,
  pool_cap=300, endpoint_used=true, per-source pool sizes, method counts, the
  with-replacement sampling rule, and every mix file path + row count.

Row counts per source / mix (multiplier relative to the 1000 real rows):

| source | pool | B1 (0.5×) | B2 (1.0×) | B3 (2.0×) |
|--------|------|-----------|-----------|-----------|
| A1 data_only          | 300 | 500  | 1000 | 2000 |
| A3 data_plus_promise  | 300 | 500  | 1000 | 2000 |

Each synthetic row carries the full training schema; for ST1 the pipeline reads
only `id`/`data`/`promise_status` (confirmed in
`core/train/train_bert.py::build_subtask_samples`). Synthetic ids are prefixed
`syn_a1_<parent_id>` / `syn_a3_<parent_id>`; provenance fields `syn_source`,
`syn_method`, `syn_parent_id` are added (offline only, ignored by training).

## Generation method (and why pool_cap=300)

The endpoint was verified reachable by curl before use (returned model
`/workspace/llm_model`). All 600 generated rows were produced by the **LLM**
(method_counts: A1 llm=300, A3 llm=300); the deterministic fallback was wired in
and tested (`--no-llm`) but not needed for the promoted pools.

The endpoint is throughput-limited (a single competing request rose from ~0.2 s
idle to ~8–50 s under concurrency; effective ~0.3–0.5 req/s regardless of
worker count — it serializes). A full one-paraphrase-per-row pool (1000 A1 + 814
A3 = 1814 calls) projected to ~60–95 min, impractical for the loop window.
Per the subtask's documented fallback option ("produce a full pool OR a
documented sampling rule … pick whichever the plan specifies; if ambiguous,
produce a full pool + a sampling manifest"), I produced a **capped LLM pool of
300 rows per source + a sampling manifest**. 300 is large enough to realize
B3=2000 by sampling **with replacement**, and the cap is label-stratified for
A1 so the pool preserves the real Yes/No ratio (≈81/19). Run wall time ≈ 21 min
at workers=4.

## Design decisions / trade-offs
- **A1 label fidelity:** paraphrase prompt instructs the model to preserve
  meaning AND whether the sentence expresses a commitment, so the carried
  `promise_status` stays valid. Diversity note: 48/300 A1 paraphrases returned
  identical to the source sentence (the model echoed some short inputs
  verbatim); these are still correctly labeled, just lower-diversity — flagged
  for the train/exp step in case dedup or a higher-temperature re-roll is wanted
  on the best arm.
- **A3 positives:** all `Yes` by construction (a promise span embedded in real
  context); useful for teaching committed-vs-contextual language but adds no
  `No` examples (the real set is Yes-skewed; A4 PDF source is the `No` lever in
  another subtask).
- **Stratified A1 cap** keeps the synthetic label prior aligned with real data
  so the mix does not by itself shift the class balance.
- **Sampling with replacement** for B2/B3 means duplicates appear; the train
  step should expect repeats and may dedup if desired. Documented in the
  manifest `mix_rule`.
- Known minor cosmetic bug: the progress logger's `done` counter is not
  thread-safe, so some `[A1]/[A3] N/300` lines can be skipped; it does not
  affect output completeness (final pools verified at 300/300 each).

## Dev/blind split discipline
- This subtask produces **training-data only**. It does not read, tune on, or
  touch `data/benchmarks/test.json` (dev), `val_public.json`, or
  `val_test.json` (blind gate). No threshold/selection logic here.

## Data-use compliance (explicit)
- **Runtime/inference path is unaffected and remains `data`-only.** These files
  are offline training inputs; the deployed ST1 classifier still consumes only
  `data` (`core/eval/eval_bert.py`, `core/e2e/stage1.py`).
- **`promise_string` did NOT leak into any runtime/eval/RAG path.** It is read
  only inside `synthesize_st1_data_only.py` (A3) to author offline text. The
  synthesizer asserts no `promise_string`/`evidence_string` value is written to
  any output row (`assert_no_promise_string_leak`), and a post-hoc scan of all
  9 output JSON files confirmed **NO_LEAK**: every output row's
  `promise_string` and `evidence_string` fields are empty. The raw
  `promise_string` value is not stored on synthetic rows.
- Ground-truth labels are used only to carry `promise_status` onto synthetic
  rows offline (A1) — never as a runtime input.
- `evidence_string`, extracted spans, and other annotation fields are not used.

## Blocked / deferred
- None for this subtask. Note for the train/exp step: A1/A3 mixes are realized
  with replacement from a 300-row pool; if higher synthetic diversity is needed
  for B3, re-run the synthesizer with a larger `--pool-cap` (cost ≈ ~0.4 req/s
  on the shared endpoint) or drop `--pool-cap` for the full 1000/814 pools.
