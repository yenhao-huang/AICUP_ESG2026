# Loop 002 — Confidence-Routed LLM-RAG Fallback on the Proven Recipe BERT (decision-time escalation)

## Method Family

**Decision-time confidence-routed LLM-RAG escalation for Stage 1.** Take the
strongest *pure-BERT* ST1 checkpoint from loop001 as a frozen base predictor,
emit per-row softmax confidence, and route only the **lowest-confidence** rows to
an LLM (endpoint `http://192.168.1.79:3134`) that decides `Yes`/`No` from a
**data-only** prompt seeded with top-k training examples retrieved by `data`-`data`
embedding similarity. High-confidence BERT rows are kept unchanged. The runtime
artifact is `pred_by_bert.py` (scored CSV) → `pred_by_bert_codex_rag.py`
(fallback), both already in-repo (`docs/methods.md` Stage 1 Module B).

This is a **runtime routing** method, not a training-data method. Nothing about
the BERT weights or its training set changes; the only thing this loop optimizes
is *which rows to escalate, what context to retrieve, and what the LLM is asked*.

### Base predictor (frozen, chosen, not re-tuned)

Per the loop001 reflect, the validated win is the canonical-data **recipe**, and
the strongest defensible pure-BERT checkpoint is `models/loop001_st1_a0_real_only`
(real-only retrain; val_public 0.8031, val_test 0.8110, no synthesis-noise risk).
That checkpoint is the **base** for the fallback. We also carry
`models/loop001_st1_a3_b1` (val_test 0.8234) as a *secondary* base only to check
whether the fallback's lift is base-invariant (diagnostic, not for selection
double-dipping — see gate). The exp4 baseline is never the base; it is only the
gate baseline number.

Confidence = `max(softmax(score_yes, score_no))`, emitted by `pred_by_bert.py`
(verified lines 150–160: it writes `score_yes`/`score_no` per row).
`pred_by_bert_codex_rag.py` routes rows with `conf < threshold` to the LLM
(verified `get_confidence` lines 104–110, routing lines 503–514).

### Why a real grid (multiple variants/ablations, not one point)

The loop runs a multi-axis grid. All selection is on **val_public only**; the
blind gate `val_test` is touched exactly once for the single selected config.

**Axis T — confidence threshold (routing volume):** the central knob.
`T ∈ {0.60, 0.70, 0.80, 0.90, 0.95}`. Lower T routes fewer rows (only the most
uncertain); higher T escalates more. `T=0.50` ≈ "BERT always wins" (no routing) is
the implicit no-op control. We report, per T, how many val_public rows escalate and
the realized Macro-F1, to find the sweet spot where the LLM corrects BERT errors
without overturning correct high-confidence rows. A flat/негative response curve is
itself a result (escalation does not help → abandon, see Failure Modes).

**Axis K — retrieval depth:** `top-k ∈ {3, 5, 8}` in-context training examples
(retrieved over train `data` only, `--info-mode data-only`). Tests whether more
data-only context helps the LLM on the hard rows.

**Axis P — LLM prompt:** `{strict_negative, strict_negative_with_certficate,
default}` (the three available data-only ST1 prompts under
`configs/prompt/stage1/codex/`). The certificate prompt is the methods.md default
and treats ISO/GRI/certification language as positive signal; `strict_negative` is
the conservative anchor. This ablates whether the decision rule the LLM is given
matters on escalated rows.

**Axis C — mode-context:** `{bert-prediction, none}` — whether the LLM is shown the
BERT's low-confidence label+scores as soft context. Tests anchoring vs.
independence on exactly the rows BERT was unsure about.

**Routing-policy ablation R (on the best T/K/P/C arm only):** instead of a flat
`max-prob < T` gate, route by **margin** `|score_yes − score_no| < m`
(`m ∈ {0.10, 0.20}`) — a near-tie router — to check if margin-routing selects a
cleaner escalation set than absolute-confidence routing. Implemented by mapping the
margin band to an equivalent confidence threshold at scoring time (no new runtime
field; still `data`-only).

Grid size is bounded: a coarse T-sweep (5) × K (1 fixed at 5 for the sweep) × P
(certificate default) × C (bert-prediction default) locates the best T; then a
focused refinement varies K (3/5/8), P (3 prompts), C (2) around that T, plus the
2 margin-router points = on the order of 5 + 3 + 3 + 2 + 2 ≈ 15 fallback
configurations. **Critically, the LLM is only ever called on the low-confidence
subset**, so cost scales with routed rows, not with 500×15. The base BERT scored
CSV for val_public is computed once and reused across all fallback configs.

## Novelty Check

- **Closest prior loop: loop001** (synthetic-data augmentation — varying
  *training-data synthesis source × mix weight* for the BERT classifier).
- **Method-level difference:** loop001 changed *what the model is trained on*;
  loop002 changes *what happens at decision time to a frozen model's
  predictions*. loop002 adds no synthetic rows, retrains nothing, and introduces a
  second model (LLM) plus a retrieval+routing controller that did not exist in
  loop001. This is the decision-time-escalation family explicitly named as a NEW
  family in the loop001 plan ("A later loop introducing confidence-routed LLM-RAG
  escalation would be a different method family (runtime routing), not a parameter
  change of this one") and in the loop001 reflect advice item 5. It is **not** a
  hyperparameter sweep of loop001 and does not reuse its method family.
- **Non-repetition compliance (CLAUDE.md):** loop001 exhausted "vary synthesis
  source × mix." loop002 runs **no synthesis grid**; the supervision target,
  representation, and architecture (BERT→{retrieval, LLM judge}) are materially
  different.

## loop001 reflect recommendations — accept / reject / defer

The loop001 reflect (`loops/loops001/reflect/001_agent_loop_reflect.md`, "Advice
for loop 002") is cited item-by-item:

1. **"Fix the standing dev split first (retire test.json; adopt val_public as
   selection, val_test as blind gate; create `docs/loops/agent_loop_state.json`
   with BASE_GATE=0.802383)."** — **ACCEPT.** This loop selects on
   `data/benchmarks/val_public.json` ONLY and treats `data/benchmarks/val_test.json`
   as the locked blind gate; `test.json` is never used. On promotion (only) we will
   create/update `docs/loops/agent_loop_state.json` with BASE_GATE=0.802383. The
   inherited CRITICAL fact (test.json 200/200 train-leaked) is taken as established.
2. **"Isolate recipe vs synthesis with ≥3 seeds; promote a0_real_only if
   synthesis is within noise."** — **DEFER (partially superseded).** loop002 is a
   different family (routing), so a fresh seed-noise study of the synthesis arms is
   out of scope. We **accept its conclusion** by using `a0_real_only` (the proven
   recipe checkpoint) as the frozen base, not a synthetic arm. The recipe-vs-
   synthesis seed study, if still wanted as a standalone promotion of a0, is left to
   a dedicated promotion loop; this loop instead measures the fallback as a marginal
   lift over a0 (reflect advice item 5's intent).
3. **"Promote the safe proven win (a0_real_only)."** — **DEFER.** This loop does
   not promote a0 by itself; it builds on a0 and will promote the **fallback** only
   if it strictly beats the blind gate. If the fallback fails, the reflect will
   recommend a0-only promotion as the fallback-free alternative (so the proven win
   is not lost).
4. **"Do not run another synthesis grid; drop B3."** — **ACCEPT.** No synthesis
   grid in this loop.
5. **"Introduce the LLM-RAG fallback as the next family, measured as a marginal
   lift over the best BERT, gated on val_public, routing only lowest-confidence
   rows, retrieval over `data` only."** — **ACCEPT (this is the loop).** Exactly
   the design above: marginal lift over the frozen a0 base, val_public selection,
   confidence-routed lowest-conf rows, `--info-mode data-only` retrieval.
6. **"Always run the mandatory Claude review before the reflect verdict."** —
   **ACCEPT.** The reflect phase will run `claude --dangerously-skip-permissions`
   over methods.md + this loop's plans/dev/exp and store the review under
   `reflect/`, per CLAUDE.md.

## Baseline (immutable gate baseline — reuse, do NOT re-measure on the gate)

- **BASE_GATE = 0.802383** — exp4 model ST1 Macro-F1 on `data/benchmarks/val_test.json`
  (verified in loop001 `baseline_val_test_st1.json`). This is the immutable blind-gate
  baseline; loop002 does not re-touch the gate to "re-measure" it.
- **val_public baseline 0.7873** (exp4) and the **frozen base** a0_real_only
  val_public = **0.8031** / val_test = **0.8110** are the reference points the
  fallback must beat. The honest acceptance bar for *this loop's method* is to beat
  the **a0 base it is layered on** (val_public 0.8031), not merely the old exp4
  number — a fallback that only matches a0 adds LLM cost for nothing.
- **Primary metric:** ST1 Macro-F1 on `data/benchmarks/val_test.json` (blind gate),
  measured exactly once for the selected config.
- **Selection metric:** ST1 Macro-F1 on `data/benchmarks/val_public.json` (ALL
  threshold/K/prompt/context/router selection here).
- **Diagnostics:** per-class `No`-F1 / `Yes`-F1; #rows escalated; #labels flipped by
  LLM; flip correctness (of flipped rows, how many moved toward GT) on val_public;
  base-invariance check (same best config on a0 vs a3_b1 base).

## score_first_promote gate (before development)

- **Baseline artifact:** `models/exp4_optimize2_highconf_yes_balanced_no_large`
  (current ST1 default in `docs/methods.md`; BASE_GATE = 0.802383, immutable).
- **Frozen base (layered-on) artifact:** `models/loop001_st1_a0_real_only/best_st1.pt`
  (val_public 0.8031, val_test 0.8110). Its weights are NOT modified.
- **Candidate artifact path:** the selected fallback config, materialized as the
  scored-CSV + RAG runner config recorded at
  `exp/agent_loop/claude/20260608T152150/loops/loops002/exp/candidate_config.json`,
  with eval JSON `candidate_val_test_st1.json` and selection JSON
  `candidate_val_public_st1.json`. (No new model weights; the candidate is the
  base checkpoint + routing config + prompt + top-k + embedding settings.)
- **Primary blind-gate threshold:** selected-config ST1 Macro-F1 on
  `data/benchmarks/val_test.json` must be **strictly greater than BASE_GATE =
  0.802383**. A tie is not a pass.
- **Secondary (method-validity) threshold:** selected-config val_public Macro-F1
  must be **strictly greater than the frozen base a0 val_public 0.8031**. If the
  fallback only ties/loses to a0 on the selection split, the method adds no value
  and is rejected regardless of the blind number (no selection-on-the-gate, exactly
  the loop001 trap).
- **Selection rule:** pick the single config with the highest **val_public**
  Macro-F1; break ties toward the config that escalates **fewer** rows (lower LLM
  cost / less risk). Touch the blind gate exactly once for that config. The a3_b1
  secondary base is a diagnostic only; it cannot be used to pick between configs.
- **Tolerances (unchanged stages / inherited artifacts):** Stages 2–4 unchanged;
  their inherited cascade artifacts must be byte-identical (delta = 0.000 on every
  non-ST1 stage Macro-F1). The ST1 *high-confidence* path remains the data-only BERT
  forward pass; only low-confidence rows change.
- **Data-only runtime compliance:** the entire promoted runtime path consumes
  `data` ONLY — BERT forward over `data`; embedding index + query over train/eval
  `data` (`--info-mode data-only`, verified `get_info_text` returns `row["data"]`
  for `data-only`, lines 141–148); LLM prompt shows retrieved-example `data` text +
  the query `data` (the prompt's example `label` is the train row's
  `promise_status` GT, which is a label-as-fewshot-supervision use, permitted as an
  in-context demonstration label, NOT a model input feature of the query row).
  **promise_string is offline-only and MUST NOT appear at runtime**: `--info-mode`
  is locked to `data-only` and `--text-col data`; any config using `promise-str-only`
  or `data+promise_str` is INVALID and excluded.
- **Decision logic:** promote ONLY if (primary gate strictly passes) AND (secondary
  method-validity threshold passes) AND (data-only compliance verified) AND (no
  downstream regression) AND (reflect record + stored Claude review exist).
  Otherwise `reject` or `defer`. A missing / mis-baselined / label-derived /
  promise_string-leaking metric counts as a fail.
- **On promotion:** update `docs/methods.md` Stage 1 Module B (record selected
  threshold, prompt, top-k, info-mode=data-only, mode-context, embedding endpoint)
  and create/update `docs/loops/agent_loop_state.json` (BASE_GATE=0.802383, new ST1
  best artifact = the fallback config over a0) in the same work session. Otherwise
  leave both unchanged. Per CLAUDE.md, update `docs/feature_list.json` if a feature
  state changes.

## Data-Use Boundaries

- **Runtime input: `data` ONLY**, on every path: (a) BERT forward over `data`;
  (b) retrieval index + query embeddings over `data`; (c) LLM prompt body = query
  `data` + retrieved `data`. `--info-mode` LOCKED to `data-only`; `--text-col`
  LOCKED to `data`.
- **Retrieved-example labels:** the in-context examples carry the train rows'
  `promise_status` as demonstration labels (standard few-shot). This is permitted —
  it is the label of a *retrieved training row*, never the query row's annotation,
  and never a feature of the query. Ground-truth labels of eval rows are used ONLY
  for offline scoring.
- **promise_string / evidence_string / extracted spans / eval-row GT labels:**
  FORBIDDEN anywhere in the runtime path. Any config that reads them is INVALID.
  (promise_string offline use was only relevant to loop001 synthesis; this loop has
  no synthesis and no offline promise_string use at all.)
- **Split discipline:** selection on val_public ONLY; val_test touched once,
  post-selection. test.json never used.

## Failure Modes & Abandonment Criteria

- **Escalation does not help:** across all T, val_public Macro-F1 ≤ a0 base
  (0.8031). Evidence: LLM flips are net-neutral or net-harmful on routed rows
  (flip-correctness ≤ 50%). Decision: `reject`; record that confidence-routed
  LLM-RAG does not beat the frozen recipe BERT; recommend a0-only promotion as the
  fallback-free win. Abandon this family for the a0 base.
- **Gate-vs-selection divergence:** best val_public config fails the blind gate
  (val_test ≤ 0.802383). Decision: `reject` (do not hunt the gate for a passing
  config — that is selection-on-the-gate).
- **LLM overrides correct BERT rows:** raising T increases escalation but
  *lowers* Macro-F1 (LLM overturns rows BERT had right). Action: prefer the lowest-T
  arm; if even the lowest-T arm loses, abandon.
- **Endpoint / cost instability:** `http://192.168.1.79:3134` errors or rate-limits.
  `pred_by_bert_codex_rag.py` already keeps the BERT label on LLM error
  (`bert_fallback_codex_error`). If a non-trivial fraction of routed rows fall back
  to BERT, the run is INVALID for selection (incomplete escalation); re-run.
- **Compliance leak:** any config needing a non-`data` runtime field → INVALID,
  excluded from selection.
- **Method-family abandonment:** if NO (T,K,P,C,router) config beats both
  thresholds after val_public selection, decision-time LLM-RAG escalation is
  considered unproductive over this base; loop003 must move to a materially
  different family (e.g. calibration/threshold-decision on the BERT itself, or a
  representation/pooling-head change — all genuinely distinct from both loop001 and
  loop002).

## Compute / Environment Notes

- **GPU0 only** for all BERT scoring and any local embedding:
  `CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`, `--batch-size 4`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. GPU1 is occupied by the
  unrelated exp32 job; do not use it.
- **LLM endpoint:** `http://192.168.1.79:3134` (probed reachable, HTTP 200).
  Embedding backend per `pred_by_bert_codex_rag.py` defaults (OpenAI-compatible
  base url) or `--embedding-backend local` on GPU0; cache the train index once
  (`--embed-cache`) and reuse across all configs. Retrieval corpus =
  `data/raw_data/vpesg_4k_train_1000.json` (train `data` only).
- **Reuse, don't reinvent:** `core/human/predict/stage1/pred_by_bert.py` (scored
  CSV) → `core/human/predict/stage1/pred_by_bert_codex_rag.py` (routing+RAG+LLM).
  Score val_public/val_test once with the a0 base; vary only the fallback config.
  Offline scoring of the merged CSV vs GT via the same Macro-F1 used in
  `eval_bert.py` (sklearn `f1_score(average="macro")`).

## Parallel Subtasks
- id: base_score  phase: exp  description: Run pred_by_bert.py with --finetune-path models/loop001_st1_a0_real_only/best_st1.pt on val_public.json and val_test.json (GPU0, batch 4) to produce scored ST1 CSVs (score_yes/score_no per row); also score the a3_b1 base on val_public for the base-invariance diagnostic. BLOCKING for all fallback configs. Record base-only Macro-F1 as the no-routing control.
- id: embed_index  phase: dev  description: Build/cache the train data-only embedding index over data/raw_data/vpesg_4k_train_1000.json (--info-mode data-only, --embed-cache) once; reuse across all configs. Verify retrieval is data-only.
- id: fallback_grid_valpublic  phase: exp  description: Run the confidence-routed LLM-RAG fallback on the a0 base val_public scored CSV across the T-sweep then the K/P/C refinement + margin-router ablation; merge each config's CSV and compute val_public Macro-F1 + diagnostics (rows escalated, flips, flip-correctness). Selection only.
- id: select_and_gate  phase: exp  description: Pick the single best val_public config (tie-break: fewer escalations), verify it beats a0 base (0.8031) on val_public, then run that one config once on val_test.json; compare to BASE_GATE 0.802383 for the promotion decision. Run the base-invariance check (same config on a3_b1 base) as diagnostic only.
- id: methods_doc  phase: both  description: On promotion only, update docs/methods.md Stage 1 Module B (threshold/prompt/top-k/info-mode=data-only/mode-context/endpoint), create/update docs/loops/agent_loop_state.json (BASE_GATE=0.802383, new ST1 best), and docs/feature_list.json if feature state changes.
