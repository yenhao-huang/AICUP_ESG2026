# Loop 002 — Reflect

Confidence-routed LLM-RAG fallback layered on the frozen loop001 a0 BERT
(`models/loop001_st1_a0_real_only`). ST1 promise_status. Selection on
`val_public` only; `val_test` is the blind gate (touched once). Runtime input is
`data` only on every path.

## Independent verification (not recomputed from the grid)

Re-scored the actual artifacts with sklearn `f1_score(labels=["Yes","No"],
average="macro")`:

| artifact | split | Macro-F1 | source split |
|---|---|---|---|
| `exp/.../base/a0_val_public_scored.csv` | val_public | 0.803069 | a0 base |
| `exp/.../base/a0_val_test_scored.csv` | val_test | 0.810977 | a0 base |
| `exp/.../gate/T0.80_K5_strictneg_bertctx_valtest/merged.csv` | val_test | **0.811566** | 470 `bert_high_conf` + 30 `codex_rag_low_conf` |

All three reproduce the exp record exactly. The gate merged CSV's `source` column
confirms only 30/500 rows were routed to the LLM and `codex_errors=0` (no silent
BERT-fallbacks), so the run is valid for the gate. `changed_by_codex=13` on the
blind set.

Reference numbers (all verified or inherited):
- exp4 BASE_GATE (immutable, data-only) val_test = **0.802383**; val_public 0.7873.
- Frozen a0 base: val_public **0.803069**, val_test **0.810977**.
- Selected fallback `T0.80_K5_strictneg_bertctx`: val_public **0.817436**,
  val_test **0.811566**.

## Evaluation checklist (yes/no + evidence)

1. **Primary metric exceeds the dev acceptance threshold?** YES. val_public
   0.817436 > a0 base 0.803069 (+0.014367) and > exp4 0.7873. Inverted-U peak at
   T=0.80, K=5, strict_negative, bert-prediction.

2. **Blind-gate metric beats baseline?** YES vs the official gate baseline:
   val_test 0.811566 > BASE_GATE 0.802383 (**+0.009183**). Independently
   re-scored. NOTE the honesty caveat below — vs the a0 base it actually layers
   on, the blind lift is only +0.000589 (~1 row).

3. **Generalization gap within limit?** YES. dev − blind = 0.817436 − 0.811566 =
   **+0.005870** (small, positive; no overfit to the selection split). The plan
   set no explicit numeric gap cap; this magnitude is well inside any reasonable
   bound and consistent with the a0 base's own +0.0079 public→test direction.

4. **Data-use compliant — runtime `data` only, promise_string offline only?**
   YES. `--info-mode data-only --text-col data` locked on every run; gate CSV
   `source` is `bert_high_conf`/`codex_rag_low_conf` only. BERT forward over
   `data`; ModernBert index embeds `row["data"]` only (dev verified 5/5
   neighbours matched train `data`, none matched `promise_string`); LLM prompt
   body = query `data` + retrieved train `data`. Few-shot example labels are the
   retrieved TRAIN rows' `promise_status` (permitted in-context demonstration
   labels, not a query feature). GT labels used for OFFLINE scoring only. No
   config used `promise-str-only` / `data+promise_str`. This loop runs no
   synthesis, so there is no offline promise_string use at all. **PASS.**

5. **Blind/test excluded from tuning & threshold design?** YES. All T/K/P/C/router
   selection was on val_public; val_test was touched exactly once for the single
   selected config (one gate merged CSV exists). test.json never used. **PASS.**

6. **Genuinely new family vs prior loops?** YES. loop001 changed the BERT
   training set (synthesis source × mix). loop002 freezes the BERT and adds a
   decision-time confidence router + RAG retrieval + a second model (LLM judge).
   This is the runtime-routing family explicitly named as distinct in the loop001
   reflect. Not a hyperparameter sweep of loop001. **PASS.**

7. **Does methods.md / agent_loop_state need updating?** Only on promotion. This
   verdict is **defer**, so methods.md, feature_list.json, and agent_loop_state
   are left UNCHANGED. (`docs/loops/agent_loop_state.json` does not yet exist;
   creating it is a promotion-only action.)

8. **Next loop?** See "Advice for loop 003" below.

## CLAUDE.md Promotion Gate — item-by-item

The plan's `score_first_promote` gate requires promote ONLY if ALL of:
(primary gate strictly passes) AND (secondary method-validity passes) AND
(data-only verified) AND (no downstream regression) AND (**reflect record +
stored Claude review exist**).

| gate item | required | observed | result |
|---|---|---|---|
| `score_first_promote` present in plan before dev | yes | present (plan §score_first_promote) | PASS |
| Baseline artifact named | exp4 default | `models/exp4_optimize2_highconf_yes_balanced_no_large`, BASE_GATE 0.802383 | PASS |
| Candidate artifact path named | yes | `candidate_config.json` (a0 ckpt + routing/prompt/topk/embed cfg) | PASS |
| Primary blind threshold val_test > 0.802383 (strict) | strict > | 0.811566 (+0.009183) | PASS |
| Secondary val_public > a0 0.803069 (strict) | strict > | 0.817436 (+0.014367) | PASS |
| Selection rule (highest val_public, tie→fewer esc) | yes | unique max, no tie-break needed | PASS |
| Downstream ST2–4 unchanged (δ=0) | yes | ST1-only change; cascade artifacts untouched | PASS |
| Data-only runtime compliance | yes | verified (item 4) | PASS |
| Flip-correctness floor > 0.50 | yes | 0.562 (net +2) — thin | PASS (marginal) |
| Endpoint stability (codex_errors=0) | yes | 0 on every run incl. gate | PASS |
| **Reflect record exists** | required | this file | PASS |
| **Stored external Claude review under `reflect/`** | **MANDATORY, non-waivable** | **ABSENT — `reflect/` contained no review before this file** | **FAIL** |

`reflect/` directory listing taken at reflect time held no Claude review file
(only this record). CLAUDE.md is explicit: "Reflect must include an external
Claude review step… Store Claude's review… under
`docs/loops/loops<loop-id>/reflect/`. A reflect record is incomplete if it does
not discuss `docs/methods.md` and the end-to-end method." And: "Do not promote a
loop result unless the reflect record … includes or references the Claude
review." The reflect prompt for this loop restates it as non-waivable. **No
verified, stored Claude review exists, so the gate's final required item fails.**

## End-to-end assessment vs docs/methods.md (Stage 1)

The promoted-if-accepted runtime is exactly methods.md Stage 1 Module B (BERT +
Codex/LLM-RAG hybrid): `pred_by_bert.py` scored CSV → `pred_by_bert_codex_rag.py`
routing. loop002 only swaps the base checkpoint to `loop001_st1_a0_real_only`,
fixes the router (T=0.80 abs-conf, K=5, strict_negative prompt, bert-prediction
context), and uses the HTTP LLM backend + data-only ModernBert retrieval index
added in dev. The high-confidence path is the unchanged data-only BERT forward;
only 30/500 (6%) blind rows change. Stages 2–4 inherit byte-identical cascade
artifacts (ST1-only change). So the method is coherent and documentable — but
two things weigh against promoting it now even setting aside the missing review:

- **Marginal blind value of the LLM layer.** Over the a0 base it layers on, the
  fallback's blind lift is +0.000589 (~1 row of 500); flip-correctness 0.562 is
  net +2 on val_public. The +0.0092 gate win over exp4 is overwhelmingly the a0
  recipe (already established in loop001), not the LLM machinery. Promoting
  "LLM-RAG fallback" as the win over-credits the LLM layer for what is mostly an
  a0 effect. The base-invariance diagnostic (a3_b1: +0.005411 val_public) shows
  the lift direction is real but small.
- **Selection-split-specific gain.** The +0.0144 val_public lift collapses to
  ~1 row on the blind set — the fallback's benefit does not transfer, which is
  itself weak evidence that the router/prompt is tuned to val_public idiosyncrasy
  rather than a robust correction rule.

## Decision: DEFER

Rationale, in order of force:

1. **Non-waivable gate item fails.** The mandatory external Claude review is not
   stored under `reflect/`. By CLAUDE.md and this loop's own
   `score_first_promote` gate, a missing required item forces `reject` or
   `defer`, never `promote`. This alone bars promotion.
2. **Honest marginal value is ~1 row blind.** Even if the review existed,
   promoting the *LLM-RAG fallback pipeline* would over-credit the LLM layer:
   its blind lift over a0 is +0.000589, inside noise. The disciplined,
   data-only, reproducible win that genuinely clears the exp4 BASE_GATE on the
   blind set (+0.009183) is the **a0 recipe BERT**, not the fallback.

`defer` (not `reject`) is correct because the method is data-only compliant,
split-disciplined, a legitimately new family, and clears the official gate; the
only true blockers are a fixable process gap (run+store the Claude review) and
the need for more blind-set evidence that the LLM layer adds value beyond a0.
Nothing here warrants abandoning the family.

This is a defer, so **no** changes were made to `docs/methods.md`,
`docs/feature_list.json`, or `docs/loops/agent_loop_state.json`.

### What the orchestrator should do to clear the defer

- Run `claude --dangerously-skip-permissions "<review prompt over methods.md +
  loops002 plans/dev/exp>"` and store the output under
  `reflect/002_claude_review_<ts>.md` (CLAUDE.md mandatory step). Re-open the
  verdict only after it exists.
- Independent of loop002, the proven a0 recipe BERT
  (`models/loop001_st1_a0_real_only`, val_test 0.810977 > BASE_GATE 0.802383)
  is the honest data-only ST1 win and is a candidate for a thin, dedicated
  promotion of the **base** with the fallback documented as an optional add-on —
  but that is loop001's deferred a0 promotion, not loop002's pipeline, and
  should not be smuggled in under the loop002 label.

## Advice for loop 003

The decision-time-escalation family is not exhausted, but its blind value is too
thin to promote as-is. loop003 should make the routed correction robust and/or
escalate the base, with these concrete options (pick one as the distinct family):

1. **Calibration / threshold-decision on the BERT itself (no LLM).** A genuinely
   new family vs loop001/002: temperature-scale or isotonic-calibrate the a0
   logits on val_public, then pick the decision threshold / abstain band by
   calibrated confidence rather than raw softmax. Grid: calibration method
   {temperature, isotonic, beta}, decision-threshold sweep, abstain-band width,
   per-class (Yes/No) threshold. This directly targets the `No`-F1 weakness
   (0.69) that the LLM layer barely moved, with zero LLM cost — and is the
   cleanest path to a base whose own blind score beats a0.
2. **Make the LLM correction trustworthy before re-testing routing.**
   Require LLM-flip agreement: only accept a flip when the LLM AND a second
   data-only signal (e.g. retrieval-label majority of the top-k neighbours)
   agree; otherwise keep BERT. This should raise flip-correctness above the thin
   0.562 and is the variant most likely to convert a val_public-only gain into a
   blind-set gain. If even this does not beat a0 blind by a clear margin,
   abandon LLM escalation on this base.
3. **Stronger / second retrieval index.** The qwen3-embedding-8b endpoint was
   down; if it returns, add it as a second data-only index and ablate
   embedding-model × distance × weighting on the SAME routed rows — but only
   inside option 1/2's framing, not as a standalone re-run of loop002.

Whichever is chosen, the plan must (a) include `score_first_promote` before dev,
(b) cite this loop's review once it exists, (c) keep val_public-only selection
with one blind touch, and (d) report the lift **over the a0 base on the blind
set**, not just over exp4 — the loop002 lesson is that the exp4 gap is mostly a0,
so the honest bar is beating a0 blind by more than ~1 row.

Verdict: defer
