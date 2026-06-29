# Loop 001 — Reflect

ST1 synthetic-data augmentation grid for the `hfl/chinese-roberta-wwm-ext-large`
classifier, gated against the `exp4` baseline. This record works the CLAUDE.md
Promotion Gate and the `reflect.md` checklist item-by-item (yes/no + evidence),
addresses the recipe-vs-synthesis tension explicitly, and ends with one verdict.

All headline numbers were re-verified from the raw eval JSON in
`loops/loops001/exp/` (not taken on faith from the exp markdown):

| metric | source JSON | value |
|---|---|---|
| BASE_GATE (exp4, val_test) | `baseline_val_test_st1.json` | 0.802383 |
| baseline exp4 val_public | `valpublic_eval/baseline_exp4_valpublic.json` | 0.787283 |
| a0_real_only val_public | `valpublic_eval/a0_real_only_valpublic.json` | 0.803069 |
| a3_b1 val_public | `valpublic_eval/a3_b1_valpublic.json` | 0.803069 |
| a0_real_only val_test (gate) | `gate_eval/a0_real_only_valtest.json` | 0.810977 |
| **a3_b1 val_test (gate)** | `gate_eval/a3_b1_valtest.json` | **0.823377** |

a0 and a3_b1 are an **exact** tie on the locked selection metric
(0.8030692029408333 to all printed digits).

## Split-discipline correction (accepted as established fact)

The plan designated `data/benchmarks/test.json` (n=200) as the dev selection
split. Re-verified here: test.json is **200/200 id-overlapped with the canonical
train file** `data/raw_data/vpesg_4k_train_1000.json`; val_public (n=500) and
val_test (n=500) have **0 overlap**. The exp step correctly discarded the
contaminated dev scores (0.96–0.99, pure memorization) and moved selection to the
clean, train-disjoint `val_public.json`, keeping `val_test.json` as the blind
gate. Selection therefore never touched the blind gate. This is sound.

## reflect.md checklist

**1. Did the primary metric exceed the acceptance threshold on the dev split?**
Yes — but with a caveat. The *valid* selection split is val_public (the
plan's nominal dev split test.json is contaminated and invalid). On val_public,
a3_b1 = 0.803069 > baseline 0.787283 (+0.0158). The plan's acceptance criterion
was "beat the gate baseline on the blind set"; on the selection set the candidate
clears the real-only control only by a tie, not strictly.

**2. Did the blind-gate metric (val_test Macro-F1) beat the baseline?**
Yes. a3_b1 = 0.823377 > BASE_GATE 0.802383, **+0.020994**, strictly greater.
Verified from `gate_eval/a3_b1_valtest.json`. This is the gate's single
hard pass/fail item and it passes.

**3. Is the generalization gap (selection − blind) within the allowed limit?**
Yes — and favorable. a3_b1 improves from val_public 0.803069 to val_test
0.823377 (+0.0203); a0 improves +0.0079. The candidate does *not* overfit the
selection set (gap is positive, i.e. it generalizes up, not down). No limit
breach.

**4. Data-use compliant — runtime uses ONLY `data`, promise_string strictly
offline?** Yes. Verified: `core/eval/eval_bert.py` ST1 path reads `d["data"]`
only (lines 73–92, 204). The candidate is a plain BERT forward pass over `data`.
The dev record documents that promise_string / PDF text were used only to author
synthetic training rows offline; A3 output rows carry only
`{id, data, promise_status}`, with leakage guards and 0/793 byte-identical
copies of any real promise_string. No runtime/eval/RAG/feature path consumes a
forbidden field. Compliant.

**5. Blind/test data excluded from tuning and threshold design?** Yes. Selection
used val_public only; val_test was touched once, post-selection, for exactly the
two tied-best arms. No threshold was tuned on either clean split (the manual_ce
ablation was not even reached). Split discipline holds.

**6. Genuinely new family vs prior loops?** Yes (vacuously). This is loop 1 of
this autonomous run; no prior loop plans exist. The Novelty Check correctly
treats exp4 as an external baseline, not a prior loop.

**7. Does docs/methods.md / agent_loop_state need updating?** Conditional on the
verdict below. Note: `docs/loops/agent_loop_state.json` **does not exist** in the
repo (verified via `find`), so there is no machine baseline record to update;
methods.md Stage 1 Module A currently points at the exp4 checkpoint.

**8. Next-loop advice.** See "Advice for loop 002" below.

## CLAUDE.md Promotion Gate — item-by-item

| gate requirement | satisfied? | evidence |
|---|---|---|
| `score_first_promote` gate present before dev | Yes | plan §score_first_promote |
| names current baseline + candidate artifact | Yes | baseline `models/exp4_optimize2_highconf_yes_balanced_no_large`; candidate `models/loop001_st1_a3_b1/best_st1.pt` (exists, 1.3 GB) |
| primary weighted/stage threshold defined | Yes | val_test Macro-F1 strictly > BASE_GATE |
| tolerances for unchanged stages | Yes | ST2–4 delta = 0.000; nothing downstream changed |
| data-only compliance for promoted path | Yes | data-only forward pass (item 4) |
| exp reports cmd, artifacts, scores, deltas, diagnostics | Yes | exp record + per-class No-F1/Yes-F1 |
| reflect verifies every gate item + Claude review | **Partial** | every gate item verified here; **the external Claude review step required by the Agent Looping Workflow was not run/stored** |
| no metric missing / mis-baselined / label-derived | Yes (numerically) | all re-verified from raw JSON against the correct clean baseline |

The score-first gate's hard conditions (primary blind gate strictly beats
baseline, data-only, no downstream regression) all **pass**. The one unmet
procedural requirement is the **external Claude review**: CLAUDE.md states a
reflect record "is incomplete" without the `claude --dangerously-skip-permissions`
review of methods.md + plans/dev/exp, and "Do not promote a loop result unless
the reflect record … includes or references the Claude review." That review does
not exist in `reflect/`. Under a strict reading of the gate this alone blocks an
`accept`.

## The recipe-vs-synthesis tension (the real decision)

This is the crux, and the honest reading is unfavorable to the loop's thesis:

- **On the locked, clean selection metric (val_public), real-only ties the best
  synthetic arm exactly** (a0 = a3_b1 = 0.803069). Every synthetic-heavy mix
  (A1/A2/A4 at B2/B3) lands *below* real-only. So synthetic augmentation, as a
  family, did **not** beat a clean real-only retrain on the metric the loop was
  allowed to select on.
- **a0_real_only already beats the exp4 baseline** on both clean splits
  (val_public +0.0158, val_test +0.0086). The improvement over the old baseline
  is therefore attributable to the **canonical-data training recipe**, not to
  synthesis.
- a3_b1's larger gate margin (+0.0210 vs a0's +0.0086) is the *entire* case for
  promoting the synthetic arm over the real-only arm. But that gap was measured
  on a single 500-row blind split, on two arms that are **tied on the selection
  metric**. With a tie at selection time, choosing a3_b1 over a0 because it won
  the blind split is effectively selecting on the blind split — exactly what the
  gate forbids in spirit. The +0.0124 a3_b1-over-a0 gate gap is plausibly noise
  (No-F1 0.7086 vs 0.6919 on ~135 negatives ≈ one or two flipped sentences).

So the defensible, data-honest conclusion is: **"real-only retrain on canonical
data beats the old baseline; synthetic augmentation is unproven — it ties at
best on clean selection and its gate edge is within noise of the control."** The
candidate a3_b1 passes the *number* but the loop's *method thesis* does not.

## Decision

Two things are simultaneously true: (a) a3_b1 strictly clears the blind gate
without the gate being used for selection — the one hard, score-first pass
condition is met; (b) the loop's reason-for-being (synthetic augmentation helps)
is **not** supported — it ties real-only at selection and the winning margin is
within noise, and the gain comes from the recipe.

Given that the **external Claude review required by the Promotion Gate is
missing** (a mandatory, non-waivable gate item → "the decision must be reject or
defer"), and that promoting the *synthetic* arm a3_b1 over the *real-only* arm
a0 would be unjustified selection-on-the-blind-gate when the two tie on the
locked metric, the correct verdict is **defer**, not accept.

Deferring does not throw away the win. The actionable, validated result of loop
001 is that **a clean canonical-data retrain (a0_real_only, data-only) beats the
exp4 baseline on both clean splits** and is a safer promotion candidate than the
synthetic arm. That promotion should be made through a proper score-first gate
with the Claude review attached — which is precisely loop 002's job below. Per
the reflect-gate rules, methods.md and the (absent) agent_loop_state.json are
left **unchanged** for this loop.

## Advice for loop 002 (concrete)

1. **Fix the standing dev split first (blocking).** Permanently retire
   `data/benchmarks/test.json` as a selection split (200/200 train-leaked).
   Adopt `val_public.json` as the canonical dev/selection split and keep
   `val_test.json` as the locked blind gate. Record this in the task spec /
   methods.md data-use notes so no future loop re-leaks. Also create
   `docs/loops/agent_loop_state.json` (it does not exist) with BASE_GATE =
   0.802383 as the immutable ST1 gate baseline.

2. **Isolate recipe vs. synthesis — the central unresolved question.** Run
   a0_real_only vs the exp4 baseline as a *controlled* comparison with multiple
   seeds (≥3, e.g. 42/1/2) for BOTH a0 and a3_b1. Report mean ± spread on
   val_public and a single locked val_test read. Hypothesis to kill: "a3_b1's
   gate edge over a0 is within seed noise." If the a3_b1−a0 gate gap is smaller
   than the seed std of a0 alone, declare synthesis unproven and promote
   **a0_real_only** (recipe win) instead — and do it through a real
   score_first_promote gate with the Claude review attached.

3. **Promote the safe, proven win.** Loop 002's primary deliverable should be to
   formally promote `models/loop001_st1_a0_real_only/best_st1.pt` (or its
   reproduction) as the new ST1 Module A checkpoint if the multi-seed run holds
   — it beats baseline on both clean splits and carries no synthesis-noise
   risk. Update methods.md Stage 1 Module A default checkpoint then.

4. **Do not run another synthesis grid.** Per CLAUDE.md non-repetition: loop 001
   already exhausted "vary synthesis source × mix weight." Heavier mixes hurt
   (B2/B3 all below real-only) — drop B3 entirely, and only revisit synthesis if
   a materially new objective is introduced (e.g. targeted hard-negative mining
   with a quality filter, or a different supervision target), not another
   source×mix sweep.

5. **When to introduce the LLM-RAG fallback (deferred from loop 001).** Introduce
   confidence-routed LLM-RAG escalation (methods.md Stage 1 Module B) as the loop
   003 family *after* the recipe-vs-synthesis question is settled and the
   strongest pure-BERT checkpoint is locked, so the fallback is measured as a
   marginal lift over the best BERT, not over a moving baseline. Gate it on the
   clean val_public selection split; route only the lowest-confidence rows; keep
   retrieval over `data` only.

6. **Always run the mandatory Claude review** before the reflect verdict in every
   future loop; loop 001's gate was otherwise close to promotable and the missing
   review is the kind of procedural gap the gate is designed to catch.

Verdict: defer
