# Stage 3 add_context ablations — results

Backend Qwen (`192.168.1.78:3132`), prompt `clear_notclear_with_context_scoped.txt`,
no gate (predict every row). Metric: GT-gated 2-class (Clear/Not Clear) Macro-F1
via `core/analysis/score_st3_full_coverage.py` (the 2-class prompt never emits N/A,
so full-coverage F1 is dominated by N/A gold rows and is not the comparison metric).

## 100-row screen (val_ctxhit.100, n2=67)

| rank | method | 2cls F1 |
|------|--------|---------|
| 1 | add_evidence_promise | 0.298 |
| 2 | add_evidence_string | 0.298 |
| 3 | add_context | 0.282 |
| 4 | add_context_window | 0.282 |
| 5 | add_image | 0.266 |

`add_context_window` == `add_context`: every val row's offsets hit_kind is
`hit_exact_window`, so both modes inject identical context.

## 600-row comparison (top-3, val_ctxhit.json, n2=397)

| rank | method | 2cls F1 | full F1 | F1_Clear | F1_NotClear |
|------|--------|---------|---------|----------|-------------|
| 1 | add_evidence_string | 0.2997 | 0.1651 | 0.2948 | 0.2003 |
| 2 | add_context | 0.2972 | 0.1629 | 0.2871 | 0.2017 |
| 3 | add_evidence_promise | 0.2871 | 0.1572 | 0.2723 | 0.1993 |

## Findings

1. **All three are statistically tied** (spread 0.013 on 600). No method
   meaningfully beats another.
2. **Annotation injection gives ~0 benefit.** `add_evidence_string` (leaks the
   `evidence_string` annotation, NOT data-only) beats the compliant `add_context`
   by only 0.0025 — within noise. So the data-compliant **`add_context` is the
   practical choice**; leaking annotations does not help.
3. **All methods lose to a trivial baseline.** The model predicts `Not Clear` for
   ~88% of rows (525–532 / 600), while the gold 2-class subset is 85% `Clear`
   (336:61). Trivial baselines on that subset:
   - always-`Clear`: 2cls Macro-F1 = **0.4584**
   - always-`Not Clear`: 2cls Macro-F1 = 0.1332
   Every context method (~0.30) is far below always-`Clear` (0.458).

## Conclusion

The bottleneck is the **prompt**, not the context source: the scoped prompt drives
a heavy `Not Clear` bias that makes the classifier worse than always-`Clear`.
Comparing context/image/annotation variants is premature until the prompt's class
bias is fixed (e.g. rebalance the Clear/Not Clear decision rule, add Clear-leaning
few-shots, or calibrate the threshold). Among the variants tested, `add_context`
is the one worth keeping (data-compliant, tied with the best).

Artifacts: `preds/exp_scores_100.json`, `preds/exp_scores_600.json`,
`preds/full_*.csv`, `experiments/logs/full_*.log`.

---

# Prompt redesign — fixing the Not-Clear bias

Root cause (above): the scoped prompt over-predicts `Not Clear`. Designed three
tag-aware prompts with a generous `Clear` rule (any concrete anchor — year, number,
named standard/program, or specific accomplished/ongoing action → Clear; default to
Clear when unsure). All run with `add_context` (data-compliant).

## 100-row screen (val_ctxhit.100, add_context, n2=67)

| rank | prompt | 2cls F1 | full F1 | F1_Clear | F1_NotClear | pred dist (Clear:NotClear) |
|------|--------|---------|---------|----------|-------------|------|
| 1 | clear_lenient_tagged | 0.714 | 0.369 | 0.760 | 0.348 | 65:35 |
| 2 | clear_checklist_tagged | 0.658 | 0.347 | 0.743 | 0.296 | 57:43 |
| 3 | clear_fewshot_tagged | 0.645 | 0.337 | 0.750 | 0.261 | — |
| – | scoped (old) | 0.282 | 0.155 | 0.250 | 0.214 | 8:92 (inverted) |

## 600-row validation (val_ctxhit.json, add_context, n2=397)

| prompt | 2cls F1 | full F1 | F1_Clear | F1_NotClear |
|--------|---------|---------|----------|-------------|
| **clear_lenient_tagged** | **0.658** | 0.342 | 0.733 | 0.292 |
| clear_checklist_tagged | 0.635 | 0.333 | 0.724 | 0.273 |
| scoped (old baseline) | 0.297 | 0.163 | 0.287 | 0.202 |
| trivial always-Clear | 0.458 | – | – | – |

## Outcome

`clear_lenient_tagged` + `add_context` lifts 2-class Macro-F1 from **0.297 → 0.658**
on the full 600 (full-coverage 0.163 → 0.342), beating the trivial always-Clear bar
(0.458) by a wide margin. Not-Clear F1 also rises (0.202 → 0.292), so the gain is
genuine calibration, not a blind Clear guess. The bottleneck was the prompt, as
diagnosed; context source choice is secondary. Winner: **clear_lenient_tagged.txt**.

Prompt artifacts: `prompts/clear_lenient_tagged.txt`, `clear_checklist_tagged.txt`,
`clear_fewshot_tagged.txt`; scores `preds/prompt_scores_{100,600}.json`;
preds `preds/pp_*.csv` (100), `preds/win_*.csv` (600).

---

# Method ablation UNDER the good prompt (clear_lenient_tagged)

Re-ran the context-source ablation now that the prompt bias is fixed, to honestly
answer "does the image / context / evidence_string help?".

## 100-row screen vs 600-row validation (2cls Macro-F1)

| method | 100-row | 600-row | 600 pred (Clear:NotClear) |
|--------|---------|---------|------|
| data_only | 0.663 | **0.6927** | 472:128 |
| context_image | 0.745 | 0.677 | — |
| add_image | 0.647 | 0.660 | 479:121 |
| add_context | 0.729 | 0.658 | 394:206 |
| add_evidence_str (non-compliant) | 0.679 | — | — |

## Findings (600 is the trustworthy read)

1. **The 100-row screen was misleading.** add_context led at 100 (0.729) but is LAST
   at 600 (0.658); data_only was 4th at 100 (0.663) but FIRST at 600 (0.693). Small
   samples flip the ranking — only the 600-row numbers are reliable.
2. **No external context helps.** data_only (just the `<data-prompt>` sentence) is the
   best on 600. Same-page text, page image, and both are all BELOW it.
3. **Why context hurts:** same-page OCR nudges the model toward `Not Clear`
   (128 -> 206 Not-Clear predictions), but gold is ~85% Clear, so the extra Not-Clear
   calls cost more than they gain. The page image neither helps (alone: 0.660 < 0.693)
   nor rescues context (context_image 0.677 < 0.693).
4. **evidence_string** (non-compliant) gave only a marginal 100-row bump and is dropped.

## Final conclusion

The ONLY lever that mattered was the **prompt** (scoped 0.297 -> lenient 0.693 on 600,
data-only). Same-page-content, page image, and annotation injection are all net-neutral
to net-negative on full data. **Best deployable config: `clear_lenient_tagged.txt` +
data-only (no context, no image)** — also the simplest and fully `data`-only compliant.

Method artifacts: `preds/m_*.csv` (100), `preds/m6_*.csv` (600),
`preds/method_scores_{100,600}.json`.
