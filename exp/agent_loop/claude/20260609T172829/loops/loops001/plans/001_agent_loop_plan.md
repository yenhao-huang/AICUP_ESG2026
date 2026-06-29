# Loop 001 — ST4 Few-Shot Prompt Engineering with Mined Boundary Examples

## Novelty Check

Closest prior loops: ST1 optimization (`exp/agent_loop/claude/20260608T152150`, loop001) and ST2
optimization (`exp/agent_loop/claude/20260608T174056`, loops001-003). Those loops targeted entirely
different stages (promise identification and evidence support), using BERT fine-tuning, synthetic
data augmentation, and LLM-RAG routing for binary classification. This is the first ST4
(`verification_timeline`) optimization loop. No prior ST4 loop exists in this or any referenced
workspace. The method family — few-shot Codex prompt engineering for a 5-class timeline
classification task — is materially different from all prior work; no non-repetition concern.

---

## Method Family

**Calibrated few-shot Codex prompt with mined boundary-class examples and explicit year-band decision trees.**

The baseline prompt (`stage4_balance_rule_v1.txt`) is a zero-shot rule-based system prompt with
verbal decision rules for year mapping. Its critical failure modes are:

1. **`within_2_years` under-recall (R=0.500, P=0.143)**: the model massively over-predicts
   `between_2_and_5_years` for 2025/2026-year targets (118 FP). The boundary rule "2025/2026 →
   within_2_years, 2027-2029 → between_2_and_5_years" exists in the prompt but is overwhelmed
   by the zero-shot tendency to map any management or supply-chain forward-looking text with no
   extreme year to `between_2_and_5_years` (the most frequent labelled non-N/A class in train).

2. **`already` under-recall (R=0.358, 115 FN)**: completed/current-period events are pushed to
   `between_2_and_5_years`. The model correctly identifies obvious "已完成/已達成" language but
   fails on ongoing activities described in present tense, periodic routines, 2024-performance
   reports, and policies in force — all of which annotators classified as `already`.

The method family addresses both failure modes by injecting concrete in-context (few-shot)
examples mined from the training split, chosen to cover the exact confusion boundaries. The key
hypothesis is that showing the model 2-4 gold-annotated examples per problematic class — especially
borderline cases that look like `between_2_and_5_years` but are `within_2_years` or `already` —
provides stronger signal than verbal rules alone.

This is materially different from all prior ST4 prompt variants:
- `default.txt`: zero-shot, over-counts already (if持續→already), no boundary sharpening.
- `few_shot.txt`: has 4 synthetic/generic examples; does not target the specific confusion pairs.
- `strict_negative.txt`: conservative zero-shot; improves within_2_years boundary? — untested on val_test.
- `stage4_balance_rule_v1.txt` (baseline): zero-shot with balance rules; identified failure modes above.

The new family introduces: (a) **gold-mined examples** from training data (not synthetic), (b)
**boundary-targeted selection** (examples chosen because they sit at the within_2_years /
between_2_and_5_years boundary and the already / between_2_and_5_years boundary), and (c)
**CoT vs. direct-answer ablation** to measure whether chain-of-thought improves boundary reasoning.

---

## Training Data Analysis

ST4 class distribution in `data/raw_data/vpesg_4k_train_1000.json` (n=1000):

| Label | Count | Pct (of 814 non-empty) |
|-------|-------|------------------------|
| already | 366 | 44.9% |
| between_2_and_5_years | 238 | 29.2% |
| more_than_5_years | 197 | 24.2% |
| within_2_years | 13 | 1.6% |
| (N/A / blank) | 186 | — |

`within_2_years` is severely under-represented (13 examples, 1.6%). This imbalance explains why
the model defaults to `between_2_and_5_years` when a 2025/2026 year is present but the text
is ambiguous — it has seen almost no within_2_years examples.

Key observations for few-shot selection:
- **within_2_years signals**: explicit "2025" or "2026" year in a next-step/target context;
  "展望2025年"; "預計2025年"; "自2025年起"; "於2026年". 12/13 training examples have an explicit
  2025 or 2026 year. One has a 2024 annual target that sets 2025 as the next checkpoint — this is
  the most confusable with `already`.
- **already signals**: 2024 performance results ("2024年實績為"); ongoing activities described
  without a future year ("定期"; "每年"; "持續推動"); policies/systems currently in force. Most
  confusable already examples have NO explicit completion verb (already_unclear = 356/366).
- **Confusion boundary**: when a text mentions 2025 as a continuation of 2024 activity with
  "持續" language, both `already` (ongoing current) and `within_2_years` (next-year target) are
  plausible. Gold labels resolve this by the primary promise year.

---

## Variant Grid

All variants share the base architecture: system prompt + per-row user message containing only
`DATA: <data field>`. No post-processing. No derived fields. Selection on `val_public.json`;
gate on `val_test.json`.

### V1 — Boundary-Targeted Few-Shot (4 examples, direct-answer)

Structure: existing `stage4_balance_rule_v1` rules + 4 mined gold examples, one per most
confusable pair:
- `within_2_years` example with explicit 2025 year (vs. "looks like between_2_and_5_years" trap)
- `already` example with ongoing activity, no explicit completion verb
- `between_2_and_5_years` example with no year (management routine default)
- `more_than_5_years` example with 2030/2050 explicit year

Selection rationale: 4 examples is minimal context overhead; covers the two primary failure modes.

### V2 — Expanded Few-Shot (8 examples, direct-answer)

Structure: V1 rules + 8 mined examples (2 per class):
- `within_2_years` × 2: (a) explicit 2025 target, (b) explicit 2026 target in milestone context
- `already` × 2: (a) 2024 annual results with metric achieved, (b) periodic policy in force
- `between_2_and_5_years` × 2: (a) no year, supply-chain management, (b) 2027-2029 explicit
- `more_than_5_years` × 2: (a) 2030 SBT, (b) 2050 net-zero commitment

Hypothesis: 2 examples per class reduces within_2_years confusion better than 1; risk is context
length increases latency and cost.

### V3 — CoT Few-Shot (4 examples with chain-of-thought reasoning, then label)

Structure: same 4 boundary examples as V1 but each example includes a 1-sentence reasoning step
before the label. Format:
```
DATA: <example text>
思考：[year identification / completion-signal reasoning]
輸出：within_2_years
```

For the actual prediction, ask for `思考：... 輸出：<label>` output, then strip reasoning.
Hypothesis: CoT forces the model to explicitly identify the year anchor and completion state
before labelling, reducing within_2_years vs between_2_and_5_years confusion.

### V4 — Boundary-Sharpened Rules Only (no few-shot, enhanced rules)

Structure: rule-only variant that sharpens the `stage4_balance_rule_v1` ruleset at the two
failure boundaries:
- Add explicit rule: "TEXT 中出現 2025 或 2026 作為主要目標完成年份 → within_2_years（即使文字同時涉及持續管理或供應鏈活動）"
- Add explicit rule: "TEXT 描述 2024 年績效數據、已達成指標、通過查核、完成評鑑，且無明確未來目標年份 → already"
- Add priority ordering: year-anchor lookup BEFORE activity-type classification

This variant ablates whether rules alone (without examples) can fix the identified failure modes.
It is the most token-efficient variant and forms the upper bound for rule-only improvement.

### V5 — Asymmetric Few-Shot: within_2_years Oversampled (6 examples, direct-answer)

Structure: V1 rules + 6 examples with asymmetric class weights targeting the rarest class:
- `within_2_years` × 3: (a) 2025 explicit, (b) 2026 explicit, (c) ambiguous 2025/within-context
- `already` × 1: strongest signal example
- `between_2_and_5_years` × 1: clearest no-year management example
- `more_than_5_years` × 1: 2030/2050 example

Hypothesis: oversampling the rarest class (1.6% of training, highest FP rate at inference) most
directly counteracts the model's prior bias. Risk: might over-correct within_2_years and push
between_2_and_5_years/already recall down.

### V6 — CoT Extended Few-Shot (8 examples with CoT reasoning)

Structure: V2 (8 examples) + V3 (CoT format). Full 2-per-class coverage with explicit reasoning
step. This is the most expensive variant in prompt tokens.

Hypothesis: combined coverage + reasoning provides the strongest signal but highest cost.
If V3 and V2 both improve over V1, V6 may be the best overall.

---

## score_first_promote Gate

### Baseline Artifact

```
docs/loops/agent_loop_state.json: no ST4 entry yet
Baseline artifact: exp/integrated_stage_predictions/best_comb/detail_method/prompts/stage4_balance_rule_v1.txt
Baseline val_test ST4 Macro-F1: 0.5109
Baseline per-class: already=0.454, within_2_years=0.222, between_2_and_5_years=0.575, more_than_5_years=0.603, N/A=0.701
```

### Candidate Artifacts

New prompt files written to:
```
configs/prompt/stage4/codex/few_shot_boundary_v1_v1.txt   (V1)
configs/prompt/stage4/codex/few_shot_boundary_v2_v2.txt   (V2)
configs/prompt/stage4/codex/few_shot_cot_v3.txt           (V3)
configs/prompt/stage4/codex/boundary_rules_v4.txt         (V4)
configs/prompt/stage4/codex/few_shot_asym_v5.txt          (V5)
configs/prompt/stage4/codex/few_shot_cot_ext_v6.txt       (V6)
```

### Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| ST4 val_test Macro-F1 | > 0.5109 | primary acceptance gate |
| ST4 val_public Macro-F1 | > baseline val_public ST4 | selection must also improve; generalization gap ≤ 0.05 |
| within_2_years F1 (val_test) | improvement vs 0.222 | target class; delta expected > +0.05 |
| already F1 (val_test) | improvement vs 0.454 | target class; delta expected > +0.03 |
| between_2_and_5_years F1 (val_test) | no regression > −0.03 | most frequent class; must not collapse |
| more_than_5_years F1 (val_test) | no regression > −0.03 | stable class |
| N/A F1 (val_test) | no regression > −0.03 | cascaded from ST1 |

Integrated weighted score (0.20*ST1 + 0.30*ST2 + 0.35*ST3 + 0.15*ST4): ST4 weight=0.15; a
+0.03 ST4 Macro-F1 gain yields approximately +0.0045 weighted score, which is measurable.
ST1/ST2/ST3 scores are expected to be byte-identical when only the ST4 Codex prompt changes
(no upstream model swap); confirm this as a gate item (expected delta = 0.000 for ST1; confirm
ST2/ST3 change only due to cascade penalty re-routing if any ST4 gate changes).

**Reject criteria**: if best variant ST4 val_test Macro-F1 ≤ 0.5109, or if within_2_years F1 is
worse than baseline AND already F1 is worse than baseline, the method family fails and loop 001
result is rejected.

**Defer criteria**: if ST4 val_public improves but val_test does not (sign of selection-split
overfitting), result is deferred pending analysis.

---

## Data-Use Boundaries

Allowed at runtime:
- `data` field from input JSON only.
- Training data `data/raw_data/vpesg_4k_train_1000.json` used **offline** for few-shot example
  mining (selecting gold-annotated examples). The selected examples appear as in-context
  demonstrations in the prompt, using the `data` text only — no annotation fields (labels) are
  passed as model input beyond the gold label shown in the example output line.
- Predicted upstream labels (ST1 `promise_status`) used only for cascade gating (blank ST4 when
  ST1 != Yes); they are not passed into the ST4 prompt.

Forbidden at runtime:
- `evidence_string`, `promise_string`, extracted evidence, extracted promise
- Ground-truth labels from val splits
- Post-processing keyword rules (no rule-based label correction after model output)
- Any derived annotation field

Note on few-shot examples: the gold label for each in-context example (`already`, `within_2_years`,
etc.) is shown as the expected output in the demonstration, which is standard few-shot prompting.
This is **not** a violation of data-use rules because (a) the examples are from the training split,
not the evaluation split, and (b) the label is the output target, not an input feature.

---

## Failure Modes and Abandonment Criteria

| Failure Mode | Detection | Response |
|---|---|---|
| Few-shot examples increase context length past model limit | Token count check before run | Trim to V1 (4 examples) or shorten example texts |
| CoT reasoning leaks non-`data` content | Manual inspection of 10 random outputs | Revise CoT instruction to restrict reasoning to year + signal only |
| within_2_years P collapses while R improves | Per-class P/R check | Indicates over-correction; try V4 (rules-only) or reduce within_2_years shot count |
| already F1 degrades significantly | val_test already F1 < 0.40 | CoT or 8-shot is confusing the model; fall back to 4-shot direct-answer |
| All variants at or below baseline | val_test Macro-F1 ≤ 0.5109 for all 6 | Abandon method family; next loop should explore BERT fine-tuning calibration or label-smoothed ST4 training objective |
| N/A class collapses | N/A F1 < 0.60 on val_test | Model conflating N/A (ST1=No cascade) with within_2_years; add N/A example or explicit N/A instruction |

**Abandonment**: if all 6 variants fail to beat the baseline ST4 Macro-F1 = 0.5109 on val_test, this
loop is closed as a negative result. The reflect record must document per-class failure analysis
and recommend whether the root cause is:
(a) prompt-only ceiling reached (→ next loop: BERT ST4 calibration / threshold tuning), or
(b) model (gpt-5.5) lacks sufficient temporal reasoning for this task (→ next loop: different LLM
or BERT-ensemble), or
(c) train distribution (only 13 within_2_years examples) is too sparse for in-context learning
(→ next loop: BERT fine-tuning on balanced ST4 or class-weighted loss).

---

## Experiment Protocol

1. **Dev phase**: write 6 prompt files to `configs/prompt/stage4/codex/`.
2. **Exp phase**: for each variant, run ST4 Codex prediction on `data/benchmarks/val_public.json`
   (selection split), compute per-class Macro-F1. Select best 2 variants by val_public ST4 F1.
3. **Gate phase**: run best 2 variants on `data/benchmarks/val_test.json`. Report per-class F1
   and compare against all gate thresholds.
4. **Reflect phase**: write reflect record with Claude external review, gate verification, and
   next-loop recommendation.

Selection script: `core/e2e/stage4.py --method codex --prompt-file <file> --data val_public.json`.
Gate script: `core/e2e/stage4.py --method codex --prompt-file <file> --data val_test.json`.

---

## Parallel Subtasks

The following experiment variants can be run in parallel (independent Codex predictions, no
shared state):

```
Parallel group A — val_public selection runs:
  A1: core/e2e/stage4.py --method codex --prompt-file configs/prompt/stage4/codex/few_shot_boundary_v1.txt --data val_public
  A2: core/e2e/stage4.py --method codex --prompt-file configs/prompt/stage4/codex/few_shot_boundary_v2.txt --data val_public
  A3: core/e2e/stage4.py --method codex --prompt-file configs/prompt/stage4/codex/few_shot_cot_v3.txt --data val_public
  A4: core/e2e/stage4.py --method codex --prompt-file configs/prompt/stage4/codex/boundary_rules_v4.txt --data val_public
  A5: core/e2e/stage4.py --method codex --prompt-file configs/prompt/stage4/codex/few_shot_asym_v5.txt --data val_public
  A6: core/e2e/stage4.py --method codex --prompt-file configs/prompt/stage4/codex/few_shot_cot_ext_v6.txt --data val_public

Parallel group B — val_test gate runs (only after group A selects top-2):
  B1: val_test run for best variant from A
  B2: val_test run for second-best variant from A

Development prerequisite (must complete before A):
  D0: Write all 6 prompt files to configs/prompt/stage4/codex/
```
