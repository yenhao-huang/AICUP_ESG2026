# Loop 001 — Dev

## Summary

Implemented 6 prompt variants for ST4 (`verification_timeline`) Codex classification, targeting the two primary baseline failure modes identified in the plan:
1. `within_2_years` under-recall (R=0.500): model over-predicts `between_2_and_5_years` for 2025/2026 targets.
2. `already` under-recall (R=0.358): model pushes ongoing/periodic activities to `between_2_and_5_years`.

---

## Files Created

| File | Variant | Description |
|------|---------|-------------|
| `configs/prompt/stage4/codex/few_shot_boundary_v1.txt` | V1 | Baseline rules + 4 boundary-targeted gold examples (1 per class), direct-answer |
| `configs/prompt/stage4/codex/few_shot_boundary_v2.txt` | V2 | Baseline rules + 8 gold examples (2 per class), direct-answer |
| `configs/prompt/stage4/codex/few_shot_cot_v3.txt` | V3 | Baseline rules + 4 boundary examples with CoT reasoning (思考 + 輸出) |
| `configs/prompt/stage4/codex/boundary_rules_v4.txt` | V4 | Enhanced rules only (no few-shot), with explicit 4-step priority ordering |
| `configs/prompt/stage4/codex/few_shot_asym_v5.txt` | V5 | Baseline rules + 6 examples (3x within_2_years oversampled), direct-answer |
| `configs/prompt/stage4/codex/few_shot_cot_ext_v6.txt` | V6 | Baseline rules + 8 examples (2 per class) with CoT reasoning |

---

## Few-Shot Examples Selected

All examples are mined from `data/raw_data/vpesg_4k_train_1000.json` using the `data` field text and `verification_timeline` ground-truth labels.

### within_2_years Examples

**Example A — train index [345]** (used in V1, V2, V3, V4, V5, V6)
- Text: 台達 2024 LTIFR KPI achieved (0.38 vs target 0.48), now setting 2025 KPI target
- Selection rationale: **Most confusable with `already`** — contains a 2024 performance result that was achieved, but the forward-looking promise is to set a 2025 KPI. Model must learn that when 2024 is an achieved metric and 2025 is the next target, the primary promise year is 2025 → `within_2_years`.

**Example B — train index [256]** (used in V2, V5, V6)
- Text: 台泥 2025 Q1 platform integration and self-audit scope expansion
- Selection rationale: Explicit "自2025年第一季起" language with a new operational capability. Represents a clear 2025 system launch promise; no 2024 baseline comparison makes it less confusable but demonstrates "starting from 2025" phrasing.

**Example C — train index [976]** (used in V2, V5, V6)
- Text: 聯電 2024 water conservation results achieved, 2025 continuation pledge
- Selection rationale: Has both 2024 year AND 2025 continuation context. The "2025 年也預計透過相關措施，持續降低" signals 2025 as a pledge continuation year, not just an activity report. This is a boundary case where "持續" could be read as `already` but label is `within_2_years`.

### already Examples

**Example A — train index [20]** (used in all variants)
- Text: 自2005年起溫室氣體盤查，持續查核，2024年排放下降3.2%
- Selection rationale: **Prototypical ongoing activity with periodic results** — describes a long-standing practice with 2024 year-over-year metric. No future year target. Demonstrates that "持續" + current-year result → `already`, contrasting with within_2_years examples that also show current-year results but then pivot to 2025.

**Example B — train index [350]** (used in V2, V6)
- Text: 萬海自2024年起要求供應商每年填寫自評表，2024年共400家完成
- Selection rationale: Supply chain activity — the same category that causes `between_2_and_5_years` false positives for `already`. Shows that "供應鏈管理 + 每年 + 2024年已完成" is `already`, not `between_2_and_5_years`. This directly targets the supply-chain already/between confusion.

### between_2_and_5_years Examples

**Example A — train index [1]** (used in all variants)
- Text: 台泥供應鏈低碳轉型、人權環境生物多樣性合作夥伴關係
- Selection rationale: **Canonical no-year management promise** — no year, supply chain management focus, "建立合作夥伴關係". This is exactly the type of text that should default to `between_2_and_5_years` (not `already` since no current-period result, not `more_than_5_years` since no explicit long-term goal). Short and clear.

**Example B — train index [947]** (used in V2, V6)
- Text: IFRS永續揭露準則導入計畫，2026→2027→2028→2029四階段里程碑
- Selection rationale: **Multi-year roadmap with 2027-2029 explicit targets** — demonstrates the "farthest representative target" rule: even though 2026 appears, the plan's primary completion year is 2029 (正式公告申報). Teaches the model to select the farthest meaningful milestone.

### more_than_5_years Examples

**Example A — train index [13]** (used in all variants)
- Text: 台達供應鏈範疇三2030年降低25%（SBT目標）
- Selection rationale: **SBT 2030 target** — clear explicit 2030 year, SBT science-based target context. This is the most common more_than_5_years pattern in the training data. The "以2021年作為基準年" context teaches the model that base years are not completion years.

**Example B — train index [84]** (used in V2, V6)
- Text: 鴻海2050淨零，2030削減42%（近期），2050削減90%（長期）
- Selection rationale: **2050 net-zero commitment** — the other dominant more_than_5_years pattern. Has both 2030 and 2050 targets; demonstrates that the farthest representative commitment (2050 net-zero) determines the label. Also shows that 2030 as near-term doesn't override a 2050 primary pledge.

---

## Variant Design Decisions

### V1 (4-shot, direct-answer)
- Minimal overhead: 4 boundary examples covering the 2 primary failure modes.
- Added one explicit "anti-confusion" rule not in baseline: "文中同時提到2024年實績與2025年目標時，主要承諾年份是2025，應判within_2_years，而非already."
- One example per class: within_2_years=[345], already=[20], between_2_5=[1], more_than_5=[13].

### V2 (8-shot, direct-answer)
- Expanded to 2 examples per class for broader pattern coverage.
- within_2_years: [345] (2024 result + 2025 KPI, most confusable) + [976] (continuation 2025).
- already: [20] (periodic ongoing) + [350] (supply chain annual, targets supply-chain already/b25 confusion).
- between_2_5: [1] (no-year management) + [947] (2027-2029 explicit roadmap).
- more_than_5: [13] (2030 SBT) + [84] (2050 net-zero).

### V3 (4-shot CoT)
- Same 4 examples as V1 but with explicit reasoning step before label.
- CoT format: "思考：[year or activity reasoning]\n輸出：<label>"
- Hypothesis: forces explicit year identification step reduces within_2_years vs already boundary errors.
- Output format remains CoT (思考 + 輸出) for actual predictions — caller must parse to extract label.

### V4 (rules-only, no few-shot)
- Ablation: can rules alone (without examples) fix the failure modes?
- Added 4-step priority decision tree not in baseline:
  1. Find main target completion year (ignoring base/reference years)
  2. If explicit year: map to label with KEY RULE for 2025/2026 override
  3. If no year: activity-type classification
  4. Anti-confusion notes
- Key additions: explicit rule that 2025/2026 overrides持續/supply-chain/2024 context; explicit rule that 2024 performance + no future year → always already.
- Token-efficient: no examples added, purely rule enhancement.

### V5 (6-shot asymmetric, within_2_years ×3)
- Oversamples the rarest class (1.6% train, 13 examples total).
- within_2_years: [345] + [256] + [976] = 3 examples covering different 2025 contexts.
- already: [20] only (strongest signal).
- between_2_5: [1] only (clearest no-year management).
- more_than_5: [13] only (2030 SBT).
- Hypothesis: asymmetric oversampling most directly counteracts model's prior bias against within_2_years.

### V6 (8-shot CoT extended)
- Combination of V2 (8 examples) + V3 (CoT format).
- Full per-class CoT coverage with 2 examples each.
- Highest token cost, expected strongest signal.
- Output format: CoT (思考 + 輸出) — caller must parse to extract label.

---

## Data-Use Compliance

- All few-shot examples are from `data/raw_data/vpesg_4k_train_1000.json` (training split). No val or test data used.
- Only the `data` field text appears in the prompt as example input. No annotation fields (evidence_string, promise_string, etc.) are included.
- The ground-truth `verification_timeline` label for each example is shown as the expected output in the demonstration — this is standard few-shot prompting and is explicitly permitted by the data-use rules (training split labels shown as output target, not as input feature).
- At runtime, the model receives only the `data` field of each test row. No derived fields, no upstream labels, no annotation fields are injected into the ST4 prompt.
- V4 uses no examples; all other variants use training-split examples only.
- No post-processing rules are embedded in any prompt.

---

## Design Trade-offs

| Decision | Choice | Rationale |
|---|---|---|
| Example length | Full training text (not truncated) | Short examples risk losing the key boundary signals; e.g. [345] needs both the 2024 result AND the 2025 KPI sentence. |
| Anti-confusion rule placement | Appended to existing rules section | Adding the "2024 result + 2025 target → within_2_years" rule explicitly in the rules block reduces CoT dependency for V1/V2/V5. |
| CoT instruction scope | Restricted to "year or activity reasoning" | Prevents CoT from introducing non-data reasoning (e.g. business context assumptions). |
| V4 rule ordering | Explicit 4-step priority | Without ordering, model conflates "持續/每年" (→already) with "supply chain management" (→between_2_5); step ordering makes year lookup the highest priority. |
| within_2_years in V5 | 3 shots vs 2 | Only 13 training examples exist; using 3 of the most distinct ones (different contexts: KPI, platform launch, water continuation) provides maximum coverage within reasonable prompt length. |

---

## Blocked Items

None. All 6 files created successfully.

---

## Next Step

Exp phase: run all 6 variants on `data/benchmarks/val_public.json` to select top-2 by ST4 Macro-F1, then gate on `val_test.json`.
