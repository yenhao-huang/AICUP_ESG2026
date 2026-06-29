# External Claude Review — Loop 001 a0 (real-only) ST1 promotion

Command context: external reviewer invoked per CLAUDE.md Agent Looping Workflow
(mandatory `reflect/` Claude review). Timestamp: 2026-06-09.
Decision under review: promote `models/loop001_st1_a0_real_only/best_st1.pt` as
the new Stage 1 (Module A) baseline checkpoint, replacing
`models/exp4_optimize2_highconf_yes_balanced_no_large`.

## Numbers re-verified from raw JSON (not taken from the markdown)

| metric | source JSON (`loops001/exp/...`) | value |
|---|---|---|
| BASE_GATE (exp4, val_test) | (reflect, prior) | 0.802383 |
| exp4 val_public | `valpublic_eval/baseline_exp4_valpublic.json` → `st1.macro_f1` | 0.7872834 |
| a0 val_public (selection) | `valpublic_eval/a0_real_only_valpublic.json` → `st1.macro_f1` | 0.8030692 |
| a0 val_test (blind gate) | `gate_eval/a0_real_only_valtest.json` → `st1.macro_f1` | 0.8109766 |

Eval `config` in each JSON: `stages=['st1']`, `cascade=False`,
`checkpoint_dir=models/loop001_st1_a0_real_only`,
`model=hfl/chinese-roberta-wwm-ext-large`, input split = `data` rows. Candidate
checkpoint exists (1.30 GB). Train file `vpesg_4k_train_1000.json` = 1000 rows,
Yes 814 / No 186 (real-only, confirmed).

## 1. Data-use / Problem-Definition compliance — PASS

- a0 runs **no synthesis**. It is `roberta-wwm-ext-large` fine-tuned on the
  canonical real-only train file only. `promise_string` is therefore irrelevant
  to a0 — it never enters training, runtime, prompt, rule, or feature. Confirmed.
- Runtime is a single `data`-only BERT forward pass (`cascade=False`, ST1 path
  reads `d["data"]`). No `evidence_string`/extracted spans/labels/derived fields,
  no rule-based postprocess. GT used only for offline scoring.
- a0 also clears the standing methods.md caveat (L206–207: "the training/export
  path still needs a separate audit before final promotion") that hangs over the
  exp4 checkpoint: a0's training source is fully auditable real canonical data.

## 2. Score-first gate — PASS (strict)

- Blind gate: a0 val_test **0.8109766 > BASE_GATE 0.802383, Δ +0.008594**,
  strictly greater. Hard pass/fail item: **pass**.
- Selection discipline: a0 was selected on val_public (0.8030692 > exp4
  0.7872834, **+0.015786**); both clean splits are 0/500 train-overlap and
  mutually disjoint; the blind gate was touched once. No hyperparameter was tuned
  on the gate.
- Important: a0 is the *lower*-scoring of the two val_public-tied arms on the
  blind gate (a0 0.8110 vs a3_b1 0.8234). Promoting a0 therefore does **not**
  exploit the gate — it is the conservative, synthesis-free choice that both the
  loop001 and loop002 reflect records independently recommend as the honest
  data-only win ("the improvement over the old baseline is attributable to the
  canonical-data recipe, not synthesis"). The recipe-not-synthesis reading makes
  a0 the correctly-attributed cause of the gain; a0 is the right artifact to
  promote, and is strictly preferable to a3_b1 here.

## 3. Downstream / weighted-score impact — ST1-only, but re-score integrated

- Stage 1 is the cascade head. ST2–4 **checkpoints are byte-identical**;
  promoting a0 swaps only the ST1 Module A artifact. For the metric under review
  (standalone no-cascade ST1 Macro-F1) there is no downstream concern.
- Caveat the gate must record honestly: per `docs/methods.md` integrated-scoring
  section (eval_bert.py prediction-error penalty), changing ST1 *predictions*
  mechanically shifts the **integrated** ST2/ST3/ST4 Macro-F1 even with identical
  downstream models, because downstream preds are force-flipped when
  `st1_pred != st1_label`. So "ST2–4 Δ = 0.000" is true for *artifacts* but is
  **not** guaranteed for *integrated* scores and must not be gated as
  byte-identical. Direction is favorable: a0 is more ST1-accurate than exp4 →
  fewer `st1_pred != st1_label` flips → integrated ST3/ST4 should hold or improve,
  never regress in expectation. Condition: re-run the integrated 4-stage scorer
  once with the a0 head and record the weighted-score delta rather than asserting
  0; promote only if weighted score is ≥ baseline (expected).

## 4. docs/methods.md update — YES, required on promotion

Update in the same work session:

- **Stage 1 → Module A — BERT Classifier**, "Default checkpoint" block: replace
  `exp4_optimize2_highconf_yes_balanced_no_large/best_st1.pt` with
  `models/loop001_st1_a0_real_only/best_st1.pt`; recipe note:
  "`hfl/chinese-roberta-wwm-ext-large`, fine-tuned on the canonical real-only
  train `data/raw_data/vpesg_4k_train_1000.json` (1000 rows, Yes 814 / No 186),
  no synthetic data; data-only runtime input `data`; ST1 Macro-F1 val_test
  0.8110, val_public 0.8031 (exp4 baseline 0.8024 / 0.7873)." Drop the stale
  "training/export path still needs a separate audit" sentence (now satisfied).
- **Per-Stage Selectable Options** table, Stage 1 row: change the BERT checkpoint
  default cell to `loop001_st1_a0_real_only/best_st1.pt`.
- **Data-use note**: record that `data/benchmarks/test.json` is retired as a
  selection split (200/200 train-leaked); `val_public.json` is the canonical
  selection split and `val_test.json` the locked blind gate (BASE_GATE 0.802383).
- Module B (BERT+Codex-RAG hybrid) inherits a0 as its base per loop002 but is
  **not** being promoted here; leave Module B default unchanged.

`docs/loops/agent_loop_state.json` does not exist; create it on promotion with
BASE_GATE = 0.802383 and the new ST1 baseline pointer
`models/loop001_st1_a0_real_only`.

## 5. Risks / caveats / conditions

- **Modest margin.** The a0-over-exp4 gate win is +0.0086 (~a few rows of 500).
  It is real, on a clean disjoint split, and improves No-F1 (0.6919 vs exp4's
  weaker No-F1), but it is small — treat as a safe incremental baseline refresh,
  not a large gain.
- **Mandatory review now satisfied.** loop001/loop002 both verdicted *defer*
  solely because the external Claude review was absent (the one non-waivable gate
  item). This review, stored under `reflect/`, clears that blocker for the **a0**
  artifact specifically. Do not retroactively launder loop002's LLM-RAG pipeline
  under this clearance — that promotion needs its own gate (its blind lift over a0
  is ~1 row).
- **Conditions on promote:** (a) store this review under `reflect/`; (b) re-run
  the integrated scorer once with the a0 head and confirm weighted score ≥
  baseline and no integrated stage regression beyond a stated tolerance; (c)
  update `docs/methods.md` Stage 1 + the options table + data-use note and create
  `agent_loop_state.json` in the same session; (d) keep `test.json` retired.
- **Reproducibility note (low risk):** a0 is a real-only retrain; if the promoted
  checkpoint is ever rebuilt, fix the seed/recipe so the 0.8110 gate read
  reproduces.

REVIEW VERDICT: support-promote | oppose-promote | conditional
→ **conditional** (support-promote on the score/data-only merits; conditioned on
items 5(a)–(d): store this review, re-score integrated weighted, update
methods.md + agent_loop_state, keep test.json retired).
