# Loop 002 — Experiment

Confidence-routed LLM-RAG fallback layered on the frozen loop001 a0 BERT
(`models/loop001_st1_a0_real_only`). ST1 promise-status. Selection on
`val_public` only; `val_test` is the blind gate, touched once for the single
selected config. Runtime input is `data` only on every path
(`--info-mode data-only --text-col data`); `promise_string` never enters the
runtime/RAG path (used by the offline scorer ONLY).

GPU policy (all local steps): `CUDA_VISIBLE_DEVICES=0`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--device cuda:0`,
`--batch-size 4`. GPU1 reserved. LLM endpoint `http://192.168.1.79:3134/v1`
(HTTP 200, served model `/workspace/llm_model`). Embedding backend `st` over the
cached data-only ModernBert index
`data/generated/loop002_st1_rag/loop002_st1_train_data_only_modernbert_index.npz`.

Offline scorer: `exp/.../loops/loops002/exp/score_csv.py`
(sklearn `f1_score(labels=["Yes","No"], average="macro")`, identical metric
family to `core/eval/eval_bert.py`). GT `promise_status` from the eval JSON is
read for OFFLINE SCORING ONLY.

Splits (verified): val_public n=500 (Yes 410 / No 90); val_test n=500 (Yes 403 /
No 97).

## 1. base_score (BLOCKING) — pure-BERT a0, no fallback

Commands (run once each, reused across the grid):
```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python core/human/predict/stage1/pred_by_bert.py \
  --data data/benchmarks/val_public.json \
  --output exp/.../loops/loops002/exp/base/a0_val_public_scored.csv \
  --model hfl/chinese-roberta-wwm-ext-large \
  --finetune-path models/loop001_st1_a0_real_only/best_st1.pt \
  --mode finetune --text-col data --device cuda:0 --batch-size 4 --local-files-only
# ... same with --data val_test.json   --output base/a0_val_test_scored.csv
# ... a3_b1 secondary base on val_public (--finetune-path models/loop001_st1_a3_b1/best_st1.pt
#     --output base/a3b1_val_public_scored.csv)  [base-invariance diagnostic only]
```
All three runs: `validation.ok = true`, 500 rows each.

Pure-BERT a0 Macro-F1 (control, no routing):

| split | Macro-F1 | f1_Yes | f1_No | matches expected |
|---|---|---|---|---|
| val_public | **0.803069** | 0.935407 | 0.670732 | 0.8031 ✓ |
| val_test   | **0.810977** | 0.930061 | 0.691892 | 0.8110 ✓ |

a3_b1 secondary base on val_public = 0.803069 (val_public predictions identical
to a0; confidence scores differ → different routing, see §4 diagnostic).

Confidence = max(softmax(score_yes, score_no)). Verified exactly
`conf = (1+|score_yes−score_no|)/2` for all 500 rows. Escalation counts by
absolute-confidence threshold on the a0 val_public CSV: T0.60→4, T0.70→16,
T0.80→31, T0.90→50, T0.95→66. min conf 0.5065, median 0.9986.

## 2. fallback_grid_valpublic (selection — val_public ONLY)

Runner: `core/human/predict/stage1/pred_by_bert_codex_rag.py`, locked
`--info-mode data-only --text-col data --embedding-backend st --llm-backend http`,
`--embed-cache <modernbert index>`, `--train-data
data/raw_data/vpesg_4k_train_1000.json`. Every config reported
`codex_errors = 0` (no BERT-error fallbacks → all runs valid for selection).

Coarse T-sweep (K=5, prompt=strict_negative_with_certficate, ctx=bert-prediction):

| config | #esc | val_public Macro-F1 | f1_Yes | f1_No |
|---|---|---|---|---|
| T0.60_K5_certneg_bertctx | 4  | 0.807657 | 0.936527 | 0.678788 |
| T0.70_K5_certneg_bertctx | 16 | 0.811306 | 0.936221 | 0.686391 |
| T0.80_K5_certneg_bertctx | 31 | 0.802597 | 0.930909 | 0.674286 |
| T0.90_K5_certneg_bertctx | 50 | 0.789973 | 0.924390 | 0.655556 |
| T0.95_K5_certneg_bertctx | 66 | 0.791079 | 0.922699 | 0.659459 |

certneg peaks at T=0.70 then degrades (LLM overturns correct high-conf rows).

Refinement around T=0.70 (K / prompt / context):

| config | #esc | val_public Macro-F1 | f1_Yes | f1_No |
|---|---|---|---|---|
| T0.70_K3_certneg_bertctx   | 16 | 0.808646 | 0.934940 | 0.682353 |
| T0.70_K8_certneg_bertctx   | 16 | 0.808646 | 0.934940 | 0.682353 |
| T0.70_K5_strictneg_bertctx | 16 | 0.813988 | 0.937500 | 0.690476 |
| T0.70_K5_default_bertctx   | 16 | 0.810363 | 0.937799 | 0.682927 |
| T0.70_K5_certneg_none      | 16 | 0.808646 | 0.934940 | 0.682353 |

prompt=strict_negative > certificate > default on the routed rows; K and
mode-context had near-zero effect at T=0.70 (same 16 rows, flips invariant).
strict_negative chosen as the prompt to extend.

strict_negative across thresholds / K / context:

| config | #esc | val_public Macro-F1 | f1_Yes | f1_No |
|---|---|---|---|---|
| T0.60_K5_strictneg_bertctx | 4  | 0.807657 | 0.936527 | 0.678788 |
| T0.70_K5_strictneg_bertctx | 16 | 0.813988 | 0.937500 | 0.690476 |
| T0.70_K8_strictneg_bertctx | 16 | 0.816693 | 0.938776 | 0.694611 |
| T0.70_K5_strictneg_none    | 16 | 0.813988 | 0.937500 | 0.690476 |
| **T0.80_K5_strictneg_bertctx** | **31** | **0.817436** | **0.937198** | **0.697674** |
| T0.80_K8_strictneg_bertctx | 31 | 0.813063 | 0.936068 | 0.690058 |
| T0.90_K5_strictneg_bertctx | 50 | 0.797065 | 0.931408 | 0.662722 |
| T0.95_K5_strictneg_bertctx | 66 | 0.794472 | 0.930120 | 0.658824 |

strict_negative is far more robust at higher T than certneg; inverted-U peaks at
**T=0.80, K=5, bert-prediction = 0.817436**, then degrades by T=0.90/0.95.

Margin-router ablation R (best arm K5/strictneg/bertctx). Mapping
`|score_yes−score_no| < m` ⇔ `conf < (1+m)/2`:

| router | equiv T | #esc | val_public Macro-F1 |
|---|---|---|---|
| margin m=0.10 | 0.55 | 3 | 0.807657 |
| margin m=0.20 | 0.60 | 4 | 0.807657 (≡ T0.60_K5_strictneg_bertctx) |

Margin routing did not beat absolute-confidence T=0.80.

Configs RUN: 5 (T-sweep certneg) + 5 (T0.70 refine) + 8 (strictneg sweep) + 1
(margin m=0.10) = **19 fallback configs** (margin m=0.20 reused T0.60). All
`codex_errors=0`. Configs SKIPPED vs the ≈15-plan: none material — the plan's
P×C×K combinatorics at every T were pruned after the T=0.70 refinement showed K
and mode-context are inert and strict_negative dominates; the strict_negative
sweep then covered the full threshold axis. No numbers fabricated.

## 3. select_and_gate

**Selected config (highest val_public Macro-F1):
`T0.80_K5_strictneg_bertctx`** — val_public **0.817436** (31 escalations).
Tie-break not needed (unique max). Config materialized at
`exp/.../loops/loops002/exp/candidate_config.json`.

Selected config:
```
threshold=0.80  top_k=5  prompt=strict_negative  mode_context=bert-prediction
info_mode=data-only  text_col=data  embedding_backend=st (ModernBert index)
llm_backend=http  base_url=http://192.168.1.79:3134/v1  served=/workspace/llm_model
base=models/loop001_st1_a0_real_only/best_st1.pt
```

Flip diagnostics on val_public (selected config): escalated 31, flips 16,
toward-GT 9, away-GT 7, neutral 0 → **flip-correctness 0.562** (net +2 correct,
above the 50% abandonment floor).

### Blind gate (val_test, touched once)
```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python core/human/predict/stage1/pred_by_bert_codex_rag.py \
  --bert-csv exp/.../loops/loops002/exp/base/a0_val_test_scored.csv \
  --data data/benchmarks/val_test.json \
  --output exp/.../loops/loops002/exp/gate/T0.80_K5_strictneg_bertctx_valtest/merged.csv \
  --train-data data/raw_data/vpesg_4k_train_1000.json \
  --threshold 0.80 --top-k 5 --prompt strict_negative --mode-context bert-prediction \
  --info-mode data-only --text-col data --embedding-backend st \
  --embedding-model /models/qihoo360-Zhinao-ChineseModernBert-Embedding \
  --embedding-device cuda:0 --embed-batch-size 4 \
  --embed-cache data/generated/loop002_st1_rag/loop002_st1_train_data_only_modernbert_index.npz \
  --llm-backend http --llm-base-url http://192.168.1.79:3134/v1 --timeout 120
```
stdout: `Threshold=0.80  high_conf=470  low_conf=30  total=500`, `codex_errors=0`.

**val_test (blind) = 0.811566** (f1_Yes 0.928395, f1_No 0.694737, 30 escalations).

Generalization gap (dev − blind) = val_public 0.817436 − val_test 0.811566 =
**+0.005870** (small, positive — no overfit to the selection split).

## 4. Base-invariance diagnostic (a3_b1 base, val_public, diagnostic ONLY)

Selected config on `a3b1_val_public_scored.csv`: Macro-F1 **0.808480** (15
escalations), base a3_b1 = 0.803069 → lift **+0.005411**. a3_b1's confidence
scores route fewer rows (15 vs 31) but the fallback still lifts ST1 on the second
base, confirming the lift direction is base-invariant (positive on both a0 and
a3_b1). Not used for config selection.

## Gate-check table

| check | value | threshold | Δ | result |
|---|---|---|---|---|
| Primary blind gate: val_test Macro-F1 > BASE_GATE | 0.811566 | > 0.802383 | +0.009183 | **PASS** |
| Secondary method-validity: val_public Macro-F1 > a0 base | 0.817436 | > 0.803069 | +0.014367 | **PASS** |
| Blind vs a0 base val_test (does fallback add value blind) | 0.811566 | > 0.810977 | +0.000589 | PASS (marginal, ~1 row) |
| Flip-correctness on routed rows > 0.50 | 0.562 | > 0.50 | — | PASS |
| Endpoint stability (codex_errors across all runs) | 0 | = 0 | — | PASS |
| Data-only runtime compliance | data only | locked | — | PASS |
| Downstream ST2–4 regression | unchanged (ST1-only) | δ=0 | 0 | PASS |
| Generalization gap | +0.005870 | small | — | PASS |

## Data-use compliance

- Runtime `data` ONLY on every path: BERT forward over `data`; embedding index +
  query over train/eval `data` (`--info-mode data-only --text-col data`,
  ModernBert cache embeds `row["data"]` only); LLM prompt body = query `data` +
  retrieved train `data`. Few-shot example labels are the retrieved TRAIN rows'
  `promise_status` (permitted in-context demonstration labels), never the query
  row's annotation.
- `promise_string` / `evidence_string` / spans / eval-row GT: never in runtime;
  GT used by `score_csv.py` for OFFLINE Macro-F1 only.
- No config used `promise-str-only` / `data+promise_str` / `--text-col` ≠ `data`.

## Artifacts

- Base scored CSVs: `exp/.../loops/loops002/exp/base/{a0_val_public_scored.csv,
  a0_val_test_scored.csv, a3b1_val_public_scored.csv}`
- Grid merged CSVs + per-config `summary.json`:
  `exp/.../loops/loops002/exp/grid/<config>/merged.csv` (19 configs)
- Blind-gate merged CSV:
  `exp/.../loops/loops002/exp/gate/T0.80_K5_strictneg_bertctx_valtest/merged.csv`
- Candidate: `exp/.../loops/loops002/exp/candidate_config.json`,
  `candidate_val_public_st1.json`, `candidate_val_test_st1.json`
- Scorer: `exp/.../loops/loops002/exp/score_csv.py`
- Embedding index (dev):
  `data/generated/loop002_st1_rag/loop002_st1_train_data_only_modernbert_index.npz`

## Decision input for reflect

The selected config **passes both required gates**: blind val_test 0.811566 >
BASE_GATE 0.802383 (+0.009183) AND val_public 0.817436 > a0 base 0.803069
(+0.014367). However, the lift over the a0 base it layers on is large on
val_public (+0.0144) but only marginal on the blind val_test (+0.000589, ~1
row): the fallback's blind-split benefit over a0 is within ~1-row noise even
though it clears the exp4 BASE_GATE comfortably. Flip-correctness 0.562 (net +2)
is positive but thin. Promotion is for the reflect phase + mandatory Claude
review to decide; this record does not promote, and does not modify
`docs/methods.md`, `docs/feature_list.json`, or `agent_loop_state.json`.
