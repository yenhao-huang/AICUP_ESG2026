# Stage 3 Qwen Same-Page Context Probe

This directory contains the 2026-06-16 Stage 3 online Qwen context experiment.

## Headline Comparison: Qwen vs Codex vs BERT

Same 600-row `data/val_ctxhit.json`, same GT `evidence_status` gate, scored
against GT `evidence_quality`. Predictor artifacts:

```text
Codex   preds/st3_codex_parallel_codex_parallel_20260616_115412.csv  (gpt-5.5, clear_notclear prompt)
BERT    preds/st3_bert_a1_weighted_ce.csv                            (A1 weighted CE checkpoint)
Qwen    preds/st3_qwen_ctx.csv                                       (online Qwen + same-page context)
```

| Metric            | Codex      | BERT A1    | Qwen ctx   |
| ----------------- | ---------- | ---------- | ---------- |
| 4-class Macro-F1  | **0.5955** | 0.5900     | 0.5435     |
| 3-class Macro-F1  | **0.7941** | 0.7871     | 0.7251     |
| 2-class Macro-F1  | **0.6923** | 0.6806     | 0.5876     |
| 4-class accuracy  | 0.8783     | 0.8783     | 0.7867     |

Per-class F1 (4-class):

| Class      | Codex      | BERT A1    | Qwen ctx   |
| ---------- | ---------- | ---------- | ---------- |
| Clear      | 0.8884     | 0.8892     | 0.7816     |
| Not Clear  | **0.4956** | 0.4706     | 0.3923     |
| Misleading | 0.0000     | 0.0000     | 0.0000     |
| N/A        | 0.9975     | 1.0000     | 1.0000     |

Takeaways:

- Codex is the best overall (highest Macro-F1 at every class granularity), but
  it only edges BERT A1 by +0.0055 4-class Macro-F1 while accuracy ties exactly
  at 0.8783. The gap is entirely `Not Clear` F1 (0.4956 vs 0.4706).
- BERT A1 is the strongest cheap option: it matches Codex accuracy and trails
  Macro-F1 by a hair, with the best `Clear` and `N/A` F1.
- Qwen ctx is clearly behind on every metric; it over-predicts `Not Clear` and
  flips too many GT `Clear` rows.
- The shared ceiling is `Misleading` (F1 0.0 for all three; only 1 GT row, id
  `11836`) and the `Clear`/`Not Clear` boundary.

## Files

- `run_pred.sh`: batch runner for `core/human/predict/stage3/pred_by_qwen.py`.
- `run_pred_reason.sh`: Qwen reason/thinking batch runner with separate output
  names; does not overwrite `run_pred.sh` artifacts.
- `run_pred_codex_parallel.sh`: parallel shard runner for
  `core/human/predict/stage3/pred_by_codex.py`.
- `app.py`: Gradio inspector for selecting a row, rebuilding the exact system
  prompt, user prompt, and same-page OCR context, then calling Qwen online.
- `data/val_ctxhit.json`: 600-row validation subset used in this probe.
- `preds/st3_qwen_ctx.csv`: completed Qwen prediction with same-page context and
  the original GT `evidence_status` gate.
- `logs/pred_ctx.log`: batch log for the completed context run.

## Completed Context Run

Command family:

```bash
DATA=exp/integrated_stage_predictions/0616/test_add_context/stage3/data/val_ctxhit.json \
GATE=exp/integrated_stage_predictions/0616/test_add_context/stage3/data/val_ctxhit.json \
bash exp/integrated_stage_predictions/0616/test_add_context/stage3/run_pred.sh
```

Completed artifact:

```text
exp/integrated_stage_predictions/0616/test_add_context/stage3/preds/st3_qwen_ctx.csv
```

Summary:

```text
rows: 600
Qwen-predicted rows: 398
stage2_filter N/A rows: 202
context hits: 398
```

Scores against `data/val_ctxhit.json`:

```text
3-class full-coverage Macro-F1, dropping the single Misleading GT row: 0.7251
4-class Macro-F1, counting the single Misleading GT row as wrong: 0.5435
4-class accuracy: 0.7867
```

4-class per-class F1:

```text
Clear:      0.7816
Not Clear:  0.3923
Misleading: 0.0000
N/A:        1.0000
```

The single GT `Misleading` row is id `11836`; this run predicted it as `Clear`.

## Comparison Against A1 Weighted CE BERT

Baseline checkpoint:

```text
exp/agent_loop/claude/20260610T012103/models/loops001/A1_weighted_ce/best_st3.pt
```

Prediction command:

```bash
/workspace/esg_contest/.venv/bin/python core/human/predict/stage3/pred_by_bert.py \
  --data exp/integrated_stage_predictions/0616/test_add_context/stage3/data/val_ctxhit.json \
  --stage2-csv exp/integrated_stage_predictions/0616/test_add_context/stage3/data/val_ctxhit.json \
  --stage2-gate-col evidence_status \
  --finetune-path exp/agent_loop/claude/20260610T012103/models/loops001/A1_weighted_ce/best_st3.pt \
  --output exp/integrated_stage_predictions/0616/test_add_context/stage3/preds/st3_bert_a1_weighted_ce.csv \
  --text-mode data \
  --batch-size 16 \
  --device cuda:1 \
  --local-files-only
```

Completed artifact:

```text
exp/integrated_stage_predictions/0616/test_add_context/stage3/preds/st3_bert_a1_weighted_ce.csv
```

Same 600 rows, same GT `evidence_status` gate:

```text
                 Qwen ctx    BERT A1 weighted_ce
4-class Macro-F1  0.5435     0.5900
3-class Macro-F1  0.7251     0.7871
2-class Macro-F1  0.5876     0.6806
4-class accuracy  0.7867     0.8783
```

Per-class F1:

```text
                 Qwen ctx    BERT A1
Clear             0.7816     0.8892
Not Clear         0.3923     0.4706
Misleading        0.0000     0.0000
N/A               1.0000     1.0000
```

2-class confusion matrices:

```text
Qwen ctx
rows = GT, cols = pred

             Clear  Not Clear
Clear          229        107
Not Clear       20         41

BERT A1

             Clear  Not Clear
Clear          293         43
Not Clear       29         32
```

Disagreement summary:

```text
different rows: 117
Qwen=Not Clear, BERT=Clear: 95
Qwen=Clear, BERT=Not Clear: 22

Qwen correct / BERT wrong: 31
BERT correct / Qwen wrong: 86
both wrong with different labels: 0
```

Interpretation: on this 600-row subset, A1 weighted CE is stronger than the
Qwen+same-page-context run under the same gate. Qwen is more aggressive on
`Not Clear`; it recovers some NC rows but flips too many GT `Clear` rows to
`Not Clear`.

## Parallel Codex Runner

Script:

```text
exp/integrated_stage_predictions/0616/test_add_context/stage3/run_pred_codex_parallel.sh
```

Purpose: split the 600-row `data/val_ctxhit.json` into `WORKERS` shards, run
`core/human/predict/stage3/pred_by_codex.py` concurrently, and merge shard CSVs
back into the original input order.

Default run:

```bash
bash exp/integrated_stage_predictions/0616/test_add_context/stage3/run_pred_codex_parallel.sh
```

Common overrides:

```bash
WORKERS=6 \
MODEL=gpt-5.5 \
TIMEOUT=300 \
RUN_ID=codex_gpt55_val_ctxhit_w6 \
bash exp/integrated_stage_predictions/0616/test_add_context/stage3/run_pred_codex_parallel.sh
```

Smoke run:

```bash
LIMIT=12 WORKERS=3 RUN_ID=smoke_codex_parallel \
bash exp/integrated_stage_predictions/0616/test_add_context/stage3/run_pred_codex_parallel.sh
```

Print the exact prompt for the first Stage2-passed row in each shard:

```bash
LIMIT=1 WORKERS=1 RUN_ID=smoke_one PRINT_PROMPT=1 \
bash exp/integrated_stage_predictions/0616/test_add_context/stage3/run_pred_codex_parallel.sh
```

Run Codex with the same same-page OCR context format used by `pred_by_qwen.py`:

```bash
ADD_CONTEXT=1 \
PROMPT_PATH=configs/prompt/stage3/codex/clear_notclear_with_context.txt \
WORKERS=4 \
RUN_ID=codex_ctx_w4 \
bash exp/integrated_stage_predictions/0616/test_add_context/stage3/run_pred_codex_parallel.sh
```

Context knobs:

```text
ADD_CONTEXT=1
CONTEXT_BUDGET=800
CONTEXT_WINDOW=after_biased   # or symmetric
CROSS_PAGE=1
PREFIX_CHARS=18
```

With `ADD_CONTEXT=1`, each shard JSON rewrites the row `data` field to:

```text
承諾句：<original data>

同頁內容：
<same-page OCR window>
```

The temporary shard manifest records `context_hit_counts`.

Prompt dumps are written next to shard logs:

```text
exp/integrated_stage_predictions/0616/test_add_context/stage3/logs/codex_parallel/<RUN_ID>/shard_00.log.prompt.txt
```

Inputs and outputs:

```text
DATA default: exp/integrated_stage_predictions/0616/test_add_context/stage3/data/val_ctxhit.json
GATE default: exp/integrated_stage_predictions/0616/test_add_context/stage3/data/val_ctxhit.json
Prompt default: configs/prompt/stage3/codex/clear_notclear_only.txt

Merged CSV:
exp/integrated_stage_predictions/0616/test_add_context/stage3/preds/st3_codex_parallel_<RUN_ID>.csv

Shard logs:
exp/integrated_stage_predictions/0616/test_add_context/stage3/logs/codex_parallel/<RUN_ID>/

Raw Codex dumps:
exp/integrated_stage_predictions/0616/test_add_context/stage3/raw/codex_parallel/<RUN_ID>/
```

`RESUME=1` is enabled by default, so rerunning the same `RUN_ID` reuses existing
successful per-id raw dumps inside each shard and only fills missing/failed rows.

## Qwen Reason/Thinking Runner

Script:

```text
exp/integrated_stage_predictions/0616/test_add_context/stage3/run_pred_reason.sh
```

Purpose: run the same Stage 3 Qwen predictor with `--enable-thinking` and larger
`max_tokens`, writing reason-specific outputs so the normal `run_pred.sh` files
are not overwritten.

Default endpoint:

```text
http://127.0.0.1:8000/v1/chat/completions
```

Default run:

```bash
bash exp/integrated_stage_predictions/0616/test_add_context/stage3/run_pred_reason.sh
```

Smoke run:

```bash
LIMIT=20 RUN_ID=reason_smoke \
bash exp/integrated_stage_predictions/0616/test_add_context/stage3/run_pred_reason.sh
```

Common overrides:

```bash
CONC=2 \
MAX_TOKENS=768 \
MODEL=local-qwen \
RUN_ID=reason_mt768 \
bash exp/integrated_stage_predictions/0616/test_add_context/stage3/run_pred_reason.sh
```

Outputs:

```text
exp/integrated_stage_predictions/0616/test_add_context/stage3/preds/st3_qwen_<RUN_ID>_ctx.csv
exp/integrated_stage_predictions/0616/test_add_context/stage3/preds/st3_qwen_<RUN_ID>_ctrl.csv
exp/integrated_stage_predictions/0616/test_add_context/stage3/logs/pred_<RUN_ID>_ctx.log
exp/integrated_stage_predictions/0616/test_add_context/stage3/logs/pred_<RUN_ID>_ctrl.log
```

By default `RUN_ID=reason`, so the first full run writes:

```text
preds/st3_qwen_reason_ctx.csv
preds/st3_qwen_reason_ctrl.csv
logs/pred_reason_ctx.log
logs/pred_reason_ctrl.log
```

It does not touch the normal Qwen outputs:

```text
preds/st3_qwen_ctx.csv
preds/st3_qwen_ctrl.csv
logs/pred_ctx.log
logs/pred_ctrl.log
```

## Replaying With `0611/test_2` Stage 2 Gate

Gate artifact:

```text
data/predictions/0611/test_2/stage2/bert.csv
```

On the same 600 rows, the Stage 2 gate quality against GT `evidence_status` is:

```text
3-class accuracy: 0.7983
3-class Macro-F1: 0.7173

GT distribution:   Yes 398, No 93, N/A 109
Pred distribution: Yes 446, No 66, N/A 88
```

Yes-gate binary metrics:

```text
accuracy:  0.8033
precision: 0.8139
recall:    0.9121
F1:        0.8602
TP 363, FP 83, FN 35, TN 119
```

For Stage 3 scoring under this gate, align rows as follows:

- `test_2 evidence_status == Yes`: use the available Qwen Stage 3 prediction.
- `test_2 evidence_status in {No, N/A}`: set Stage 3 to `N/A`.

Important caveat: `preds/st3_qwen_ctx.csv` was generated with the GT Stage 2
gate, not the `test_2` gate. There are 83 rows where `test_2` predicts
`evidence_status == Yes` but the GT-gated Qwen CSV has `N/A`; those rows were not
actually sent to Qwen in this run.

If all 600 rows are scored and those 83 currently-available `N/A` values are
used as-is:

```text
4-class Macro-F1: 0.4984
accuracy: 0.7467

Clear F1:      0.7592
Not Clear F1:  0.3141
Misleading F1: 0.0000
N/A F1:        0.9203
```

Confusion matrix:

```text
rows = GT, cols = Pred

             Clear  Not Clear  Misleading  N/A
Clear          216        100           0   20
Not Clear       16         30           0   15
Misleading       1          0           0    0
N/A              0          0           0  202
```

If the 83 rows not actually sent to Qwen are excluded:

```text
included rows: 517
skipped rows: 83

4-class Macro-F1: 0.4863
accuracy: 0.7060

Clear F1:      0.7592
Not Clear F1:  0.3141
Misleading F1: 0.0000
N/A F1:        0.8718
```

Do not treat either replay score as the final `test_2`-gate Qwen result. The
clean result requires rerunning Qwen with:

```bash
DATA=exp/integrated_stage_predictions/0616/test_add_context/stage3/data/val_ctxhit.json \
GATE=data/predictions/0611/test_2/stage2/bert.csv \
bash exp/integrated_stage_predictions/0616/test_add_context/stage3/run_pred.sh
```

## Online UI

Run:

```bash
/workspace/esg_contest/.venv/bin/python \
  exp/integrated_stage_predictions/0616/test_add_context/stage3/app.py \
  --port 7863
```

Default online endpoint in the UI:

```text
http://192.168.1.76:3135/v1/chat/completions
```

## 2026-06-17 Codex Tagged-Prompt Ablation (100-row)

Prompts under `prompts/codex/` rendered through `stage3/core/vlm_pred.py`
(template order `system-prompt -> same-page-context -> ... -> data-prompt`).
Each `experiments/run_<name>_100.sh` runs the 100-row `data/val_ctxhit.100.jsonl`
with no Stage 2 gate (full coverage, model emits only Clear / Not Clear), then
scores with `core/analysis/score_st3_full_coverage.py` against
`data/val_ctxhit.100.json`. Codex backend is `gpt-5.5`.

Inputs per run:

```text
vanilla       <data-prompt> only                       (--no-add-context)
add_context   <data-prompt> + <same-page-context>       (--add-context --context-mode all; uses page table text_clean)
add_evidence  <data-prompt> + <evidence-string>         (--no-add-context --add-evidence-string)
```

BERT baseline (data-only checkpoint, no gate, full coverage):

```text
exp/agent_loop/claude/20260610T012103/models/loops001/A1_weighted_ce/best_st3.pt
```

Scores (100 rows; gtgated_2cls = GT-`evidence_status`-gated Clear/Not Clear
Macro-F1, n=67; the primary comparison metric):

| Predictor             | 2cls Macro-F1 | Clear F1 | Not Clear F1 | 4cls full-cov F1 | Clear/NotClear |
| --------------------- | ------------- | -------- | ------------ | ---------------- | -------------- |
| BERT A1               | **0.7295**    | 0.7717   | **0.3500**   | 0.3739           | 71 / 29        |
| vanilla               | 0.7281        | 0.7969   | 0.3077       | 0.3682           | 72 / 28        |
| add_context text_clean| 0.7281        | 0.8031   | 0.3000       | 0.3677           | 71 / 29        |
| add_evidence          | 0.6474        | 0.7907   | 0.2105       | 0.3337           | 73 / 27        |

Artifacts:

```text
preds/codex/vanilla_100_codex.csv
preds/codex/add_context_100_codex.csv
preds/codex/add_evidence_100_codex.csv
preds/codex/st3_bert_a1_100.csv
```

Takeaways:

- BERT A1 is best on this 100-row slice (2cls 0.7295, best Not Clear F1 0.35),
  but only edges codex vanilla/add_context by +0.0014 — a tie within noise at
  n=100; needs the 600-row set to separate.
- `add_context` injects same-page context from the page table's **`text_clean`**
  column (`stage3/core/build_prompt/add_context.py`, falls back to raw `text`),
  hitting all 100 rows (`context_hit=offset_hit_exact_window`). vs vanilla it
  changes only **1** prediction (id `11073`, a GT N/A row, wrong either way), so
  the 67 gated Clear/Not Clear rows are identical → 2cls Macro-F1 unchanged at
  0.7281; only the 4-class distribution shifts (72/28 → 71/29). With raw `text`
  the earlier run differed on 4 rows but also netted the same 2cls score. Net:
  same-page context (clean or raw) gives codex no lift on this slice.
- `add_evidence` (evidence-string as reference only, not counted as anchor) is
  clearly worse: 2cls 0.6474, Not Clear F1 drops to 0.2105. `add-evidence-2.txt`
  (counts evidence-string quantified content as a Clear anchor) is the opposite
  policy and not yet run here.
- All four share N/A F1 = 0 because no Stage 2 gate is applied; compare on
  gtgated_2cls, not 4cls full-coverage.

Overwrite note: the old `_common.sh`-based qwen ablation (now
`experiments/qwen/run_add_context.sh`) writes `preds/exp_add_context.csv` and
does NOT touch the codex file above. The codex 100-row prediction is (re)written
only by `experiments/codex/run_add_context_100.sh`
(`preds/codex/add_context_100_codex.csv`, overwritten on every run, no resume).

### Directory layout — qwen / codex split (2026-06-17)

Runner scripts are split by backend; the 100-row ablation family is duplicated
into a backend-pinned copy on each side (output filenames carry the backend, so
the two never collide):

```text
experiments/
  codex/   run_{vanilla,add_context,add_evidence,add_image,scoped,all}_100.sh   (BACKEND pinned codex)
  qwen/    run_{vanilla,add_context,add_evidence,add_image,scoped,all}_100.sh   (BACKEND pinned qwen)
           _common.sh + run_add_{context,image,evidence_promise,context_window,evidence_string}.sh + run_all.sh   (qwen ablation cluster)
  (root)   score_all.py, wait_and_score_100.sh                                  (shared utilities)
           run_full600.sh, run_method_screen.sh, run_method_600.sh,
           run_prompt_screen.sh, run_prompt_600.sh                              (older qwen screen/600 series, left in place)

preds/
  codex/   <name>_100_codex.csv   + st3_bert_a1_100.csv (BERT baseline kept beside the codex comparison)
  qwen/    <name>_100_qwen.csv    (written by experiments/qwen/run_*_100.sh)
  (root)   exp_*.csv / full_*.csv / m_*.csv / pp_*.csv / win_*.csv             (legacy flat preds; scanned by score_all.py — left in place)
```

The pinned `experiments/<backend>/run_<name>_100.sh` scripts write to
`preds/<backend>/`. The qwen `_common.sh` ablations still write `exp_*.csv` to
the flat `preds/` dir so `score_all.py` continues to find them. The older
screen/600 series and the legacy flat preds were intentionally left untouched to
avoid breaking `run_all.sh` orchestration and `score_all.py` globbing.

## 2026-06-17 Balanced val.100 Evaluation

New eval set `data/val.100.{jsonl,json}`: 100 rows pulled from `val_ctxhit.json`,
balanced 50 Clear / 50 Not Clear, all `evidence_status=Yes` (no N/A rows). This
fixes the old `val_ctxhit.100` artifacts — 33 N/A + only 11 Not Clear — that made
Not Clear F1 look terrible (N/A false-positive pollution + tiny support). Here
Not Clear support is 50 and the gate is implicit, so `gtgated_2cls_macro_f1`
(n=100) is the primary metric. `page_abstract` is joined in from
`../add_page_abstract/val_with_page_abstract.jsonl` (id → whole-page summary).

Codex backend `gpt-5.5`; BERT A1 is the data-only checkpoint
`exp/agent_loop/claude/20260610T012103/models/loops001/A1_weighted_ce/best_st3.pt`.
All 100 rows predicted, 0 missing / 0 except, every run aligned to the val.100 ids.

| Predictor          | 2cls Macro-F1 | Clear F1 | Not Clear F1 | 4cls full-cov | Clear/NotClear |
| ------------------ | ------------- | -------- | ------------ | ------------- | -------------- |
| add_page_abstract  | **0.7442**    | **0.7826** | **0.7059**  | 0.4962        | 65 / 35        |
| add_context        | 0.7220        | 0.7692   | 0.6747       | 0.4813        | 67 / 33        |
| add_page_abstract_context | 0.7144 | 0.7544   | 0.6744       | 0.4763        | 64 / 36        |
| add_context_hit_exact | 0.7106     | 0.7627   | 0.6585       | 0.4737        | 68 / 32        |
| add_image          | 0.7106        | 0.7627   | 0.6585       | 0.4737        | 68 / 32        |
| BERT A1            | 0.7033        | 0.7478   | 0.6588       | 0.4689        | 65 / 35        |
| vanilla            | 0.7014        | 0.7521   | 0.6506       | 0.4676        | 67 / 33        |
| add_evidence       | 0.6875        | 0.7500   | 0.6250       | 0.4583        | 70 / 30        |

Artifacts: `preds/codex/{vanilla,add_context,add_evidence,add_image,add_page_abstract}_100_codex.csv`,
`preds/codex/st3_bert_a1_val100.csv`.

Each `<name>` input set: vanilla `<data-prompt>` only; add_context `+<same-page-context>`
(text_clean); add_evidence `+<evidence-string>` (reference-only, not an anchor);
add_image `+<page-image>` (codex `--image`, now wired); add_page_abstract
`+<page-abstract>` (whole-page summary, understanding-only).

Takeaways:

- **`add_page_abstract` is the best run** (2cls 0.7442, top Clear and Not Clear
  F1), beating vanilla by +0.043 and BERT by +0.041. The whole-page summary helps
  the model read each promise in context without the OCR noise of raw same-page text.
- **Combining the two page signals hurts: `add_page_abstract_context` = 0.7144,
  below `add_page_abstract` alone (0.7442, -0.030) and even below `add_context`
  alone (0.7220, -0.008).** Feeding both the page summary and the raw same-page OCR
  together dilutes the clean signal the abstract gives on its own — abstract-only
  is the configuration to keep; do not stack same-page-context on top of it.
- **`add_context` (+0.021) > `add_image` (+0.009) > vanilla**; all three page-level
  context signals help, abstract > clean OCR text > image.
- **`add_context_hit_exact` (context-mode `hit_exact_window_norm_window`) = 0.7106,
  worse than `add_context` mode=all (0.7220) by -0.011.** On val.100 the strict
  mode still injected context for all 100 rows (`context_hit=offset_hit_exact_window`),
  so the drop is not from lower coverage — the same-page text it kept just helped
  less than the permissive `all` page text. (Result is identical to add_image here.)
- **`add_evidence` is the only run that hurts** (-0.014 vs vanilla): injecting the
  evidence fragment as reference-only still nudges the model toward Clear and costs
  Not Clear F1 (0.625, lowest).
- Every codex prompt variant except add_evidence beats the BERT A1 baseline on
  this balanced set; on the old `val_ctxhit.100` they were tied (~0.728 2cls).
- The 4cls full-cov column carries a phantom N/A class (F1 0) since val.100 has no
  N/A; compare on 2cls / per-class, not 4cls.

## 2026-06-17 Full non-N/A val_nonNA Evaluation (398 rows)

Larger eval set `data/val_nonNA.{jsonl,json}`: all 600 `val_ctxhit` rows with the
N/A `evidence_quality` rows filtered out → 398 rows (Clear 336 / Not Clear 61 /
Misleading 1), all `evidence_status=Yes`, page_abstract joined in. This is the
natural (imbalanced ~5.5:1) Clear:Not Clear distribution, unlike the balanced
val.100. Codex `gpt-5.5`, 0 missing / 0 except.

| Predictor          | 2cls Macro-F1 | Clear F1 | Not Clear F1 | 4cls full-cov | Clear/NotClear |
| ------------------ | ------------- | -------- | ------------ | ------------- | -------------- |
| add_page_abstract  | **0.6827**    | **0.8785** | **0.4868**  | 0.4551        | 307 / 91       |
| vanilla            | 0.6760        | 0.8711   | 0.4810       | 0.4507        | 301 / 97       |

Artifacts: `preds/codex/{vanilla,add_page_abstract}_nonNA_codex.csv`. Runners:
`experiments/codex/run_{vanilla,add_page_abstract}_all.sh`.

Takeaways:

- `add_page_abstract` still beats vanilla here (+0.0067 2cls, +0.006 Not Clear F1),
  same direction as val.100 but a smaller margin — consistent, not a fluke.
- Not Clear F1 drops to ~0.48 (vs ~0.66-0.71 on balanced val.100) because Not Clear
  is only 61/398 (~15%) here; the balanced set is better for separating prompts on
  the Not Clear axis, the full set is the realistic-distribution check.
- Remaining `_all` runs (add_context, add_page_abstract_context) pending.
