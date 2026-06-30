# Stage 4 `verification_timeline` — test_add_context Probe (2026-06-17)

Tests whether enriching the Stage 4 timeline classifier with extra inputs
(same-page context, page image, promise_string) beats the data-only vanilla
prompt. Built on the same modular VLM harness as the sibling `../stage3/`.

## ⚠ Data-use status

This is an **explicit, user-approved probe** that intentionally exceeds the
CLAUDE.md `data`-only rule. `same-page-context` and the page image inject raw
report content beyond the `data` field, and `promise_string` is an annotation
field. Results here are a research signal only and **must NOT be promoted as a
data-only runtime path** (no `docs/methods.md` / `agent_loop_state.json` update
from this directory). The vanilla variant is the only data-only baseline.

## Variants

All variants share the boundary_rules_v4 four-step judgment logic; they differ
only in which inputs are exposed. Unlike Stage 3 (where context is scoped to
*not* add specificity), here the extra inputs are **allowed to help identify the
target year**, since the GT timeline reflects the promise's real completion
horizon.

| variant       | prompt                       | inputs beyond `data`                         |
| ------------- | ---------------------------- | -------------------------------------------- |
| `vanilla`     | `prompts/codex/vanilla.txt`  | none (= `configs/prompt/stage4/codex/boundary_rules_v4.txt`) |
| `add_context` | `prompts/codex/add-context.txt` | `<same-page-context>` (whole matched page OCR) |
| `add_promise` | `prompts/codex/add-promise.txt` | `<promise-string>` (annotation)            |
| `add_image`   | `prompts/codex/add-image.txt`   | page image (`--image`, gpt-5.5 / qwen VLM) |
| `all`         | `prompts/codex/all.txt`         | context + image + promise_string           |

## Data

- `data/val_yes.json` — 491 promise rows (`promise_status == Yes`) from the
  600-row `../data/val_ctxhit.json`, carrying GT `verification_timeline`.
- `data/val_yes.100.json` / `.jsonl` — 99-row stratified screen subset (all four
  timeline classes present). GT distribution: already 46, between_2_and_5 29,
  more_than_5 22, within_2 2.

Class join for context/image uses `../data/offsets.jsonl` (covers all 491 ids)
plus the repo doc/page tables via `core/human/predict/stage4/build_page_context.py`.
GT labels are used ONLY for offline scoring, never as model input.

## Run

```bash
# 100-row screen, all variants, Codex gpt-5.5 (default), then summary table:
SET=100 bash experiments/run_all_variants.sh

# full 491-row run:
SET=full bash experiments/run_all_variants.sh

# single variant / quick wiring check:
SET=100 LIMIT=3 bash experiments/run_vanilla.sh
```

Knobs (see `experiments/_common.sh`): `SET=100|full`, `BACKEND=codex|qwen`,
`MODEL=gpt-5.5`, `CONC`, `TIMEOUT`, `LIMIT`. Codex gpt-5.5 is multimodal and
honours `--image`, so `add_image` / `all` send the page image for real.

## Outputs

```text
preds/codex/<variant>_<set>_<backend>.csv          prediction CSV
preds/codex/<variant>_<set>_<backend>.score.json   Macro-F1 / acc / per-class
experiments/logs/<variant>_<set>_<backend>.log      run log
```

## Scoring

`experiments/score_st4.py` — full-coverage Macro-F1 over the four live timeline
classes on the Yes-gated benchmark. Missing / invalid predictions fold to a
never-correct sentinel (counted against the score). `experiments/summarize.py`
collects all `*.score.json` into one Δ-vs-vanilla table.

## Harness layout (`core/`, copied & adapted from `../stage3/core/`)

```text
core/vlm_pred.py                 runner (gate_col=promise_status; data-only by flag)
core/schemas.py                  TARGET_FIELD=verification_timeline, 4 live labels
core/inference/parser.py         snake_case / spaced / 中文 timeline-label parser
core/build_prompt/system_prompt.py  default = prompts/codex/vanilla.txt
core/build_prompt/{template,add_context,add_image}.py  (unchanged from stage3)
core/inference/{qwen,codex}.py   (unchanged from stage3)
```
