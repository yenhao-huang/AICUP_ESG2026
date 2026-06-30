# vlm_pred configs

YAML configs that drive `stage3/core/vlm_pred.py` instead of long CLI lines.

```bash
PY=/workspace/esg_contest/.venv/bin/python
cd exp/integrated_stage_predictions/0617/test_add_context/stage3
$PY core/vlm_pred.py --config ../configs/qwen_ctx.yaml
```

## Precedence

`explicit CLI flag` > `config value` > `built-in default` (DEFAULTS in vlm_pred.py).

So you can pin a config and tweak one thing on the fly:

```bash
$PY core/vlm_pred.py --config ../configs/qwen_ctx.yaml --limit 20
```

## Path resolution

Relative paths **inside a config** resolve against the config file's directory
(so `../data/...`, `../prompts/...`, `../preds/...`). Relative paths passed on the
CLI resolve against the current working directory.

## Keys

Every key is optional; omit one to keep its default. Full list and defaults live in
`DEFAULTS` in `core/vlm_pred.py`. Unknown keys raise an error.

| key | meaning |
|-----|---------|
| `data` / `output` / `prompt_path` | input rows / output CSV / system prompt file |
| `data_col`, `limit` | input column for the sentence; row cap (`null` = all) |
| `backend` | `qwen` (endpoint) or `codex` (CLI) |
| `endpoint`, `model`, `max_tokens`, `temperature`, `enable_thinking`, `timeout`, `retries`, `concurrency` | backend params (`model: null` = per-backend default) |
| `logprobs` | record per-token confidence (qwen only; codex leaves it empty) |
| `gate_csv`, `gate_col` | optional Stage 2 gate (`null` = predict every row) |
| `add_context`, `context_mode`, `context_max_chars` | same-page-content = whole matched page (`mode`: `all` / `hit_exact_window_norm_window`; `context_max_chars: 0` = whole page) |
| `add_evidence_string`, `add_promise_string`, `add_image` | data-use extras, OFF by default |
| `prompt_role` | chat role of the single tagged message (`user` / `system`) |

## Prompt template

The request is one tagged block in a fixed order (empty optional blocks omitted):

```
<same-page-context> ... </same-page-context>   # whole matched page, when add_context
<promise-string> ... </promise-string>          # when add_promise_string
<evidence-string> ... </evidence-string>         # when add_evidence_string
<data-prompt> ... </data-prompt>                 # the sentence to classify
<system-prompt> ... </system-prompt>             # classifier instructions
```

`prompt_role` sets whether this single message is sent as the `user` (default) or
`system` role. There is no before/after windowing — same-page-context is the whole page.

## Presets

| file | what |
|------|------|
| `qwen_ctx.yaml` | Qwen + same-page-content (all modes), logprobs on — the default probe |
| `qwen_ctx_window.yaml` | same, but only inject window-matched pages |
| `qwen_image.yaml` | Qwen VLM with the page image attached |
| `qwen_dataonly.yaml` | control: plain `data`, no context |
| `codex_dataonly.yaml` | Codex CLI backend, data-only |

> Data-use note: `add_context` / `add_evidence_string` / `add_promise_string` /
> `add_image` exceed the CLAUDE.md `data`-only default. They are experiment probes;
> promotion needs explicit sign-off.
