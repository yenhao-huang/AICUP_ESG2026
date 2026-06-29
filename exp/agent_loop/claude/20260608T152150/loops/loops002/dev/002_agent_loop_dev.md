# Loop 002 — Dev

Dev-phase scope (per plan `002_agent_loop_plan.md`, subtask `embed_index` + fallback-runner wiring):
build/cache the train-side, **data-only** RAG embedding index once, and prove the
`pred_by_bert.py` → `pred_by_bert_codex_rag.py` confidence-routed LLM-RAG fallback
wires end-to-end so the experiment phase can run the full grid. The full grid is
NOT run here — only a tiny smoke proof.

## Created / changed files (exact paths)

1. **CREATED** `core/human/predict/stage1/build_data_only_embed_index.py`
   - Builds the train-side embedding index over the train `data` field ONLY and
     writes it in the exact `.npz` format (`vecs` float32 L2-normalized, `ids`
     str) that the runtime `build_index()` in `pred_by_bert_codex_rag.py` loads
     back unchanged (no rebuild at runtime).
   - Embeds nothing but `row["data"]`; `promise_string`/`evidence_string`/labels
     are never read. Default embedder is the local 882M Chinese ModernBert
     SentenceTransformer (`/models/qihoo360-Zhinao-ChineseModernBert-Embedding`),
     which fits GPU0 (~0.9 GB) and uses the model's native CLS-pool + normalize.

2. **CHANGED** `core/human/predict/stage1/pred_by_bert_codex_rag.py` (additive, non-breaking; defaults unchanged):
   - Added a **SentenceTransformer embedding backend** (`--embedding-backend st`,
     helpers `_load_st_embed_model` / `_st_embed_batch`, wired into `batch_embed`
     and `retrieve_top_k`). This makes the query-time embedding use the SAME model
     and pooling as the cached index, so retrieval is consistent. (The pre-existing
     `local` backend uses last-token pooling, which is wrong for this CLS model;
     `st` is correct and is what the experiment phase must use.)
   - Added an **HTTP LLM backend** (`--llm-backend http`, default still `codex`):
     `call_llm_http()` sends the already-assembled RAG prompt to an
     OpenAI-compatible `/v1/chat/completions` endpoint (mirrors the Stage 2 sibling
     `core/human/predict/stage2/pred_by_bert_llm_rag.py`). New args:
     `--llm-base-url` (default `http://192.168.1.79:3134/v1`), `--llm-api-key`,
     `--llm-temperature`, `--llm-max-tokens`. When `--llm-backend http` and the
     model is still the codex default, the served model id is auto-resolved from
     `/v1/models` (vLLM serves `/workspace/llm_model`).
   - The routing logic, confidence gate, prompt construction, few-shot RAG, raw/
     token logging, and CSV merge are all REUSED untouched. Only the LLM transport
     and the embedding transport were extended.

3. Environment: installed `sentence-transformers==5.5.1` into the project venv
   `.venv` (was missing; torch/transformers/openai/sklearn already present).

No `docs/methods.md`, `docs/feature_list.json`, or `agent_loop_state.json` changes:
this is dev wiring only; promotion is gated to the reflect phase.

## Embedding index cache — path + how built

- **Cache path:** `data/generated/loop002_st1_rag/loop002_st1_train_data_only_modernbert_index.npz`
  (1000 rows, dim 768, ~3.0 MB). Location/name follow `docs/rules/filetree.md`
  (generated intermediate data → `data/generated/`) and `docs/rules/named.md`
  (loop-id-prefixed snake_case).
- **Build command (GPU0 only):**
  ```bash
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python core/human/predict/stage1/build_data_only_embed_index.py \
    --train-data data/raw_data/vpesg_4k_train_1000.json \
    --embed-cache data/generated/loop002_st1_rag/loop002_st1_train_data_only_modernbert_index.npz \
    --embedding-model /models/qihoo360-Zhinao-ChineseModernBert-Embedding \
    --device cuda:0 --batch-size 4 --max-seq-length 512
  ```
- Verified at runtime that every retrieved few-shot `info` text equals a train
  `data` field and matches NO `promise_string` (5/5 neighbours on the smoke row).

### Embedder choice / trade-off
The plan's default `qwen3-embedding-8b` (15 GB) does NOT fit GPU0's free VRAM
(~11 GB free; GPU1 reserved), and the remote embedding endpoint
`http://192.168.1.78:3132` is **DOWN** (probe → 000). The local Chinese ModernBert
embedder is the data-only, GPU0-fitting substitute: correct CLS pooling, normalized
768-dim, semantically sane (promise-vs-promise cos 0.318 > promise-vs-general 0.098).
The experiment phase's embedding-axis ablation runs over this cached index; if the
remote qwen endpoint comes back it can be added as a second index without changing
the runner.

## Smoke-test (wiring proof — NOT the grid)

Inputs: `exp/.../loops/loops002/exp/smoke/smoke6_val_public.json` (first 6 val_public rows, `id`+`data`).

1. **Base scoring** (per-row `score_yes`/`score_no` confirmed):
   ```bash
   CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   .venv/bin/python core/human/predict/stage1/pred_by_bert.py \
     --data <eval.json> --output <scored.csv> \
     --model hfl/chinese-roberta-wwm-ext-large \
     --finetune-path models/loop001_st1_a0_real_only/best_st1.pt \
     --mode finetune --text-col data --device cuda:0 --batch-size 4 --local-files-only
   ```
   Result: 6/6 scored, validation ok; lowest-conf row id 11896 = 0.846.

2. **Fallback** (1 row routed to the HTTP LLM, data-only retrieval; 0 errors):
   ```bash
   CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   .venv/bin/python core/human/predict/stage1/pred_by_bert_codex_rag.py \
     --bert-csv <scored.csv> --data <eval.json> --output <merged.csv> \
     --train-data data/raw_data/vpesg_4k_train_1000.json \
     --threshold 0.90 --info-mode data-only --text-col data --top-k 5 \
     --mode-context bert-prediction --prompt strict_negative_with_certficate \
     --embedding-backend st \
     --embedding-model /models/qihoo360-Zhinao-ChineseModernBert-Embedding \
     --embedding-device cuda:0 --embed-batch-size 4 \
     --embed-cache data/generated/loop002_st1_rag/loop002_st1_train_data_only_modernbert_index.npz \
     --llm-backend http --llm-base-url http://192.168.1.79:3134/v1 --timeout 120
   ```
   Result: high_conf=5, low_conf=1, codex_called=1, codex_errors=0,
   changed_by_codex=1 (id 11896 No→Yes), served model `/workspace/llm_model`,
   validation ok. Endpoint `http://192.168.1.79:3134/v1/models` → HTTP 200.

Smoke artifacts: `exp/.../loops/loops002/exp/smoke/{smoke6_val_public.json,
smoke6_bert_scored.csv, smoke6_rag_merged.csv, raw/0001_11896.json, summary.json}`.

## Exact commands the experiment phase will use

GPU policy for ALL local steps: `CUDA_VISIBLE_DEVICES=0`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--device cuda:0`,
`--batch-size 4`. GPU1 is reserved (unrelated job) — do NOT use it. Use `.venv/bin/python`.

**Step A — base scoring (subtask `base_score`, run once each, reuse across the grid):**
```bash
# val_public (selection) — a0 base
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python core/human/predict/stage1/pred_by_bert.py \
  --data data/benchmarks/val_public.json \
  --output exp/agent_loop/claude/20260608T152150/loops/loops002/exp/base/a0_val_public_scored.csv \
  --model hfl/chinese-roberta-wwm-ext-large \
  --finetune-path models/loop001_st1_a0_real_only/best_st1.pt \
  --mode finetune --text-col data --device cuda:0 --batch-size 4 --local-files-only
# val_test (blind gate, score now but only the SELECTED config is merged/evaluated on it once)
... same as above with --data data/benchmarks/val_test.json --output .../a0_val_test_scored.csv
# a3_b1 secondary base on val_public (base-invariance diagnostic only)
... --finetune-path models/loop001_st1_a3_b1/best_st1.pt --data data/benchmarks/val_public.json --output .../a3b1_val_public_scored.csv
```

**Step B — fallback grid on val_public (subtask `fallback_grid_valpublic`, selection only).**
For each config vary `--threshold T` ∈ {0.60,0.70,0.80,0.90,0.95}, then refine
`--top-k K` ∈ {3,5,8}, `--prompt P` ∈ {strict_negative, strict_negative_with_certficate, default},
`--mode-context C` ∈ {bert-prediction, none}. Template (LOCK `--info-mode data-only --text-col data`):
```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python core/human/predict/stage1/pred_by_bert_codex_rag.py \
  --bert-csv exp/.../loops/loops002/exp/base/a0_val_public_scored.csv \
  --data data/benchmarks/val_public.json \
  --output exp/.../loops/loops002/exp/grid/T<thr>_K<k>_<prompt>_<ctx>/merged.csv \
  --train-data data/raw_data/vpesg_4k_train_1000.json \
  --threshold <T> --top-k <K> --prompt <P> --mode-context <C> \
  --info-mode data-only --text-col data \
  --embedding-backend st \
  --embedding-model /models/qihoo360-Zhinao-ChineseModernBert-Embedding \
  --embedding-device cuda:0 --embed-batch-size 4 \
  --embed-cache data/generated/loop002_st1_rag/loop002_st1_train_data_only_modernbert_index.npz \
  --llm-backend http --llm-base-url http://192.168.1.79:3134/v1 --timeout 120
```
Margin-router ablation R: implement by mapping the margin band to an equivalent
`--threshold` (max-prob band) on the best T/K/P/C arm — no new runtime field.

Score each merged CSV vs offline GT with sklearn `f1_score(average="macro")`
(same metric as `core/eval/eval_bert.py`). GT labels are OFFLINE scoring only.

**Step C — select + gate (subtask `select_and_gate`):** pick the single highest
val_public Macro-F1 config (tie-break: fewer escalations), confirm it beats the a0
base val_public 0.8031, then run that ONE config once on the `a0_val_test_scored.csv`
+ `data/benchmarks/val_test.json` and compare to BASE_GATE 0.802383. Base-invariance
diagnostic: re-run the selected config on `a3b1_val_public_scored.csv`.

## Data-use compliance notes

- **Runtime path is `data` ONLY.** `--info-mode` LOCKED `data-only`, `--text-col`
  LOCKED `data`. BERT forward consumes `data`; the embedding index + query
  embeddings consume `data`; the LLM prompt body = query `data` + retrieved train
  `data`.
- **promise_string never enters runtime/RAG.** Verified: the index script embeds
  only `row["data"]`; at runtime every retrieved `info` matched a train `data`
  field and matched no `promise_string` (5/5 neighbours). No offline
  promise_string use in this loop at all (no synthesis).
- **Few-shot labels** are the retrieved TRAIN rows' `promise_status`
  (permitted in-context demonstration labels), never the query row's annotation.
- **GT labels** are used only for offline Macro-F1 scoring of merged CSVs.
- Any config using `--info-mode promise-str-only` / `data+promise_str`, or
  `--text-col` ≠ `data`, is INVALID and must be excluded.

## Blocked / notes for exp phase

- Remote embedding endpoint `http://192.168.1.78:3132` is DOWN; the grid must use
  `--embedding-backend st` with the cached ModernBert index (do NOT pass
  `--embedding-backend openai`). The LLM endpoint `http://192.168.1.79:3134` is UP
  (HTTP 200, model `/workspace/llm_model`).
- `summary.json`'s `embedding_base_url` field is cosmetic (prints the unused
  default) when backend=`st`; ignore it. Embedding backend actually used is in
  `embedding_backend`.
