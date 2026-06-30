# add_context 實作說明

依 [`add_context_plan.md`](add_context_plan.md) 將 `core/human/predict/stage3/pred_by_qwen.py`
重構為模組化的 Stage 3 VLM 預測器。

- 產出位置：`exp/integrated_stage_predictions/0617/test_add_context/stage3/`
  - plan 寫的 `test_add_content` 為 typo，沿用既有目錄 `test_add_context`。
- 參考來源：`core/human/predict/stage3/pred_by_qwen.py`、`pred_by_codex.py`、
  `core/human/predict/stage4/build_page_context.py`（同頁脈絡 join / 視窗抽取直接複用）。

---

## 1. 目錄結構

```
stage3/core/
  vlm_pred.py                主程式：讀資料 → 組 prompt → 推論 → 解析 → 寫 CSV
  schemas.py                 output 欄位定義 + 結果物件（含每 token 信心）
  build_prompt/
    system_prompt.py         設定當前 prompt（system message）
    add_context.py           same-page-content + evidence_string + promise_string
    add_image.py             頁面圖片（VLM）
  inference/
    qwen.py                  llama-server / OpenAI 相容 endpoint（支援圖片 + logprobs）
    codex.py                 Codex CLI
    parser.py                output → label（markdown / JSON / CoT 容錯）
```

### import 設計（為何不會撞名）

repo 根目錄本身就有一個 `core/` 套件，本實驗也叫 `core/`。為避免 `import core`
歧義，採兩條獨立路徑：

- 本地模組以「`stage3/core` 這個目錄」加入 `sys.path`，用裸名匯入：
  `import schemas`、`from build_prompt.add_context import ...`、`from inference.qwen import ...`。
  本地沒有名為 `core` 的子模組，因此不會與 repo `core` 衝突。
- 需要複用 repo 的 `build_page_context` 時，才把 repo 根加入 `sys.path`，用
  `import core.human.predict.stage4.build_page_context`。repo 根的 `core` 只走這條。

各模組用 `_find_repo_root()`（向上找同時含 `core/human/predict` 與 `data/generated`
的目錄）定位 repo 根，因此不依賴硬編碼的 `parents[N]` 深度。

---

## 2. 各模組實作

### schemas.py — 輸出格式

- `OUTPUT_COLUMNS`：`id, evidence_quality, evidence_quality_raw, evidence_quality_source,
  evidence_quality_reason, context_hit, token_confidence`。
- `PredictionResult` dataclass + `to_row()`，集中產生 CSV 列。
- **每 token 信心**：`token_confidence` 為 `[{token, logprob, prob}]` 的 JSON 字串。
  Qwen 由 endpoint 的 logprobs 取得；**Codex 不支援，留空**（符合 plan「若是 codex 不需要」）。

### build_prompt/system_prompt.py — 設定 prompt

- `load_system_prompt(path)`：讀取 prompt 檔當作 system message。
- 預設 `DEFAULT_PROMPT_PATH = test_add_context/prompts/clear_notclear_with_context_scoped.txt`
  （本實驗的 scoped 同頁脈絡 prompt）。可換成任意檔，例如 stage4 的
  `configs/prompt/stage4/codex/*.txt`。

### build_prompt/add_context.py — 同頁脈絡（核心）

`ContextBuilder.augment(row, data)` 依序組裝以下區塊，回傳 `(augmented_data, hit_kind)`：

1. `承諾標註：`（`promise_string`，需 `--add-promise-string`）
2. `佐證內容：`（`evidence_string`，需 `--add-evidence-string`）
3. `同頁內容：`（same-page-content，需 `--add-context`）

最終格式沿用 build_page_context 的 header：`承諾句：{data}\n\n{各區塊}`。

**same-page-content 取得流程**

- 來源：`data/offsets.jsonl`（預先算好的 `id → hit_kind, matched_page_no`）
  + `data/generated/raw_doc_table.jsonl`（`url → doc_id`）
  + `data/generated/raw_page_table.jsonl`（`(doc_id, page_no) → OCR text`）。
- 用 `offsets.jsonl` 的 `matched_page_no` 直接定位頁面，再於該頁用 prefix（exact→NFKC norm）
  找到 `match_idx`，呼叫 `bpc.extract_window()` 抽出視窗（budget / after_biased / cross_page）。
- 若某 id 不在 offsets 中，且 mode 允許，退回 live 定位 `bpc.locate_anchor_fallback()`。

**mode（plan 要求的多模式）**

| mode | 接受的 hit_kind | 行為 |
|------|-----------------|------|
| `all`（預設） | 全部 | 任何命中皆注入；無 offset 時 live-locate 退回 |
| `hit_exact_window_norm_window` | `hit_exact_window`, `hit_norm_window` | 只注入這兩種，其餘 data-only（`skip_mode:<kind>`） |

`hit_kind` 會記進 CSV 的 `context_hit` 欄，方便事後分析命中來源。

> **資料使用註記**：same-page-content 注入原始 OCR、`evidence_string`/`promise_string`
> 為標註欄位，三者皆超出 CLAUDE.md 的 `data`-only 預設，故**預設關閉**，僅供 test_add_context
> 探查，promote 前需明確同意。

### build_prompt/add_image.py — 頁面圖片

- `ImageBuilder` 從 `raw_page_table.jsonl` 建 `(doc_id, page_no) → 圖片路徑`。
- plan 寫欄位是 `image_url`，但實際表上是 **`image_path`**（repo 相對 PNG 路徑）；
  兩個 key 都接受，`image_path` 優先。
- 頁碼取 offsets 的 `matched_page_no`，無則用 row 的 `page_number`。
- `data_url_for_row()` 把圖片編成 `data:image/png;base64,...` 供 VLM 使用。

### inference/qwen.py — llama-server 後端

- `POST /v1/chat/completions`（OpenAI 相容），system + user 訊息。
- 帶圖時 user content 改為 `[{type:text}, {type:image_url}...]`。
- `enable_thinking=false`（bare label）；`--logprobs` 時帶 `logprobs:true` 並解析
  `choices[0].logprobs.content` 成每 token 信心（`prob = exp(logprob)`）。
- 連線錯誤指數退避重試。

### inference/codex.py — Codex CLI 後端

- 複用 `pred_by_codex.py` 的 `codex exec --json --output-last-message` 子行程呼叫。
- 不支援 logprobs（token 信心回 `None`），不支援圖片（簽名相容但忽略）。

### inference/parser.py — output 解析

`parse_label(raw) → (label, reason)`，容錯處理 plan 提到的 markdown 問題：

- 去除 ```` ``` ```` code fence、行內 backtick、`*#>` 等 markdown 裝飾。
- 解開 JSON / 引號包裝（`{"evidence_quality":"Clear"}`、`"Clear"`）。
- CoT 回覆取最後一個輸出標記（`輸出：` / `output:` / `答案：`）之後的 label。
- 中英文 alias + 全半形標準化；最長 label 優先（避免 `Clear` 命中 `Not Clear`）。

### vlm_pred.py — 主程式

- **預設輸入** `data/val_ctxhit.json`；**預設無 gate**，給哪些 example 就預測哪些。
  `--gate-csv` 才用 Stage 2 `evidence_status` 過濾（被擋的標 `N/A`）。
- `--backend qwen|codex`；Qwen 用 `--concurrency` 多執行緒，Codex 強制單執行緒。
- 流程：`read_data_rows → ContextBuilder.augment → (ImageBuilder) → backend.classify
  → parse_label → PredictionResult → write_csv`。

---

## 3. 驗證

對線上 `local-qwen`（`:8000`）實測通過：

- **離線單元**：parser 對 ```` ``` ````、JSON、`輸出：<label>`、`**bold**`、空字串全數正確；
  ContextBuilder 在 `all` 與 `hit_exact_window_norm_window` 下行為正確（後者對
  `fallback_embedding` 正確 `skip_mode`）；ImageBuilder 在 val 600 筆解析到 118 筆頁面圖。
- **線上推論**：`--limit 2 --logprobs` 兩筆皆注入 `offset_hit_exact_window` 脈絡、
  輸出 `Not Clear`、CSV 記錄每 token 機率，欄位完整。
- **錯誤路徑**：endpoint 不通時優雅落為 `except` 列並照常寫出 CSV。

val 的 `offsets.jsonl` hit_kind 全為 `hit_exact_window`（600/600），故兩種 mode 在
此資料集都會注入脈絡。

## 4. 範例指令

```bash
PY=/workspace/esg_contest/.venv/bin/python
cd exp/integrated_stage_predictions/0617/test_add_context/stage3

# Qwen + 同頁脈絡（預設）+ 每 token 信心
$PY core/vlm_pred.py --output preds/st3_qwen_ctx.csv --logprobs

# 只注入 window 命中
$PY core/vlm_pred.py --output preds/st3_qwen_win.csv --context-mode hit_exact_window_norm_window

# data-only 控制組
$PY core/vlm_pred.py --output preds/st3_qwen_ctrl.csv --no-add-context

# VLM：加頁面圖片
$PY core/vlm_pred.py --output preds/st3_qwen_img.csv --add-image

# Codex 後端
$PY core/vlm_pred.py --backend codex --no-add-context --output preds/st3_codex.csv
```

## 5. 未動到的部分

這是 `exp/` 下的實驗 harness，**未**修改 `core/e2e`、`docs/methods.md`，也未做任何
promotion；資料使用旗標全部預設關閉。
