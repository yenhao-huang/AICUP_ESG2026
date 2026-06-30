# add_img_path — 2026-06-17 session script 記錄

本次 session 在 `0617/test_add_context/stage4/` 為 **Stage 4 verification_timeline**
建立的 test_add_context enrichment 探針，以及 page-image 對齊／`offsets.jsonl`
`image_path` 重作的所有 script 索引。所有路徑相對於 repo root
`/workspace/esg_contest/`。

> 資料使用：context / page-image / promise_string / page-abstract 變體都**超出
> CLAUDE.md 的 data-only 規則**，為使用者明確授權的 probe，不可當 data-only 路徑升級；
> vanilla 是唯一 data-only baseline。

base 目錄（以下 `S/` 代表）：
`exp/integrated_stage_predictions/0617/test_add_context/stage4/`

## 1. 影像 / 頁碼對齊（本次重點）

| script | 作用 |
| --- | --- |
| `S/build_offsets_image_path.py` | 在 `test_add_context/data/offsets.jsonl`（4000 列）每列加 `image_path` / `image_page_no` / `image_source` 三欄。挑頁規則：`matched_page_no` → 缺則 `weakly_matched_page_no`（依序取第一個有圖檔的頁），**不退回標註 `page_number`**。idempotent，支援 `--dry-run` / `--output`，in-place 會先備份 `offsets.jsonl.bak`。結果：3280 matched / 720 weakly / 0 未解析。 |
| `S/verify_image_alignment.py` | Gradio UI（port 7864/7872）：左＝input example（承諾句＋頁碼資訊），右＝實際 `--add-image` 會送出的頁圖，標示用 matched 或 weakly、與標註 `page_number` 是否不同。圖以 PIL 物件回傳（避開 Gradio 6 allowed_paths）。 |
| `S/core/build_prompt/add_image.py`（已改） | runtime 抓圖：`_page_candidates()` = `matched_page_no` + `weakly_matched_page_no`（去重保序），`image_path_for_row()` 依序試到第一個存在的圖；移除 `page_number` fallback。 |

執行：
```bash
cd S
/workspace/esg_contest/.venv/bin/python build_offsets_image_path.py --dry-run   # 看統計
/workspace/esg_contest/.venv/bin/python build_offsets_image_path.py             # 寫入 + .bak
fuser -k 7872/tcp; setsid /workspace/esg_contest/.venv/bin/python verify_image_alignment.py --port 7872 --host 0.0.0.0 >/tmp/st4_verify_img.log 2>&1 </dev/null &
```

備註（page 對齊結論）：標註 `page_number` 是 PDF 印刷頁碼，與渲染頁序差 1~2（封面/目錄偏移），約 46% 列不一致；OCR `matched_page_no` 與圖檔號一致且內容對得上（已對 id 11004 視覺確認 p46 即風險頁）。

## 2. Stage 4 預測 harness（`S/core/`，從 `../stage3/core/` 複製改寫）

| script | 作用 |
| --- | --- |
| `S/core/vlm_pred.py` | 主 runner（Codex / Qwen），gate_col=`promise_status`，輸入旗標控制 context/image/promise/evidence。 |
| `S/core/schemas.py` | TARGET_FIELD=`verification_timeline`，4 個 live label。 |
| `S/core/inference/parser.py` | timeline 標籤解析（snake_case / spaced / 中文）。 |
| `S/core/build_prompt/system_prompt.py` | 預設 prompt = `prompts/codex/vanilla.txt`。 |
| `S/core/build_prompt/{template,add_context}.py`、`S/core/inference/{qwen,codex}.py` | 沿用 stage3，未改邏輯。 |

## 3. Prompts（`S/prompts/codex/`）

| 檔 | 變體 | 輸入 / 設計 |
| --- | --- | --- |
| `vanilla.txt` | vanilla | = `configs/prompt/stage4/codex/boundary_rules_v4.txt`，data-only baseline |
| `add-context.txt` | add_context | + `<same-page-context>`；**scoped**：頁面只用來理解，年份只取自 `<data-prompt>` |
| `add-page-abstract.txt` | add_page_abstract | + 整頁，先摘要再判；**scoped**（同上，年份不採頁面） |
| `add-promise.txt` | add_promise | + `<promise-string>`，**第一級採計**（年份與 `<data-prompt>` 同等） |
| `add-image.txt` | add_image | + 頁圖（目前仍允許同等採計，未 scoped） |
| `all.txt` | all | context+image+promise（仍允許同等採計，未 scoped） |

## 4. 實驗 runner / 計分（`S/experiments/`）

| script | 作用 |
| --- | --- |
| `_common.sh` | 共用 `run_variant`，SET=100/full、BACKEND=codex/qwen、CONC/TIMEOUT/LIMIT |
| `run_vanilla.sh` `run_add_context.sh` `run_add_promise.sh` `run_add_image.sh` `run_add_page_abstract.sh` `run_all.sh` | 各變體 |
| `run_all_variants.sh` | 依序跑全部 + summary |
| `score_st4.py` | 4 類 full-coverage Macro-F1（Yes-gated benchmark） |
| `summarize.py` | 收集 `*.score.json` 成 Δ-vs-vanilla 對照表 |
| `RESULTS.md` | 100 列 screen 結果與分析 |

執行：
```bash
cd S/experiments
SET=100 CONC=8 bash run_add_page_abstract.sh
python summarize.py --set 100 --backend codex
```

## 5. 評估資料（`S/data/`，由 inline python 從 `../data/val_ctxhit.json` 篩 Yes 而成）

- `val_yes.json`（491 promise 列，含 GT timeline）
- `val_yes.100.json` / `.jsonl`（99 列分層 screen）
- 共用 `../data/offsets.jsonl`（已加 image_path 欄）

## 6. 其他

- `S/README.md`：stage4 探針總說明。
- `docs/feature_list.json`：新增 `stage4_test_add_context_enrichment_probe`（status experimental）。

## 100 列 screen 結果摘要（Codex gpt-5.5，Yes-gate 4 類 Macro-F1）

| 變體 | Macro-F1 | Δvanilla | acc |
| --- | --- | --- | --- |
| vanilla | 0.6440 | — | 0.5455 |
| add_promise（第一級） | 0.6332 | −0.0108 | 0.6061 |
| add_context | 0.5761 | −0.0679 | 0.5758 |
| all | 0.5574 | −0.0866 | 0.5758 |
| add_image | 0.5564 | −0.0876 | 0.5859 |
| add_page_abstract | 0.5152 | −0.1288 | 0.5657 |

註：add_context / add_page_abstract 的 score 為 scoped 改寫**前**的舊跑；prompt 已改 scoped，需重跑才反映新設計。within_2_years 僅 2 筆，Macro-F1 對它高度敏感，定論需 full 491。
