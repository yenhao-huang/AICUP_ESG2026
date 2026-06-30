# add_page_abstract

為每筆 val row 所在的「報告同頁」產生一段繁中摘要（abstract），透過本機 Qwen
server `http://192.168.1.78:3132`（OpenAI 相容，model `local-qwen`）。

## 方法

1. 每筆 row 用與 runtime 相同的對位流程
   （`core/human/predict/stage4/build_page_context`：`pdf_url/id → doc_id`、
   `collapse(data)[:18]` 文字比對定位頁碼、吸收 ±1~2 偏移）找到該句來源的 OCR
   實體頁 `(doc_id, page_no)`。
2. 以 `(doc_id, page_no)` 去重，每個唯一頁面只摘要一次。
3. 取該頁 `raw_page_table.jsonl` 的 `text_clean`（缺則 `text`），送 server 摘要。
   摘要 prompt：`prompts/page_abstract.txt`（2–4 句、120 字內、不杜撰）。

## 執行

```bash
EXP=exp/integrated_stage_predictions/0617
# 全部（val_ctxhit.json 600 列 → 551 唯一頁）
.venv/bin/python $EXP/add_page_abstract/build_page_abstract.py \
  --data $EXP/test_add_context/data/val_ctxhit.json \
  --out-dir $EXP/add_page_abstract --concurrency 8
# 100 頁 smoke
... --data $EXP/test_add_context/data/val.100.jsonl --out-dir $EXP/add_page_abstract/out_val100
```

可調：`--max-page-chars`（頁面字數上限，預設 6000）、`--max-tokens`（摘要長度，256）、
`--temperature`（0.3）、`--concurrency`（server 有 8 slot）、`--limit`（smoke）、
`--prompt-path`。

## 輸出

- `page_abstracts.jsonl`：每個唯一頁一筆 `{doc_id, page_no, page_id, image_path,
  company, ticker, n_chars, abstract}`。
- `val_with_page_abstract.jsonl`：每筆 row 一筆 `{id, data, company, page_number,
  matched_doc_id, matched_page_no, page_abstract}`，供下游當脈絡使用。
- `out_val100/`：100 頁 smoke 結果。

## 資料合規旗標

page abstract 由 `raw_page_table` 的報告 OCR 文字摘要而來，與 test_add_context 的
same-page context 同類，**超出** CLAUDE.md「只用 `data`」預設邊界——屬使用者明確指定
的實驗方向。**未**使用 `evidence_string` / `promise_string` / 任何標註或標籤欄位。
若要 promote 進 runtime 仍需依 loop 規範簽核。
