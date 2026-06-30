# weakly_img_path — offsets.jsonl top@3 頁圖檢索 (2026-06-18)

為 `offsets.jsonl` 每列用 `data` 文字打 RAG Search API，取語意最相關的 top@3
頁面圖片、落地存檔，並加上 `weakly_img_path` 欄（長度 3 的圖片路徑清單）。

「weakly」= 用語意檢索（非 OCR 精準頁碼）得到的弱對齊頁圖，沿用 0617
`add_img_path` / `page_table_image_path` 的 image-path 探針脈絡。

## 產出

| 項目 | 路徑 |
| --- | --- |
| 腳本 | `exp/integrated_stage_predictions/0618/build_weakly_img_path.py` |
| 加欄後 offsets | `exp/integrated_stage_predictions/0618/weakly_img_path/offsets.jsonl`（4000 列，全填） |
| 落地圖片 | `/data/integrated_stage_predictions_0618/weakly_img/`（3626 張獨立 JPEG，1.1 GB） |
| /data 工作檔 | `/data/integrated_stage_predictions_0618/offsets_work.jsonl`（checkpoint 來源，與定版檔同內容） |

`weakly_img_path` 欄格式（每列）：
```json
"weakly_img_path": [
  "/data/integrated_stage_predictions_0618/weakly_img/<報告>_p<頁>__<page_id>.jpg",  // rank1
  "...rank2.jpg",
  "...rank3.jpg"
]
```

## 資料流

```
offsets.id --join--> 原始 json (train/val/test) 的 `data` 文字（唯一允許的 raw 輸入）
   --POST 192.168.1.78:8766 /api/search (config configs/retriever/search.yml,
     collection colnomic_esg_contest, top_k=3)--> 3 個 hit
   --存圖(以 page_id 去重)--> weakly_img_path = [rank1, rank2, rank3]
gate 外/查無 -> 該列無此欄。本批 4000 列全部成功取得 3 張。
```

## 重跑 / 續跑

```bash
P=/workspace/esg_contest/.venv/bin/python
WORK=/data/integrated_stage_predictions_0618/offsets_work.jsonl
$P exp/integrated_stage_predictions/0618/build_weakly_img_path.py \
    --offsets "$WORK" --out "$WORK" --concurrency 8 --checkpoint 200 --top-k 3
# 完成後再 cp "$WORK" 回本目錄 offsets.jsonl
```
已填且檔案都在的列會自動跳過（`--no-resume` 可關閉）；圖以 `page_id` 去重，
重複頁不重存。

## 注意

- **儲存路徑**：圖片與反覆 checkpoint 一律寫 `/data`（66G 穩定）；`/workspace`
  是 99% 滿的揮發性分割區，初版直接寫 offsets 到 /workspace 曾觸發 ENOSPC，
  故改為 checkpoint 寫 /data、最後只複製一次 2.9MB 定版檔回 /workspace。
- **格式**：存原生 JPEG（API 回傳即 JPEG，~0.3MB/張）。若轉 PNG 最壞約 48GB，
  會撐爆 /data，故不轉。
- **資料使用**：查詢只用 `data` 欄（data-only）；落地圖片是使用者授權的
  enrichment 探針，非可升級的 data-only runtime 路徑。collection
  `colnomic_esg_contest` 對應 `/workspace/data/esg_contest/*.pdf`（台灣永續報告書，
  VeriPromiseESG4K 來源）。
- 來源 offsets = `0617/test_add_context/data/offsets.jsonl` 的複本（0617 原檔未動）。
```
