參考 core/human/predict/pred_by_qwen.py 的程式。
給我在 exp/integrated_stage_predictions/0617/test_add_content/stage3/ 目錄下給我 core/vlm_pred.py 但需要以下更動


* input 的預設是 data/val_ctxhit.json 格式，支援不加入 gate 機制，給哪些 example 就預測哪些 example。
* 如何加入 same-page-content?
    * 透過 data/offsets.jsonl、data/generated/raw_page_table.jsonl、data/generated/raw_doc_table.jsonl 來取得。
    * same-page-content 支援多個 mode: all(default)、hit_exact_window_norm_window
* add_image.py
    * 透過 data/generated/raw_page_table.jsonl 的 image_url 欄位 
* 拆成以下目錄格式 

目錄
```bash
core/build_prompt/system_prompt.py、add_context.py、add_image.py
core/inference/qwen.py、codex.py
core/inference/parser.py
core/vlm_pred.py
core/schemas.py
```
system_prompt.py: 設定目前的 prompt (e.g., configs/prompts/stage4/clear_andnotclear.txt)
add_context.py: 能夠允加入 same-page-content、能夠允許加入 evidence_string、能夠允許加入 promise_string
add_image.py: 能夠允許加入圖片
core/inference/parser.py: 處理 output parse，可能有 markdown 的解析問題。
core/schemas.py: 定義 output 格式 (output 欄位、每個 token 信心(若是 codex 不需要))