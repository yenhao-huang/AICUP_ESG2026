### 合成資料 a3_b1
用 data + promise_str 改寫

合成資料的 script
`core/data/synthesize_st1_data_only.py`



```bash
1. 挑來源列(只挑正例)

從真實 train set data/raw_data/vpesg_4k_train_1000.json 篩出同時滿足兩條件的列(build_a3, line 288):

- promise_status == "Yes"
- promise_string 非空

也就是只拿「本來就是承諾、而且標註者有框出承諾片段」的真實列當種子。

2. 用 LLM 把 (data context + promise span) 改寫成一句承諾

對每個種子列,組 prompt(line 196-202),內容是把真實 data 當 context、把真實 promise_string 當 promise,丟給繁中 LLM endpoint(192.168.1.79:3134):

你是繁體中文永續報告寫作助手。以下提供一段報告原文context,
以及其中表達企業承諾的核心片段promise。請以promise為核心,
搭配context的背景資訊,改寫成一句通順、明確表達企業未來承諾的繁體中文句子。
句子必須仍清楚包含該承諾。只輸出最終句子。

context:\n{data_text}\n\npromise:\n{promise_text}

LLM 回傳的那一句,就成為合成列的 data,label 一律設 promise_status = "Yes"(line 216,A3 全是正例)。輸出列標 syn_source=A3_data_plus_promise、syn_method=llm、syn_parent_id=<原id>,id 為 syn_a3_<原id>。

3. Fallback(LLM 掛掉或輸出太短時)

走 fallback_a3(line 136-144):不呼叫 LLM,直接把 promise span 套進 3 種模板之一,例如:

- {p}此舉為本公司永續策略的一環。
- 為回應利害關係人期待,{p}
- {p}相關規劃將納入後續永續報告揭露。

(此次 loop001 的 method_counts 顯示 A3 全 300 筆都是 llm,沒有走 fallback。)

4. data-only 合規護欄

- 輸出列用 SCHEMA_EMPTY 把 promise_string / evidence_string 等欄位清空。
- assert_no_promise_string_leak(line 299-302)會檢查任何合成列都不得殘留 promise_string,否則直接 raise。
- 所以 promise_string 只在離線生成時被讀取,絕不寫進訓練列;runtime 模型輸入仍只有 data,符合 CLAUDE.md 的 data-only 規則。

5. 形成 pool 再混入

生成後 label-stratified 上限 pool_cap=300 → pool_a3_data_plus_promise.json(300 筆 unique)。之後 sample_mix 從這 300 筆可重複抽樣 b1=500 / b2=1000 / b3=2000 筆,與 1000 筆真實資料串接,得到 mix_a3_b1/b2/b3.json。
```