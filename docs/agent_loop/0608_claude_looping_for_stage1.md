目標：使用 Claude agent loop 優化 stage1 step
loop: 100
Steps

A. 模型訓練

1. 合成各種資料集
2. 混合原資料集
3. 模型訓練/評估

B. 模型測試

1. 用訓練的模型來預測測試集
2. 將低信心的預測交給 LLM 判斷 （192.168.1.79:3134）
    1. 透過 RAG 來檢索相關的 prompt

資料集來源：
1. data/raw_data/vpesg_4k_train_1000.json
2. 查詢每個 data 來源: data/generated/raw_doc_table、data/generated/raw_page_table

驗證集來源：data/benchmarks/test.json

測試集來源：data/benchmarks/val_public.json

優化方向：

1. 生成最適合這個任務的資料集 （data 來生、promise-str來生、data+promise 來生 or data 對應的 pdf 內容來生）
2. 混合合成資料集與真實資料集 （權重自選）
3. 模型架構調整
4. Loss function 調整
5. 超參數調整
6. 信心分數門檻調整
7. LLM 的 prompt 調整 

Harness:

1. 不能用 post process（keyword rule、rule-based 後處理）
2. 可以用 LLM fallback（低信心 BERT 預測交 LLM 判斷）
3. 需要通過最終條件此方法才被接受

最終條件：

1. 評估 benchmark/val_test.json 時要比 baseline 高分