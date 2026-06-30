# test_add_feature — Method A：可驗證性輔助監督 (ST3)

對應 `docs/plans/0617_improvement.md` §P3。目標：提升 Stage 3 (evidence_quality) 對
**Not Clear** 類的鑑別力，做法是把「可驗證性 (verifiability)」當成**輔助訓練監督**，
而不是再調 loss 或加關鍵字後處理。

## 假設（已用資料驗證）

Clear vs Not Clear 的本質是「承諾是否可檢核/可量化」，而非語氣。對
`data/raw_data/vpesg_4k_train_1000.json` 統計（見 `verifiability_features.py`）：

| 訊號（從 `data` 文字算） | Clear | NotClear | 差 |
| --- | ---: | ---: | ---: |
| has_quantified_target（數字+單位/%） | 0.486 | 0.089 | **+0.397** |
| has_temporal_anchor（年份/期限） | 0.607 | 0.282 | **+0.325** |
| has_percent | 0.223 | 0.065 | +0.158 |
| has_binding_commitment（簽署/制定/認證/ISO…） | 0.560 | 0.452 | +0.108 |
| 避險詞（持續/致力/推動…） | 0.732 | 0.742 | **−0.010（無訊號）** |

→ 真正的訊號是「可量化/可檢核」，避險詞無鑑別力。但它**不是硬規則**（36% NC 也含數字、
21% Clear 無數字），是組合性、語境性的，所以用輔助監督讓 encoder 學，而非 keyword 後處理。

## 兩個模型（唯一差異 = 輔助頭）

| 模型 | 架構 | 角色 |
| --- | --- | --- |
| **vanilla** | `BertClassifier`：encoder + 1 個 3-class 主頭 | 控制組（隔離「重訓」效果） |
| **multitask** | 共享 encoder + 3-class 主頭 `classifier` + `aux_head`（4 個二元 verifiability 頭） | **Method A** |

- 主頭標籤空間與 exp23 對齊：`gated_mis` 3-class {Clear:0, Not Clear:1, Misleading:2}，
  只取 `evidence_status == "Yes"` 且 `evidence_quality` 非空的列（train 677 / val 668）。
- 輔助頭只在**訓練時**提供梯度（loss = 加權CE(主) + λ·BCE(輔)，λ 預設 0.5）；
  **推論只用主頭** → multitask checkpoint 可用 `strict=False` 載入成一般 `BertClassifier`
  （`aux_head.*` 被忽略），與 exp23 完全可比。

## 合規 (data-only)

- 模型輸入只有 `d["data"]`。
- 輔助標籤由 `data` 文字 regex 衍生（`verifiability_features.py`），**未用**
  `promise_string` / `evidence_string` / 任何 annotation 欄位。
- 屬「訓練時輔助監督」，非 harness 禁止的 rule-based 後處理（輸出後不套任何規則）。

## 檔案

```
verifiability_features.py   data-only 可驗證性特徵（4 個輔助任務 + 自檢報告）
train_st3_feature.py        --mode vanilla|multitask 訓練（train_1000 訓練 / val_1000 選模）
compare_st3.py              載入 exp23 / vanilla / multitask，於 val_1000 ST3-applicable 子集逐類別比較
run.sh                      串接：特徵自檢 → 訓練兩模型 → 比較
results/                    train_*.json、compare_st3.json、run.log
```

## 執行

```bash
bash exp/integrated_stage_predictions/0617/test_add_feature/run.sh
# 可調： DEVICE=cuda:0 SEED=42 AUX_LAMBDA=0.5 bash .../run.sh
```

- 訓練 recipe 對齊 exp23 的 `large` 設定：`hfl/chinese-roberta-wwm-ext-large`,
  max_len 512, batch 8, grad_accum 2, lr 1e-5, epochs 5, weighted-CE 主損失。
- checkpoint 存 `/models/test_add_feature_0617/{vanilla,multitask}/best_st3.pt`（依 CLAUDE.md
  儲存規則，模型不放 `/workspace`）。

## 評估口徑

`compare_st3.py` 在 val_1000 的 **ST3-applicable 子集**（GT `evidence_status==Yes`，3-class）評估，
直接量測 ST3 模型本身的 Clear/NC 鑑別力（N/A 類由 cascade 決定、非這三個模型產生，故不在此比較）。
重點看 **Not Clear 的 precision / recall / F1** 與 3-class macro-F1；對照組為 vanilla 與 exp23。

## 結果（seed=42, aux_lambda=0.5；val_1000 ST3-applicable 子集 N=668）

來源：`results/compare_st3.json`、`results/train_*.json`、`results/run.log`。
GT 分布：Clear 566 / Not Clear 101 / Misleading 1。

| model | macroF1(3c) | 2c-F1(Clear/NC) | acc | Clear-F1 | NC-P | NC-R | NC-F1 | Mis-F1 | NC 預測數 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ref (exp23) | 0.4564 | 0.6850 | 0.823 | 0.8939 | 0.434 | 0.525 | 0.475 | 0.000 | 122 |
| vanilla（控制） | 0.4686 | 0.7033 | 0.832 | 0.8991 | 0.460 | 0.564 | 0.507 | 0.000 | 124 |
| **multitask（Method A）** | **0.4724** | **0.7090** | **0.834** | **0.8997** | **0.465** | **0.584** | **0.518** | 0.000 | 127 |

差異分解：

| 比較 | 意義 | macroF1(3c) | NC-F1 | NC-P |
| --- | --- | ---: | ---: | ---: |
| multitask − vanilla | **Method A（輔助頭）的乾淨效果** | **+0.0038** | **+0.0109** | +0.0049 |
| vanilla − exp23 | 純重訓（換 train_1000）效果 | +0.0122 | +0.032 | +0.026 |
| multitask − exp23 | 總效果 | +0.0160 | +0.043 | +0.031 |

### 解讀（誠實版）

1. **Method A 在 7 個指標上一致優於 vanilla**（macroF1 / 2c-F1 / acc / Clear-F1 / NC-P / NC-R / NC-F1
   全部 ≥），方向一致是正向訊號；NC 預測數 124→127 但 precision 與 recall 同時上升 ⇒ NC 決策邊界
   確實變好（輔助監督有重塑表徵）。
2. **但量級小且為單一 seed**：Method A 淨效果 +0.0038 macro / +0.0109 NC-F1，落在歷史上 NC 類
   （~101 列）單 seed ±0.025 的雜訊帶內，**尚不具結論性**。
3. **增益主要來自 NC recall（0.564→0.584），precision 只 +0.005**；P3 原本鎖定 precision，
   故結果部分對齊、未完全命中。
4. **Misleading 仍為 0**（val 僅 1 列，模型不預測 Misleading）——符合預期，非本方法目標。
5. 對 exp23 的總增益 +0.0160，其中 **+0.0122 來自重訓、+0.0038 才是輔助頭**。

## 多 seed 配對驗證（是否為浮動）

固定資料 / recipe / `aux_lambda=0.5`，**只變 seed**，逐 seed 算配對差 `Δ=multitask−vanilla`
（`run_seeds.sh` + `summarize_seeds.py`，`results/seeds/summary.json`）。

| 指標 | seed7 | seed13 | seed42 | seed123 | seed2024 | mean Δ | median Δ | std | sign test | vanilla 均值±std | multitask 均值±std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- |
| macroF1(3c) Δ | −0.0022 | −0.0023 | +0.0008 | −0.0066 | +0.0062 | **−0.0008** | −0.0022 | 0.0043 | **2/5** | 0.4706±0.0098 | 0.4697±0.0075 |
| NC-F1 Δ | −0.0152 | +0.0012 | +0.0059 | −0.0249 | +0.0243 | **−0.0017** | +0.0012 | 0.0171 | **3/5** | 0.5157±0.0208 | 0.5139±0.0138 |

### 結論：**是浮動，Method A（λ=0.5 輔助頭）不成立** ❌

- 跨 5 seed，配對 Δ 兩個指標 **mean 皆微負**（macroF1 −0.0008、NC-F1 −0.0017），multitask 平均值
  甚至略低於 vanilla。
- **sign test 2/5、3/5**（與丟銅板無異），配對 Δ 的 std（0.0043 / 0.0171）遠大於 mean ⇒
  與 0 無法區分。
- 先前單 seed 的 +0.0038 / +0.0109 是 **seed=42 的有利抽籤**（恰好是少數 Δ>0 的 seed），
  落在 per-seed noise band（vanilla macroF1 std 0.0098、NC-F1 std 0.0208）之內 ⇒ 不是真實效果。
- **唯一穩健的真實增益是「重訓 train_1000」**：vanilla 均值 macroF1 0.4706 vs exp23 0.4564
  （+0.014），與輔助頭無關。

### 下一步建議

- 否決「multitask 輔助頭 @ λ=0.5」這一具體實作。可能原因：large RoBERTa 已從主任務隱含學到
  數字/年份，regex 可解的輔助訊號對 CLS 表徵幾乎沒有額外塑形作用。
- 若仍要救 P3 的 NC：改試**實質不同**的變體 —— (a) 特徵向量直接 concat 進分類層（非輔助頭）、
  (b) LLM 可驗證性 rubric 對 borderline 列高精度覆寫（§P3 方法 B）。每個都必須以**多 seed 配對**
  為驗收門檻，不可再用單 seed 判定。
- 任何 promote 仍須官方 scorer 全 stage cascade 複評 + Claude review（CLAUDE.md Promotion Gate）。
