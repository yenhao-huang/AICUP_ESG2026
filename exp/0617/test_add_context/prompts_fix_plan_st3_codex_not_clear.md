# Stage 3 Codex Prompt Fix Plan — 解決 Not Clear 答對率差

製表日期：2026-06-17
範圍：`exp/integrated_stage_predictions/0617/test_add_context/`
backend：**Codex（gpt-5.5）**，data-only runtime
目標類別：**Not Clear**（NC）

> 本計畫是 **Codex prompt 實驗**。Qwen 已在 RESULTS.md 用 lenient prompt 把 Clear-bias 修掉；
> 本計畫專做 Codex，且 Codex 的錯誤型態與 Qwen **相反**（見 §2），故修法不同。

---

## 1. 問題定義與 Codex 基線

Stage 3 = clarity；本 prompt 家族只輸出 `Clear` / `Not Clear`（N/A 由上游 ST2 gate 決定，
Misleading 不輸出）。主指標：GT `evidence_status` gate 後，GT `evidence_quality ∈
{Clear, Not Clear}` 子集（val_ctxhit.json，n2=397）的 **2-class Macro-F1**。

Codex 是專案目前 ST3 最強 backend（0616 README：4-class Macro-F1 0.5955 > BERT A1 0.59 >
Qwen 0.5435）。基線 artifact：
 分析 data/raw_data/vpesg4k_test_2000.json 看
```
0616/test_add_context/stage3/preds/st3_codex_parallel_codex_parallel_20260616_115412.csv
prompt = clear_notclear_with_context（scoped 嚴格規則），data-only gate
```

實測（val_ctxhit.json，n2=397）：

| 指標 | 值 |
|---|---|
| 2cls Macro-F1 | **0.6937** |
| Clear  F1 (P/R) | 0.891 (0.918 / 0.866) |
| **Not Clear F1 (P/R)** | **0.496 (0.438 / 0.574)** |

2-class 混淆（列=GT，欄=pred）：

```
            Clear   Not Clear   N/A
Clear        290        45       1     ← 45 過度翻成 NC（NC precision 殺手，Codex 主問題）
Not Clear     26        35       0     ← 26 漏判（次要）
```

---

## 2. 根因：Codex 的問題是 NC **precision**（與 Qwen 相反）

對照 Qwen lenient（NC P.509/R.443，**under-predict** NC）：**Codex 是 over-predict NC**
（P.438/R.574）。Codex 是高服從度的推論模型，會**忠實照 scoped prompt 的嚴格規則執行**，
於是 `步驟二 模糊動詞陷阱` + `步驟三 規則 3–6（時間框架/絕對宣稱/包裝願景/其他模糊→NC）`
把太多本該 Clear 的句子打成 NC。

### 2a. 45 個 Clear→NC（precision 損失，主缺口）— 都是「具體具名計畫/政策/機制」被誤殺

逐句歸納（括號為 val id）：

- **具名訓練/培育體系**：年度訓練體系含具體課程（11006）、培訓發展體系（11366）、
  新人體驗營（11562）、訓練四環（11889）、YOUNG Program 儲備人才計畫（11619）。
- **具名技術/制度導入**：5G 專網切片智慧醫療（11045）、內部碳定價 ICP（11513）、
  職業安全衛生體系含八大項（11615）、資訊安全政策+管理要點（11653）。
- **具名政策/原則條列**：五不原則（11810）、永續採購政策 1–4 點（11820）。
- **具名組織+具體運作**：水資源因應小組+應變計畫（11113）、節能委員會每月檢討（11239）、
  弊端防制處+年度稽核（11259）、中華電信基金會 2006 成立（11194）。
- **具名計畫+具體成果/獎項/年份**：OPENPOINT 2023 起獲獎（11382）、淨零座談會（11273）。

→ 失誤機制：這些句子常**同時**含「具體具名計畫」與「模糊動詞（持續/強化/致力…）」，
Codex 的模糊動詞陷阱與「其他模糊→NC」規則優先觸發，把具體性蓋掉。

### 2b. 26 個 NC→Clear（recall 損失，次要）— 「假錨點」型態（與 Qwen 漏判高度重疊）

具名錨點存在但 gold 判 NC：T1 治理/委員會存在（11025、11223、11288、11714）、
T2 例行流程/頻率（11058、11781）、T3 只有時間框架（11595「短中長期」、11026）、
T4 維持狀態/認證（11139 TIPS A 級、11297）、T5 數字歸屬他方/過去
（11330 出資 70 億助他方 2050、11124「2016 已完成、2024 無執行」）。

### 2c. 統一邊界（修法核心）

兩個方向同指一條更精準的邊界：
- **Clear** ＝ 句中對**本承諾**有「具體可指認、可追蹤的具名計畫/政策/制度/機制」
  **或**「具名年份+可量化目標/成果」——即使夾帶模糊動詞也算。
- **Not Clear** ＝ 只有「泛泛治理存在 / 例行頻率描述 / 純時間框架 / 維持既有狀態 /
  數字歸屬他方或過去」這類**假錨點**，無對本承諾的可驗證具體性。

Codex 目前兩邊都錯：把 2a 的具體計畫殺成 NC（模糊動詞陷阱過度），又把 2b 的假錨點
當成步驟一錨點而判 Clear。修法 = **重寫規則 + 邊界 few-shot**，Codex 對明確規則與範例反應強。

---

## 3. Novelty Check

- vs 本目錄 Qwen 工作（RESULTS.md「lenient 放寬錨點」）：那是 Qwen、修 under-predict；
  本計畫是 **Codex、修 over-predict（precision）**，方向相反，且用「除錯誤殺＋假錨點除外」
  雙向邊界，不是單純放寬。
- vs 0616 Codex scoped 基線：基線就是被本計畫要改的 prompt；本計畫提出規則重寫、
  邊界 few-shot、**NC 二次驗證 pass**、self-consistency 投票等 Codex 原生槓桿。
- 非重複：不是換錨點措辭，而是改決策結構 + 監督 + 後處理驗證。

---

## 4. 方法與實驗格（Codex，同一 loop 多變體）

固定：backend Codex gpt-5.5，data-only（`add_context: false`），透過
`run_pred_codex_parallel.sh`（`PROMPT_PATH=<變體>`）跑、`experiments/score_all.py` 評分。
新變體放 `prompts/codex/<name>.txt`。**Codex 無 logprobs**，故不用 token_confidence 翻轉，
改用 Codex 原生槓桿（reason、二次驗證、投票）。
**成本控管**：gpt-5.5 較貴 → 一律先 100-row（`val_ctxhit.100.json`）篩，過門檻才上 600。

### P1 — 規則重寫（boundary rule，純 prompt）
基於 `prompts/codex/vanilla.txt`（data-only），改三處：
1. **弱化模糊動詞陷阱**：明定「若句中同時含具名計畫/政策/制度/機制或具名年份+量化目標，
   即使夾帶模糊動詞，仍判 Clear」；模糊動詞只在「全句無任何具名具體物」時才致 NC。
2. **加假錨點除外條款**（T1–T5）：治理/委員會存在、例行頻率、純時間框架、維持狀態、
   數字歸屬他方/過去——不因此判 Clear。
3. 移除過度傾向 NC 的「步驟三規則 6 其他模糊→NC」兜底，改為「難判時看是否有 2c 的 Clear 具體物」。
- 變體 P1a（只做 1+2）／P1b（1+2+3 全套）。

### P2 — P1 + 邊界對照 few-shot
加 8–12 則成對範例：2a 型「具體計畫夾模糊動詞 → Clear」與 2b 型「假錨點 → Not Clear」對照。
- 變體 P2a（4 對）／P2b（6 對）。
- **防洩漏**：few-shot 取自 **train**（`data/raw_data/vpesg_4k_train_1000.json`）或改寫，
  禁止放入任何 val_ctxhit 句子（§6）。

### P3 — reason-then-label（Codex 原生）
Codex 已輸出 `evidence_quality_reason`。要求它先寫一句「本承諾的可驗證具體物＝？
（具名計畫/政策/年份量化目標；若只有假錨點則寫『僅假錨點』）」再給 label。把推理顯式化，
降低模糊動詞陷阱誤觸。

### P4 — NC 二次驗證 pass（取代 confidence-flip，直攻 45 FP）
對第一輪 pred=Not Clear 的列，跑第二次 Codex 專問：「這句是否含**具名的**計畫/政策/制度/
機制或具名年份+量化目標？有 → 改 Clear；只有治理存在/例行頻率/時間框架/維持狀態 → 維持
Not Clear」。只重跑 NC 子集，成本低、針對 precision。
- 新增 `core/analysis/postprocess_st3_codex_nc_verify.py`（讀第一輪 CSV，挑 NC 列重判、合併）。

### P5 — self-consistency 投票（變異數縮減）
對最佳 prompt 跑 k=3 次（temperature 略升或重排 few-shot），多數決。處理邊界搖擺；
成本 ×3，僅在 P1–P4 選出最佳後驗證是否再加分。

### P6 — context 再驗證（僅確認）
RESULTS 在 Qwen 證 context 有害；在新 Codex 規則下用 P1/P2 各跑一次 `ADD_CONTEXT=1`，
確認 data-only 仍最佳，不作主結果。

**最小執行格**：P1a、P1b、P2a、P2b、P3（100-row 篩）→ 最佳上 600 → 疊 P4 →（選配 P5）。

---

## 5. 指標、基線、接受/拒絕門檻

- 基線：Codex scoped，2cls Macro-F1 **0.6937**，NC F1 0.496（P.438/R.574），Clear F1 0.891。
- 主指標：2cls Macro-F1（Codex，data-only，GT `evidence_status` gate，n2=397）。
- 次指標：NC F1 / NC **precision** / NC recall / Clear F1；官方 4-class full-coverage F1
  （`core/analysis/score_st3_full_coverage.py`）；診斷量「45 FP 救回幾個」「26 漏判動幾個」
  「290 正確 Clear 被弄壞幾個」。
- **接受**（候選進 promotion gate）：
  - 2cls Macro-F1 ≥ **0.72**（≥ 基線 +0.026），且
  - NC F1 ≥ **0.55**，且 **NC precision ≥ 0.52**（基線 0.438，主修方向必須是精度），且
  - Clear F1 ≥ **0.90**（基線 0.891，不得倒退），且
  - 4-class full-coverage F1 不低於基線。
- **拒絕**：Macro-F1 < 0.70，或 NC precision 未升（仍 < 0.46），或 Clear F1 < 0.88。
- **暫緩**：Macro-F1 ∈ [0.70, 0.72) 且 NC precision 已升 → 方向對、幅度不足，留格內續調
  （few-shot 數 / P4 驗證強度），不另開新 loop。

---

## 6. Split 紀律 / 資料合規（data-only）

- runtime 輸入只有 `data`（承諾句）；prompt 與 few-shot 不含 `evidence_string` /
  `promise_string` / 抽取證據 / 任何標註或標籤欄位。
- 錯誤分析用 GT **僅供離線選 prompt**，不進 runtime 輸入。
- **無洩漏**：few-shot 一律取自 train 或改寫，禁用 val_ctxhit 句原文；主結果在 val 報分，
  P4/P5 在 val 上的選擇屬 prompt/threshold 選擇，promote 前須在獨立 split（或 train holdout）
  複驗 Macro-F1 / NC precision / Clear F1 不崩。
- Codex 的 `clear_notclear_with_context` / `add-context` / `add-image` / `add-evidence*`
  變體**非 data-only**（注入 context/圖/標註）——本計畫 runtime 只用 `prompts/codex/vanilla.txt`
  系的 data-only prompt，其餘僅作對照、不得 promote。

---

## 7. 失敗模式與放棄證據

- **修過頭**：放寬 Clear 後把 2b 假錨點也救成 Clear → NC recall 崩、26 漏判變多、NC F1 反降。
- **二次驗證盲救**：P4 把太多 NC 翻回 Clear（含真 NC）→ NC recall 崩。以「翻回的列中 GT=Clear
  佔比」監控，須 > 0.6 才算有效翻轉。
- **few-shot 洩漏假象**：誤用 val 句 → 換 train-only few-shot 後分數崩即判定。
- **Codex 不穩/成本**：gpt-5.5 變異或逾時 → 用 P5 投票穩定；若成本不可控則停在 P1+P4。
- **放棄判準**：P1–P5 無法同時達 Macro-F1 ≥ 0.71 且 NC precision ≥ 0.50 且 Clear F1 ≥ 0.88，
  判定 Codex 純 prompt 已達 Clear/NC 邊界天花板；結論導向非 prompt 槓桿（Codex×BERT A1 集成、
  或 NC 專用判別器），本 prompt loop 收斂。

---

## 8. score_first_promote gate（開發前定義）

- baseline artifact：`0616/.../st3_codex_parallel_codex_parallel_20260616_115412.csv`
  （2cls 0.6937）；對照 `docs/loops/agent_loop_state.json` 的 ST3 promoted baseline。
- candidate artifact：`preds/st3_codex_<best_variant>.csv`（data-only）。
- 主門檻：2cls Macro-F1 ≥ 0.72；NC F1 ≥ 0.55；**NC precision ≥ 0.52**；Clear F1 ≥ 0.90。
- 未變動 stage（ST1/ST2/ST4）容差：0（只改 ST3 codex prompt，不動上下游 inherited artifacts）。
- data-only 合規：promoted runtime 僅用 `data`，few-shot 無禁用欄位、無 val 洩漏。
- 實驗紀錄須報：命令、artifact 路徑、weighted score、各 stage Macro-F1、對基線 delta、
  §5 診斷量；reflect 逐項驗證本 gate，含 Claude review，方可更新 `docs/methods.md` 與
  `docs/loops/agent_loop_state.json`。

---

## 9. 執行步驟（摘要）

```bash
EXP=exp/integrated_stage_predictions/0617/test_add_context
# 1) 寫 prompts/codex/clear_boundary.txt (P1)、clear_boundary_fewshot.txt (P2)、
#    clear_reason.txt (P3)；few-shot 取自 train。
# 2) 100-row 篩（data-only，Codex）
for p in clear_boundary clear_boundary_fewshot clear_reason; do
  DATA=$EXP/data/val_ctxhit.100.json ADD_CONTEXT=0 \
  PROMPT_PATH=$EXP/prompts/codex/$p.txt WORKERS=4 \
  bash $EXP/run_pred_codex_parallel.sh
done
# 3) 評分（per-class F1 + 混淆），挑最佳上 600-row
.venv/bin/python $EXP/experiments/score_all.py --benchmark $EXP/data/val_ctxhit.json \
  --preds-dir $EXP/preds --pattern 'st3_codex_*.csv' --output $EXP/preds/codex_fix_scores.json
# 4) 最佳者疊 P4 NC 二次驗證
.venv/bin/python core/analysis/postprocess_st3_codex_nc_verify.py \
  --pred $EXP/preds/st3_codex_<best>.csv --data $EXP/data/val_ctxhit.json \
  --prompt $EXP/prompts/codex/nc_verify.txt --output $EXP/preds/st3_codex_<best>_ncverify.csv
# 5) （選配）P5 self-consistency k=3 多數決
```

主結果與診斷寫入 `experiments/RESULTS.md` 新節「Codex prompt fix — Not-Clear precision」，
promote 前依 §6/§8 在獨立 split 複驗。
