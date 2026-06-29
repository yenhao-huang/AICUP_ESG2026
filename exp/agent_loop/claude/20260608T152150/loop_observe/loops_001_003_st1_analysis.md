# Agent-Loop `20260608T152150` — ST1 Promise Identification loops 001–003 觀察

## 0. TOC

範圍：loops 001–003、controller `exp/agent_loop/claude/20260608T152150`、
Stage 1 (`promise_status` Yes/No, Macro-F1) BERT classifier + 選用 LLM-RAG
fallback；primary 指標為 `val_test.json` 盲測 ST1 Macro-F1。

1. [Baseline vs Methods](#1-baseline-vs-methods)
2. [Plan / Dev / Exp / Reflect 對齊狀況](#2-plan--dev--exp--reflect-對齊狀況)
3. [Gate 通過 / 失敗總表](#3-gate-通過--失敗總表)
4. [下一步建議](#4-下一步建議)
5. [方法介紹](#5-方法介紹)
6. [exp36 人工驗證 (val.json)](#6-exp36-人工驗證-valjson)

## 1. Baseline vs Methods

Baseline 工件：`models/exp4_optimize2_highconf_yes_balanced_no_large`
(`hfl/chinese-roberta-wwm-ext-large`)。
BASE_GATE (val_test ST1 Macro-F1) = **0.802383**；val_public = 0.787283；
integrated weighted (val_test, 0.2/0.3/0.35/0.15) = 0.591591。

| Loop | Method family | Best variant | Primary metric (val_test ST1 F1) | Δ vs baseline | val_public | Verdict |
|---|---|---|---|---|---|---|
| 001 | Synthetic-data augmentation (BERT) | `a3_b1` (data+promise 0.5×) | 0.823377 | **+0.020994** | 0.803069 | **defer** |
| 002 | Confidence-routed LLM-RAG fallback | `T0.80_K5_strictneg_bertctx` | 0.811566 | +0.009183 | 0.817436 | **defer** |
| 003 | Recipe promotion (real-only BERT) | `a0_real_only` | 0.810977 | +0.008594 | 0.803069 | **accept** |

註：loop 001 exp 自身 bold 的最佳 variant 是 `a3_b1`（盲測 +0.021），但 reflect
判 defer——因為 a3_b1 與 `a0_real_only` 在選擇集 val_public 上**完全平手**
(0.803069)，挑 a3_b1 等同 selection-on-gate，且合成增益在噪聲內。最終被接受並
promote 的是 loop 003 的保守 `a0_real_only`（盲測 +0.0086，雖最小但最可信）。

Reference comparators（loops 引用、用於對照）：

| 名稱 | val_public | val_test | 說明 |
|---|---|---|---|
| exp4 baseline | 0.787283 | 0.802383 | 舊 ST1 default checkpoint（即 BASE_GATE）|
| loop002 a3_b1 base（base-invariance 診斷）| 0.808480 | — | fallback 在第二 base 上仍 +0.0054，方向 base-invariant |

## 2. Plan / Dev / Exp / Reflect 對齊狀況

| Loop | Planned method | Implemented method | 對齊 | 備註 |
|---|---|---|---|---|
| 001 | 合成增益 grid（source A0–A4 × mix B1–B3 + manual_ce 消融）| 13 arms 全跑，manual_ce 消融未達 | `partial` | manual_ce 消融未跑；dev split test.json 200/200 汙染→改用 val_public |
| 002 | LLM-RAG fallback grid（T×K×P×C + margin router，≈15 configs）| 19 configs；T=0.70 後剪 K/P/C 組合 | `aligned` | K 與 mode-context 經證實無效後剪枝，strict_negative 主導 |
| 003 | —（無 plan/dev 階段）| user-directed a0 promotion：補跑 external review + integrated 複檢 | `incomplete` | 僅 exp+reflect，無 plans/、無 dev/；為 loop001 defer 的 a0 補正式 gate |

額外旗標：

- loop 001／002 兩次 reflect 都因**缺 mandatory external Claude review** 被迫
  `defer`，方法數字本身皆過 gate（非數字問題，是流程缺件）。
- loop 003 的 external review（`external_claude_review_a0_promote.md`，
  20260609T011056Z）回填了 loop 001 的 review
  (`loops001/reflect/001_claude_review_a0_promotion.md`)，正是這份 review 解除了
  loop 001 的 defer，使 a0 得以 promote。
- 無 stale reflect；各 loop 的 exp 與 reflect 數字一致（reflect 皆從 raw eval
  JSON 重新驗證）。

## 3. Gate 通過 / 失敗總表

Stage 1 任務，欄位依各 loop 共用的 `score_first_promote` gate 改寫
（primary 盲測 + 選擇集紀律 + 結構性 gate + 流程 gate）。

| Loop | val_test > BASE_GATE | 選擇紀律 (val_public only) | Data-only runtime | 下游 ST2–4 non-regress | Integrated weighted ≥ base | External Claude review stored |
|---|---|---|---|---|---|---|
| 001 | PASS (a3_b1 +0.0210 / a0 +0.0086) | PASS | PASS | — (未測 integrated) | — (未測) | **FAIL (缺件)** |
| 002 | PASS (+0.009183) | PASS | PASS | PASS (ST1-only, δ=0) | — (未測) | **FAIL (缺件)** |
| 003 | PASS (+0.008594) | PASS | PASS | PASS (各 stage 皆升、無 regress) | PASS (0.591591→0.610112, +0.018521) | PASS (conditional support-promote, 條件全滿足) |

總結：1 / 3 loops fully pass（loop 003 accept）；2 loops 落在 defer zone
（loop 001、002，唯一硬傷皆為缺 external review，方法數字過 gate）；0 rejected。

## 4. 下一步建議

來源：最新完成 reflect = loop 003（item 8）＋ loop 002 reflect 的具體選項
（loop 003 無 plan，故 next-family 設計沿用 loop 002 advice）。

1. **(mandatory) 改做 BERT-only 校準 / per-class 門檻，當作全新方法家族。**
   針對 a0 偏弱的 `No`-F1 (≈0.69) 做 temperature / isotonic / beta 校準＋
   per-class 決策門檻與 abstain band sweep，零 LLM 成本，且是直接提升 base 自身
   盲測分數最乾淨的路徑。
2. **(mandatory) 未來所有 lift 一律以「相對 a0 base 的盲測增益」回報**，不可只報
   贏 exp4——loop 002 的教訓是 exp4 gap 幾乎全來自 a0 recipe，誠實門檻是盲測上
   贏 a0 超過 ~1 row。
3. **(mandatory) loop 002 的 LLM-RAG fallback 不得搭 loop 003 的順風車 promote**，
   必須走自己的 gate；其對 a0 的盲測邊際增益僅 +0.000589（~1 row），證據太薄。
4. **(fallback) 若要救 LLM 路線**，先讓 flip 可信：只在 LLM 與第二個 data-only
   信號（如 top-k 鄰居 retrieval-label 多數決）一致時才接受翻轉，把
   flip-correctness 從薄弱的 0.562 拉高，再重測 routing；仍贏不過 a0 盲測就放棄。
5. **(fallback) 加第二個 data-only retrieval index**（qwen3-embedding-8b 端點當時
   掛掉）：只在第 1/2 項框架內 ablate embedding-model × distance × weighting，
   不可當成 loop 002 的獨立重跑。

## 5. 方法介紹

### Loop 001 — SDA (Synthetic-Data Augmentation for ST1 BERT)

- **計畫家族**：對 `hfl/chinese-roberta-wwm-ext-large` ST1 分類器做離線合成資料
  增益；2 維 grid＝合成來源 A0–A4（real-only／data-only／promise-derived／
  data+promise／pdf hard-No）× 混入權重 B1–B3 (0.5×/1×/2×)，不引入 LLM
  fallback。假設：合成資料能勝過純 real 重訓。
- **執行結果**：選擇集 val_public 上 `a0_real_only` = `a3_b1` = **0.803069 完全
  平手**，皆 +0.0158 勝 exp4；所有合成重混 arm（A1/A2/A4 的 B2/B3）皆低於
  real-only。盲測：a3_b1 0.823377 (+0.0210)、a0 0.810977 (+0.0086)。manual_ce
  消融未跑。
- **關鍵發現**：
  - 指定 dev split `test.json` 與訓練集 **200/200 全洩漏**，dev 分數 0.96–0.99
    為記憶化、無效；選擇改用乾淨的 `val_public`（與 train、val_test 皆 0 重疊）。
  - 合成增益**論點被自身結果證偽**：real-only 平手最佳合成 arm，重混反而傷害。
  - 對舊 baseline 的增益來自**canonical-data 訓練 recipe**，而非合成。
  - a3_b1 對 a0 的盲測領先 (+0.0124) 在 ~135 個負例上約等於 1–2 句翻轉，疑為噪聲。
- **Verdict**：**defer**——硬傷是缺 mandatory external Claude review；且兩 arm 在
  選擇集平手時挑 a3_b1 等同 selection-on-gate。actionable 結論：a0 是更安全的
  promote 候選（留給 loop 003）。

### Loop 002 — CRF (Confidence-Routed LLM-RAG Fallback)

- **計畫家族**：凍結 loop001 a0 BERT 當 base，依 softmax 信心把**最低信心**列
  escalate 給 LLM（`http://192.168.1.79:3134`），prompt 以 data-only top-k 訓練
  例 few-shot。grid：門檻 T×retrieval K×prompt P×mode-context C＋margin router。
  純 runtime routing，不重訓、不加合成。
- **執行結果**：最佳 `T0.80_K5_strictneg_bertctx`——val_public **0.817436**
  (+0.0144 over a0)、盲測 val_test **0.811566** (+0.009183 over BASE_GATE)；
  31 列 escalate，flip 16、toward-GT 9、flip-correctness 0.562（net +2）。
  19 configs 全 `codex_errors=0`。
- **關鍵發現**：
  - 對 a0 base 的**盲測**增益僅 **+0.000589（~1 row）**，val_public 的 +0.0144
    幾乎不轉移到盲測——疑為對選擇集特性的過配。
  - prompt `strict_negative` > certificate > default；K 與 mode-context 在
    T=0.70 幾乎無效（同 16 列、flip 不變）。
  - 響應曲線倒 U，峰在 T=0.80 後 T=0.90/0.95 退化（LLM 翻掉本來正確的高信心列）。
  - base-invariance 診斷：同 config 在 a3_b1 base 仍 +0.0054，方向真實但小。
- **Verdict**：**defer**——(1) 缺 mandatory external Claude review（非可豁免硬
  條件）；(2) 即使有 review，LLM 層對 a0 的盲測邊際價值在噪聲內，真正過 gate 的
  其實是 a0 recipe。家族未耗盡，不到 reject。

### Loop 003 — RP (Recipe Promotion of a0 real-only BERT)

- **計畫家族**：無 plan/dev 階段；user-directed，將 loop001 `a0_real_only`
  (`models/loop001_st1_a0_real_only/best_st1.pt`，1000 real rows、無合成、runtime
  data-only) 正式 promote 為新 ST1 baseline，補跑 CLAUDE.md mandatory external
  Claude review 與 integrated weighted 複檢。
- **執行結果**：標準 ST1 val_test 0.810977 (+0.008594 vs BASE_GATE)、val_public
  0.803069 (+0.0158)；**integrated cascade weighted 0.591591→0.610112
  (+0.018521)**，且只換 ST1 checkpoint 就讓 ST2 (+0.0246)/ST3 (+0.0202)/
  ST4 (+0.0157) 全部上升、無 stage regress。
- **關鍵發現**：
  - 更準的 ST1 gate 降低 cascade 對下游的懲罰，故下游 integrated 分數一起改善
    （external review 提醒：「ST2–4 Δ=0」只對 *artifacts* 成立，*integrated*
    分數會動，須實測一次——本 loop 已實測）。
  - external review 判 **conditional support-promote**，四條件（存 review／
    integrated 複檢 ≥ baseline／同 session 更新 methods.md+state／test.json 維持
    retired）全部滿足。
  - 選 a0 而非盲測更高的 a3_b1，是刻意取 synthesis-free、不剝削 gate 的保守選擇。
- **Verdict**：**accept**——每項 promotion gate 通過，integrated weighted 嚴格
  改善且無 stage regress；已更新 `docs/methods.md`、`docs/feature_list.json`、
  `agent_loop_state.json` 與 workspace `state.json`（`promoted_st1` 指向 a0）。

### 橫向觀察

三個 loop 形成一條完整的「誠實 promote」軌跡：loop 001 找到 recipe 真因並揭露
`test.json` 洩漏、loop 002 證明 LLM 層盲測邊際價值僅 ~1 row、loop 003 才把最保守
但最可信的 a0 正式 promote。**唯一反覆絆住前兩 loop 的不是分數而是流程**——缺
mandatory external Claude review 兩度逼出 defer；一旦補上 review（loop 003），
同一個 a0 立刻過關。下一步的真正戰場是 a0 仍偏弱的 `No`-F1 (≈0.69)，且任何新
方法都須以「盲測贏 a0」為門檻，而非贏已被 a0 吸收掉的 exp4。

## 6. exp36 人工驗證 (val.json)

人工（非 agent_loop）重跑驗證：取 loop001 在選擇集 val_public 上**最佳的兩個
合成 mix**，從頭重訓 ST1 BERT，再接 loop002 的 BERT+LLM-RAG 預測器評估
`data/benchmarks/val.json`（n=1000，Yes 813 / No 187）。工件在
`exp/exp36/loop1/`。訓練比照 loop001 recipe（large roberta、5 epoch、
`--val-ratio 0.2`、`--batch-size 4`、weighted CE，GPU1）。

選用的兩個資料（canonical real+synth mix，A3 = data + promise_string 離線合成）：

| rank | arm | 合成倍率 | rows (Yes/No) | loop001 val_public |
|---|---|---|---|---|
| top1 | `a3_b1` | 0.5× | 1500 (1314/186) | 0.803069 |
| top2 | `a3_b3` | 2.0× | 3000 (2814/186) | 0.795500 |

### ST1 Macro-F1 on val.json

| arm | BERT-only | BERT + LLM-RAG | RAG Δ | vs exp4 0.7950 (BERT-only) |
|---|---|---|---|---|
| `a3_b1` (top1) | **0.813525** | **0.817077** | +0.003552 | +0.018525 |
| `a3_b3` (top2) | 0.798093 | 0.804028 | +0.005935 | +0.003093 |

per-class（BERT+RAG）：a3_b1 Yes 0.9380 / No 0.6962；a3_b3 Yes 0.9353 / No 0.6727。
RAG 路由（loop002 選定 config：T=0.80、top-k 5、`strict_negative`、bert-ctx、
data-only、ModernBert index、HTTP 端點）—— a3_b1 escalate 21／changed 14、
a3_b3 escalate 19／changed 11，兩者 `codex_errors=0`。

### 關鍵發現

- **輕量合成 (0.5×) 明顯勝重量 (2.0×)**：BERT-only 0.8135 vs 0.7981（−0.0154），
  完整重現 loop001「重混合成傷 ST1」的結論，且在獨立的 val.json 上一致。
- **LLM-RAG fallback 在 val.json 上兩 arm 皆小幅正向**（+0.0036／+0.0059），
  比 loop002 在 val_test 的 ~1 row 稍大，但仍只動 ~20/1000 低信心列——信心路由的
  邊際價值依舊「小而正」，與 loop002 的倒 U 結論相符。
- 兩 arm BERT-only 都過 exp4 舊 val 基線（0.7950）：a3_b1 從容 +0.0185、a3_b3
  僅 +0.0031。

### Caveat

`val.json` 是**舊 split**（§state known_issues 與 loop001 已標記，非乾淨的
val_public/val_test gate），故本節是「依指定 split 的人工重現 sanity check」，
**非 promotion gate**；數字不可與 §1/§3 的 BASE_GATE 直接並列當作晉升依據。
