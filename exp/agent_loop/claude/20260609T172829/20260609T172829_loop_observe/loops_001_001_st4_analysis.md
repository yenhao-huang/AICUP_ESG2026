# Loop 001 ST4 分析報告 — Few-Shot Prompt Engineering (Codex)

## 0. TOC

範圍：loop 001、controller `exp/agent_loop/claude/20260609T172829`、Stage 4
(`verification_timeline`) Codex prompt 優化，baseline `stage4_balance_rule_v1.txt`。

狀態：val_public 選擇階段 **完成**（A1–A5 全部跑完）；A6 尚未執行；gate 階段（val_test）**尚未執行**；reflect **尚未撰寫**。

1. [Baseline vs Methods](#1-baseline-vs-methods)
2. [Plan / Dev / Exp / Reflect 對齊狀況](#2-plan--dev--exp--reflect-對齊狀況)
3. [Gate 通過 / 失敗總表](#3-gate-通過--失敗總表)
4. [下一步建議](#4-下一步建議)
5. [方法介紹](#5-方法介紹)

---

## 1. Baseline vs Methods

**Baseline artifact：** `configs/prompt/stage4/codex/stage4_balance_rule_v1.txt`（gpt-5.5）  
**Baseline val_test ST4 Macro-F1：0.5109**  
Per-class：already=0.454、within_2_years=0.222、between_2_and_5_years=0.575、more_than_5_years=0.603、N/A=0.701

> ⚠️ val_public 與 val_test 的 baseline 分布不完全相同；以下 Δ 為「val_public A4 vs val_test baseline」交叉比較，僅供方向性判斷，正式 gate 須在 val_test 重新驗證。

| Loop | Method family | Best variant | val_public Macro-F1 | Δ vs baseline | within_2y | already | Verdict |
|------|---------------|-------------|---------------------|--------------|-----------|---------|---------|
| 001 | Rules-only enhanced | **A4** boundary_rules_v4 | **0.6097** | **+0.0988** | 0.778 | 0.506 | pending（gate 未執行） |
| 001 | 8-shot direct | A2 few_shot_boundary_v2 | 0.6080 | +0.0971 | 0.737 | 0.481 | pending |
| 001 | 4-shot CoT | A3 few_shot_cot_v3 | 0.6069 | +0.0960 | **0.778** | 0.476 | pending |
| 001 | 4-shot direct | A1 few_shot_boundary_v1 | 0.5891 | +0.0782 | 0.737 | 0.441 | pending |
| 001 | 6-shot asym (within_2y ×3) | A5 few_shot_asym_v5 | 0.5860 | +0.0751 | 0.718 | 0.441 | pending |
| 001 | 8-shot CoT extended | A6 few_shot_cot_ext_v6 | — | — | — | — | 尚未執行 |

**全部 A1–A5 均大幅超越 baseline，within_2_years 為最大贏家（+0.496 ～ +0.556）。**

### val_public

| Variant | Macro-F1 | already | within_2y | b2_5y | mt5y | N/A |
|---------|----------|---------|-----------|-------|------|-----|
| A4 rules-only | **0.6097** | **0.506** | **0.778** | 0.552 | 0.567 | 0.645 |
| A2 8-shot | 0.6080 | 0.481 | 0.737 | **0.580** | **0.580** | **0.663** |
| A3 4-shot CoT | 0.6069 | 0.476 | **0.778** | 0.544 | 0.574 | **0.663** |
| A1 4-shot | 0.5891 | 0.441 | 0.737 | 0.547 | 0.557 | **0.663** |
| A5 6-shot asym | 0.5860 | 0.441 | 0.718 | 0.561 | 0.547 | **0.663** |
| Baseline val_test | 0.5109 | 0.454 | 0.222 | 0.575 | 0.603 | 0.701 |

---

## 2. Plan / Dev / Exp / Reflect 對齊狀況

| Loop | Planned method | Implemented method | 對齊 | 備註 |
|------|---------------|--------------------|------|------|
| 001 | 6 variants (V1–V6) few-shot + CoT + rules | A1–A5 完成，A6 尚未執行 | partial | A6（8-shot CoT ext）尚未跑 val_public；exp.md、reflect.md 均未撰寫 |

phase 完成度：

| Phase | 狀態 |
|-------|------|
| plans/001_agent_loop_plan.md | ✅ 完整 |
| dev/001_agent_loop_dev.md | ✅ 完整 |
| exp/A4/exp_A4.md | ✅ A4 單獨有 exp 紀錄 |
| exp/ 全域 exp.md | ❌ 未建立 |
| reflect/ | ❌ 未建立 |
| Gate runs (val_test) | ❌ 未執行 |

---

## 3. Gate 通過 / 失敗總表

Gate 正式評估須在 **val_test** 上跑 best-2 variants 後才能填寫。以下為 val_public 的預估狀態，所有數字均為 val_public，**非正式 gate 結果**。

| Loop | ST4 F1 > 0.5109 | within_2y > 0.222+0.05 | already > 0.454+0.03 | b2_5y 不退 > −0.03 | mt5y 不退 > −0.03 | N/A 不退 > −0.03 | Data-only | Verdict |
|------|-----------------|------------------------|----------------------|--------------------|-------------------|-----------------|-----------|---------|
| A4 | PASS (+0.099) | PASS (+0.556) | PASS (+0.052) | FAIL (−0.023)¹ | FAIL (−0.036)¹ | FAIL (−0.056)¹ | PASS | pending（val_test 待確認） |
| A2 | PASS (+0.097) | PASS (+0.515) | PASS (+0.027) | PASS (+0.005) | FAIL (−0.023)¹ | PASS (0) | PASS | pending |
| A3 | PASS (+0.096) | PASS (+0.556) | PASS (+0.022) | FAIL (−0.031)¹ | FAIL (−0.029)¹ | PASS (0) | PASS | pending |
| A1 | PASS (+0.078) | PASS (+0.515) | FAIL (−0.013) | FAIL (−0.028) | FAIL (−0.046) | PASS (0) | PASS | pending |
| A5 | PASS (+0.075) | PASS (+0.496) | FAIL (−0.013) | FAIL (−0.014) | FAIL (−0.056) | PASS (0) | PASS | pending |

¹ 這些 FAIL 是與 val_test baseline 交叉比較；val_public 的分布與 val_test 不同，可能在 val_test 上反轉為 PASS。

**現況：0/5 已通過完整 gate（均為 pending，gate phase 未執行）。**  
A4 和 A2 在 val_public 表現最強；val_test gate 執行後才能確認。

---

## 4. 下一步建議

（基於 plan gate 定義、A1–A5 val_public 結果，reflect 尚未撰寫）

1. **(優先)** 執行 **A4（boundary_rules_v4）** 的 val_test gate run。val_public Macro-F1 最高（0.6097），且 within_2_years 和 already 雙目標均改善；需確認 more_than_5_years/N/A 退步是否在 val_test 上成立。

2. **(優先)** 執行 **A2（few_shot_boundary_v2）** 的 val_test gate run。val_public 第二名（0.6080），且 between_2_and_5_years 和 more_than_5_years 退步幅度比 A4 更小；可能在 val_test 的非退步 gates 表現更穩。

3. **(可選)** 執行 **A3（few_shot_cot_v3）** 的 val_test gate run。within_2_years F1 與 A4 並列最高（0.778），且 N/A 未退；CoT 解析是否在 val_test 穩定仍需確認。

4. **(補完)** 撰寫 **001_agent_loop_exp.md**（彙整 A1–A5 val_public 完整結果）及執行 **A6** val_public run，確認是否值得加入 gate phase。

5. **(計畫外觀察)** A4 的規則強化（無 few-shot）居然優於 4-shot 及 6-shot 非對稱變體，暗示 gpt-5.5 在這個任務上對明確規則的響應比少量範例更強。val_test gate 若也成立，後續 loop 可以考慮規則 × CoT 混合路線（A6 邏輯：規則強化 + CoT 引導）。

6. **(計畫合規)** 完成 val_test gate 後，依 CLAUDE.md Promotion Gate 流程執行 Claude external review，寫 reflect 紀錄，才能更新 `docs/loops/agent_loop_state.json`。

---

## 5. 方法介紹

### Loop 001 — FSP (Few-Shot Prompt Engineering for ST4)

**計畫家族**：Calibrated few-shot Codex prompt with mined boundary-class examples + explicit year-band decision trees。計畫核心假設：baseline 的兩大失敗模式（within_2_years 極度低召回、already 低召回）可透過 (a) 從訓練資料挖取邊界案例作為 few-shot 範例、(b) 強化規則優先順序來修正。設計 6 個 variant（V1–V6）作為完整消融實驗。

**執行結果**（val_public，A1–A5 全部完成，A6 待執行）：

| Variant | 設計重點 | Macro-F1 | within_2y | already |
|---------|---------|----------|-----------|---------|
| A4 boundary_rules_v4 | 規則強化，無 few-shot，4 步優先順序 | **0.6097** | **0.778** | 0.506 |
| A2 few_shot_boundary_v2 | 8-shot（2 per class），直接回答 | 0.6080 | 0.737 | 0.481 |
| A3 few_shot_cot_v3 | 4-shot + CoT 推理步驟 | 0.6069 | **0.778** | 0.476 |
| A1 few_shot_boundary_v1 | 4-shot（1 per class），直接回答 | 0.5891 | 0.737 | 0.441 |
| A5 few_shot_asym_v5 | 6-shot（within_2y ×3 過採樣） | 0.5860 | 0.718 | 0.441 |

**關鍵發現**：

- within_2_years F1 從 0.222 → 0.718–0.778（全部 variant 均大幅提升）；模型對「明確 2025/2026 年份」規則的敏感性遠超預期。
- A4 規則強化（無 few-shot）略勝或持平有 few-shot 的版本，顯示 gpt-5.5 在此任務上「年份優先決策樹」比 few-shot 範例更有效；few-shot 的主要貢獻可能僅在 CoT 版（A3）。
- already 改善幅度有限（A4 最高 0.506 vs 0.454 baseline）；recall 從 0.358 → 0.497，但 precision 從 0.621 → 0.515，說明修正 within_2_years 偏差的同時，部分 already 被過度分類為 within_2_years（新錯誤模式）。
- between_2_and_5_years 和 more_than_5_years 在 val_public 上均有小幅退步（−0.023 ～ −0.036），但此比較混用 val_public/val_test 兩組分布，val_test gate 前不能確認。
- A5（within_2_years ×3 過採樣）反而不如 A2（2 per class 均衡），暗示過採樣稀有類別對 gpt-5.5 的效果不如預期。

**Verdict**：**pending**（val_test gate phase 尚未執行）。A4 和 A2 為 top-2 selection candidates，需在 val_test 上跑 gate run 後才能決定 accept / reject / defer。