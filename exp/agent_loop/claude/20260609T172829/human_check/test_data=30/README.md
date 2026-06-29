# ST4 Prompt Human Check — 20260609T172829

## 目的

對 6 個 ST4 Codex prompt variants 進行人工核查，評估在兩個小規模樣本集上的表現。

## 資料

| 檔案 | 描述 | 筆數 | 來源 |
|---|---|---|---|
| `data/test_preserved.json` | ST4 pipeline 保留的 test 樣本（ST1=Yes） | 30 | `benchmarks/test.json` |
| `data/stage4_predictions.csv` | 原始 pipeline 預測結果 | 30 | loop 001 A4 val_public run |
| `data/within_2y_extra.json` | 額外蒐集的 within_2_years 樣本 | 10 | `vpesg_4k_train_1000.json` |

## Prompt Variants

| Variant | Prompt 檔案 |
|---|---|
| `boundary_rules_v4` | `configs/prompt/stage4/codex/boundary_rules_v4.txt` |
| `A4_rules_only` | `configs/prompt/stage4/codex/boundary_rules_v4.txt`（同上） |
| `A3_cot` | `configs/prompt/stage4/codex/few_shot_cot_v3.txt` |
| `A2_8shot` | `configs/prompt/stage4/codex/few_shot_boundary_v2.txt` |
| `stage4_balance_rule_v1` | `configs/prompt/stage4/codex/stage4_balance_rule_v1.txt` |
| `vanilla_old_anchor2024_system` | `configs/prompt/stage4/codex/vanilla_old_anchor2024_system.txt` |

Run scripts: `run_variants.sh`（Set A）、`run_variants_w2y.sh`（Set B）

## 評估結果

### Set A：30 筆 human-check（test set）

| Variant | Macro-F1 | already | b2_5y | mt5y |
|---|---|---|---|---|
| **A3_cot** | **0.7360** | 0.778 | **0.588** | **0.842** |
| A2_8shot | 0.7012 | 0.706 | 0.556 | **0.842** |
| stage4_balance_rule_v1 | 0.6343 | 0.625 | 0.500 | 0.778 |
| A4_rules_only | 0.6053 | **0.783** | 0.500 | 0.533 |
| boundary_rules_v4 | 0.5771 | 0.727 | 0.471 | 0.533 |
| vanilla_old_anchor2024_system | 0.5486 | 0.688 | 0.333 | 0.625 |
| original (pipeline) | 0.5417 | 0.667 | 0.333 | 0.625 |

> `within_2_years` 正例為零，不列入 Set A Macro-F1 計算。

### Set B：10 筆 within_2_years（train set）

所有 prompt 均 **8/10**，差異為零。

永遠答錯的 hard case：
- **10344**：所有 prompt 預測 `already`，GT = `within_2_years`
- **10526**：所有 prompt 預測 `between_2_and_5_years`，GT = `within_2_years`

## 結論

- **A3_cot（few_shot_cot_v3）最強**，Set A Macro-F1 0.7360，`mt5y` 與 `b2_5y` 均最高
- **A2_8shot（few_shot_boundary_v2）次之**，0.7012，`mt5y` 與 A3 並列
- Rules-only variants（A4、boundary_rules_v4）在 `mt5y` 上明顯弱於 few-shot variants
- `within_2_years` recall 對所有 prompt 無差異，bottleneck 在 2 筆 hard case 本身
- Set A 主要錯誤模式：`more_than_5_years` 和 `between_2_and_5_years` 被誤判為 `already`
