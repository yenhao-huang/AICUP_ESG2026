# submit_9 分析

- pipeline dir : `exp/integrated_stage_predictions/0617/submit_9`
- DATA         : `data/raw_data/vpesg4k_test_2000.json` (blind test 2000，無 label)
- 結構         : ST1 soft-vote(5) + Gemma fallback / ST2 soft-vote(5) + Gemma fallback + ST1 gate /
                 ST3 單一 multitask BERT (submit_6 `w0_2_0_3_0_5`) + ST1+ST2 gate /
                 ST4 codex 預測 + ST1 gate + 14 列 na-fix patch

> blind set 無 label，故省略 `success by class` / `kept(不辨)` 子區塊。

```
============================================================
ST1: promise_status
============================================================

--- class_distribution (final, submission) ---
  No                                    374  (18.7%)
  Yes                                  1626  (81.3%)
  TOTAL                               2000

--- score_distribution (soft-vote raw, N=all 2000) ---
  <0.50                                   0  (0.0%)
  0.50-0.60                             130  (6.5%)
  0.60-0.70                             149  (7.4%)
  0.70-0.80                             201  (10.1%)
  0.80-0.90                             398  (19.9%)
  >=0.90                               1122  (56.1%)
  median confidence                   0.917575

--- fallback_change_rate (Gemma, conf<0.60) ---
  total rows                          2000
  kept BERT (conf>=0.60)              1870
  Gemma fallback rows                 130
  changed by Gemma                    59   (45.4% of fallback)
  overall change rate                 2.95%
    Yes→No                            46
    No→Yes                            13

============================================================
ST2: evidence_status
============================================================

--- class_distribution (final, gated) ---
  Yes                                  1376  (84.6% of active)
  No                                    250  (15.4% of active)
  N/A (ST1=No)                          374
  TOTAL                               2000

--- score_distribution (soft-vote raw, gated active N=1626) ---
  <0.50                                   0  (0.0%)
  0.50-0.60                              45  (2.8%)
  0.60-0.70                              49  (3.0%)
  0.70-0.80                              47  (2.9%)
  0.80-0.90                              60  (3.7%)
  >=0.90                               1425  (87.6%)
  median confidence                   0.999908

--- fallback_change_rate (Gemma local --no-rag, tau<0.70) ---
  total rows                          2000
  kept BERT (conf>=0.70)              1882
  escalated to Gemma                  118
  changed by Gemma                    62   (52.5% of escalated)
  overall change rate                 3.10%
    Yes→No                            34
    No→Yes                            28

============================================================
ST3: evidence_quality
============================================================

--- class_distribution (final, gated) ---
  Clear                                1129  (82.0% of active)
  Not Clear                             247  (18.0% of active)
  Misleading                              0  (0.0% of active)
  N/A (ST1=No 或 ST2=No)               624
  TOTAL                               2000

--- score_distribution (BERT, active w/ score, N=1376) ---
  <0.50                                   2  (0.1%)
  0.50-0.60                             106  (7.7%)
  0.60-0.70                              90  (6.5%)
  0.70-0.80                             127  (9.2%)
  0.80-0.90                             233  (16.9%)
  >=0.90                                818  (59.4%)
  median confidence                   0.938610

  (no fallback — ST3 為單一 multitask BERT)

============================================================
ST4: verification_timeline
============================================================

--- class_distribution (final, submission) ---
  already                               690  (42.4% of active)
  between_2_and_5_years                 627  (38.6% of active)
  more_than_5_years                     228  (14.0% of active)
  within_2_years                         81  (5.0% of active)
  N/A (ST1=No)                          374
  TOTAL                               2000

  (no score — ST4 = codex 預測 (fully-fixed) + ST1 gate；含 14 列 na-fix patch，本次 patch 10/14)
  source = stage4/tmp/stage4_codex_predictions_fixed.csv
           (== 0615/fix_stage4_predictions/stage4_codex_predictions_fixed.csv，raw 無 N/A)
```

## 觀察與檢查

- **Cascade 一致性正確**：ST1=No 共 374 列 → ST2 N/A = 374 ✓；
  ST3 N/A = 624 = 374 (ST1=No) + 250 (ST2=No) ✓。ST4 N/A = 375，
  比 ST1 gate 多 1 列（codex 原始預測該列為空/N/A，gate 後仍 N/A），屬正常。
- **ST1 confidence 偏低不是異常**：median=0.9176、`>=0.90` 僅 56.1%，遠低於單模型
  baseline (~0.9998)。原因是 5-member soft-vote 把機率平均、峰值被攤平，屬 ensemble
  的預期行為，非模型退化。fallback 門檻 0.60 因此只攔到最底部 6.5%（130 列）。
- **ST2 仍呈單模型式尖銳分布**（gated active median=0.999908、`>=0.90`=87.6%），與 ST1 soft-vote
  的平滑分布形成對比；tau=0.70 攔到 118 列 (5.9%)。
- **fallback 改動量溫和**：ST1 改 59 列 (2.95%)、ST2 改 62 列 (3.10%)，皆在歷史
  submit 的 2–3% 區間內，無暴衝。ST1 fallback 明顯偏向 Yes→No (46:13)，會壓低整體
  Yes 率；ST2 雙向接近平衡 (34:28)。
- **ST3 Misleading = 0**：multitask BERT 在此 test set 完全沒輸出 Misleading，與既有
  ST3 模型行為一致（Misleading 為極稀有類，模型傾向不預測）。
- **資料合規**：兩處 Gemma fallback 皆只讀 `data`（prompt 用 `data`、ST2 `--no-rag`），
  adapter `gemma4_st12_mix` 為 data-only st1+st2 生成式 adapter；符合 data-only 規範。

## verdict

confidence 分布正常，cascade gating 全部自洽，fallback 改動量在歷史區間內。
唯一需留意：ST1 soft-vote 的 confidence 整體偏低（ensemble 預期效果），且 ST1
fallback 強烈偏 Yes→No，會使最終 Yes 率（81.3%）較單模型 submit 略低——若驗證分數
偏離預期，優先檢查此處方向偏移是否過度。
