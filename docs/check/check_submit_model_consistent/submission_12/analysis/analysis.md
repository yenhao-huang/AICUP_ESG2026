# submission_12 analysis

- Data: `data/raw_data/vpesg4k_test_2000.json` (blind, 2000 rows, no labels → success-by-class omitted)
- Pipeline: ST1/ST2/ST4 沿用 submit_9（Gemma low-conf fallback）；**ST3 = multitask BERT → codex low-conf fallback (CONF=0.60)**，codex 來源為 `add_context_test2000_codex.csv`（offset_hit 上下文，全 2000 列皆有實際標籤，無 na-fix）。
- Final merged: `submission.csv`（2000 rows；下列 class_distribution 以 submission.csv 為準）

============================================================
ST1: promise_status
============================================================

--- class_distribution (final) ---
  No                                    374  (18.7%)
  Yes                                  1626  (81.3%)
  TOTAL                                2000

--- score_distribution (BERT raw, N=1870 scored; 130 Gemma 列無 BERT score) ---
  <0.50                                   0  (0.0%)
  0.50-0.60                               0  (0.0%)
  0.60-0.70                             149  (8.0%)
  0.70-0.80                             201  (10.7%)
  0.80-0.90                             398  (21.3%)
  >=0.90                               1122  (60.0%)
  median confidence                   0.924454

--- fallback_change_rate (Gemma, conf_threshold=0.60) ---
  total rows                           2000
  BERT-only (conf>=0.60)               1870
  Gemma fallback rows                   130
  changed by Gemma                       59  (45.4% of fallback)
  overall change rate                  2.95%
    Yes→No                               46
    No→Yes                               13

============================================================
ST2: evidence_status
============================================================

--- class_distribution (final) ---
  No                                    250  (15.4% of active)
  Yes                                  1376  (84.6% of active)
  N/A (ST1=No)                          374
  TOTAL                                2000

--- score_distribution (soft-vote, active rows only, N=1626) ---
  <0.50                                   0  (0.0%)
  0.50-0.60                              45  (2.8%)
  0.60-0.70                              49  (3.0%)
  0.70-0.80                              47  (2.9%)
  0.80-0.90                              60  (3.7%)
  >=0.90                               1425  (87.6%)
  median confidence                   0.999908

--- fallback_change_rate (Gemma, tau=0.70) ---
  total rows                           2000
  kept soft-vote (conf>=tau or N/A)    1882
  escalated to Gemma                    118
  changed by Gemma                       62  (excl. later N/A gate ≈47；Yes→No 19 / No→Yes 28)

============================================================
ST3: evidence_quality   ← submission_12 的改動點
============================================================

--- class_distribution (final) ---
  Clear                                1141  (82.9% of active)
  Not Clear                             235  (17.1% of active)
  Misleading                              0  (0.0% of active)
  N/A                                   624
  TOTAL                                2000

--- score_distribution (multitask BERT, ST2=Yes active 列, N=1376) ---
  <0.50                                   2  (0.1%)
  0.50-0.60                             106  (7.7%)
  0.60-0.70                              90  (6.5%)
  0.70-0.80                             127  (9.2%)
  0.80-0.90                             233  (16.9%)
  >=0.90                                818  (59.4%)
  median confidence                   0.938610
  注: 此處用「ST2 gate 通過」的 active 列做分母（門檻判斷的有效集合），
      非全 2000；codex_fallback 取代的列 reason 不帶 score。

--- fallback_change_rate (Codex add_context, CONF=0.60, gated/生效) ---
  ST3 active (ST2=Yes)                 1376
  Codex fallback rows (ungated)         152
    └ active (ST2=Yes, 對最終生效)      108
    └ 被 ST1/ST2 gate 成 N/A (無效浪費)   44
  changed by Codex (active only)         52   (4.8% of active fallback; 3.78% of active 1376)
    Not Clear→Clear                      32
    Clear→Not Clear                      20
  source dist (final): bert_multitask=1848, codex_fallback=152
  注1: ungated 看到的 152 fallback / 72 changed 含 44 列會被 gate 成 N/A、
       其中 20 列的 codex 變更不影響最終 submission；gated 後實際生效是 108 列 / 52 變更。
  注2: add_context codex 全 2000 列皆有實際標籤（無 N/A），故不需 submit_11 的 na-fix。

============================================================
ST4: verification_timeline
============================================================

--- class_distribution (final, submission.csv) ---
  already                               690  (43.5% of active)
  between_2_and_5_years                 627  (39.6% of active)
  more_than_5_years                     228  (14.4% of active)
  within_2_years                         81  (5.1% of active)
  N/A                                   374
  TOTAL                                2000
  (no score — codex/rule-based；沿用 submit_9。
   注: stage4/codex_gated.csv 原 already=699 / N/A=364，含 10 列 ST1=No 卻帶具體
   timeline 的 na_fix_14_codex 髒列；merge_pipeline 的 cascade 防呆把它們打回 N/A，
   最終 already=690 / N/A=374。)

============================================================
Notes / 一致性檢查
============================================================
- Cascade（submission.csv）: ST1 No=374 → ST2 N/A=374 ✓；
  ST3 N/A=624 = ST1 No 374 + ST2 No 250 ✓；ST4 N/A=374 = ST1 No 374 ✓。
- ST1/ST2/ST4 與 submit_9/submit_11 相同；ST3 差異來自 codex 來源不同
  （submission_12=add_context；submit_11=offsetctx_scoped）。
- ST3 Misleading=0：multitask head 與 codex 此版皆未輸出 Misleading。
- Blind set 無 gold → 未做 success-by-class / Macro-F1；如需裁決請用 val.json。
