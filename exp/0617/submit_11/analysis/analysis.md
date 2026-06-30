# submit_11 analysis

- Data: `data/raw_data/vpesg4k_test_2000.json` (blind, 2000 rows, no labels → success-by-class omitted)
- Pipeline: ST1/ST2/ST4 inherited from submit_9 (Gemma low-conf fallback); **ST3 = multitask BERT → codex low-conf fallback (CONF=0.60)** + na-fix re-prediction of the 46 low-conf rows that codex had left N/A.
- Final merged: `submission.csv` (2000 rows)

============================================================
ST1: promise_status
============================================================

--- class_distribution (final) ---
  No                                    374  (18.7%)
  Yes                                  1626  (81.3%)
  TOTAL                                2000

--- score_distribution (BERT raw, N=1870 scored; 130 Gemma rows have no BERT score) ---
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
  changed by Gemma (excl. later N/A)     47
    Yes→No                               19
    No→Yes                               28

============================================================
ST3: evidence_quality   ← submit_11 的改動點
============================================================

--- class_distribution (final) ---
  Clear                                1142  (83.0% of active)
  Not Clear                             234  (17.0% of active)
  Misleading                              0  (0.0% of active)
  N/A                                   624
  TOTAL                                2000

--- score_distribution (multitask BERT, active rows w/ score, N=1268 of 1376 active) ---
  <0.50                                   0  (0.0%)
  0.50-0.60                               0  (0.0%)
  0.60-0.70                              90  (7.1%)
  0.70-0.80                             127  (10.0%)
  0.80-0.90                             233  (18.4%)
  >=0.90                                818  (64.5%)
  median confidence                   0.948677
  注: codex_fallback 取代的 152 列 reason 不帶 score（故 active 1376 中僅 1268 有分數）。

--- fallback_change_rate (Codex, CONF=0.60, ungated) ---
  total rows                           2000
  BERT-only (conf>=0.60)               1848
  Codex fallback rows                   152   (全部取得實際 clarity 標籤)
  changed by Codex                       70   (46.1% of fallback; 3.50% of all)
    Not Clear→Clear                      41
    Clear→Not Clear                      29
  source dist (final): bert_multitask=1848, codex_fallback=152

--- na-fix (low-conf rows codex 原本判 N/A，已用 codex 強制重判) ---
  re-predicted rows                      46   (原 codex=stage2_filter:N/A)
    Clear                                28
    Not Clear                            18
  → 補進 st3_codex_test2000_offsetctx_scoped_fixed.csv，152 列 fallback 因此全有實際標籤。

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
  (no score — codex/rule-based; inherited from submit_9, incl. 14-row na-fix patch。
   注: stage4/codex_gated.csv 原 already=699 / N/A=364，merge_pipeline 再按 ST1=No
   重 gate（10 列具體 timeline→N/A：already −9, 2-5y −1），最終 already=690 / N/A=374。)

============================================================
Notes / 一致性檢查
============================================================
- Cascade: ST1 No=374 → ST2 N/A=374 ✓。ST3 N/A=624 = ST1 No(374) + (ST2 No=250) ✓。
- ST4 N/A=364 ≠ ST1 No=374（少 10）：沿用 submit_9 的 ST4，na-fix 對 ST1=Yes 列補了具體 timeline；此 −10 為繼承自 submit_9 的既有差異，非本次 ST3 改動造成。
- ST3 Misleading=0：multitask head 與 codex(clear_notclear_*) 此版皆未輸出 Misleading。
- Blind set 無 gold → 未做 success-by-class / Macro-F1；如需 val 驗證請用 val.json 另跑。

你說得對——這才是有意義的分母。multitask BERT 在 ST2 gate 通過（active, 1376 列） 上的信心分布：

mean=0.8770, median=0.9386, min=0.4989, max=0.9992

信心區間（active only）

┌──────────────┬──────┬───────┐
│     區間     │ 列數 │ 比例  │
├──────────────┼──────┼───────┤
│ < 0.50       │ 2    │ 0.1%  │
├──────────────┼──────┼───────┤
│ [0.50, 0.60) │ 106  │ 7.7%  │
├──────────────┼──────┼───────┤
│ [0.60, 0.70) │ 90   │ 6.5%  │
├──────────────┼──────┼───────┤
│ [0.70, 0.80) │ 127  │ 9.2%  │
├──────────────┼──────┼───────┤
│ [0.80, 0.90) │ 233  │ 16.9% │
├──────────────┼──────┼───────┤
│ [0.90, 1.00) │ 818  │ 59.4% │
└──────────────┴──────┴───────┘