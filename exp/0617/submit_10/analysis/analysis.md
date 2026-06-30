# submit_10 analysis

- Data: `data/raw_data/vpesg4k_test_2000.json` (blind, 2000 rows, no labels → success-by-class omitted)
- Pipeline: ST1/ST2/ST4 同 submit_9/submit_11（Gemma low-conf fallback）；**ST3 = Codex 全量預測（add_page_abstract context）→ gate by ST1+ST2**（無 BERT、無低信心 fallback）。
- Final merged: `submission.csv`（2000 列；下列 class_distribution 以 submission.csv 為準）

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
  changed by Gemma (excl. later N/A)     47
    Yes→No                               19
    No→Yes                               28

============================================================
ST3: evidence_quality   ← submit_10 的特性：Codex 全量 + gate
============================================================

--- class_distribution (final) ---
  Clear                                1124  (81.7% of active)
  Not Clear                             252  (18.3% of active)
  Misleading                              0  (0.0% of active)
  N/A                                   624
  TOTAL                                2000

--- score_distribution ---
  (no score — ST3 為 Codex 全量預測，無 max-softmax)

--- fallback_change_rate ---
  (no fallback — ST3 全量 Codex 直接預測，source=codex×2000；
   再以 apply_stage12_gate_to_stage3 gate by ST1+ST2：ST1 No 374 ∪ ST2 non-Yes 624 → 624 列設 N/A)

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
  (no score — codex/rule-based。注: stage4/codex_gated.csv 原 N/A=364，
   merge_pipeline 再按 ST1=No gate，最終 N/A=374。)

============================================================
Notes / 一致性檢查
============================================================
- Cascade: ST1 No=374 → ST2 N/A=374 ✓；ST3 N/A=624 = ST1 No 374 + ST2 No 250 ✓；
  ST4 N/A=374 = ST1 No 374 ✓（submit_10 的 ST4 對齊乾淨）。
- ST3 Misleading=0：Codex(add_page_abstract) 此版未輸出 Misleading。
- 與 submit_11 差異：ST1/ST2/ST4 相同；ST3 不同方法
  （submit_10=Codex 全量；submit_11=multitask BERT + codex 低信心 fallback）。
- Blind set 無 gold → 未做 success-by-class / Macro-F1。


