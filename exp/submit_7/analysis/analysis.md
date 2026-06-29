submit_7 analysis (blind test set, 2000 rows, no labels)

Pipeline: ST1/2/3 = 5-member soft-vote ensemble (focal_g3_w4 / mix_a2_b3 / mt_st123); ST4 = codex boundary_rules_v4 + 14-row na-fix. No Gemma/Codex fallback on ST1-3.

============================================================
ST1: promise_status
============================================================

--- class_distribution (final) ---
  No                                 341  (17.1%)
  Yes                               1659  (83.0%)
  TOTAL                             2000

--- score_distribution (soft-vote prob, N=all 2000) ---
  <0.50                                0  (0.0%)
  0.50-0.60                          130  (6.5%)
  0.60-0.70                          149  (7.4%)
  0.70-0.80                          201  (10.1%)
  0.80-0.90                          398  (19.9%)
  >=0.90                            1122  (56.1%)
  median confidence               0.917575

--- fallback_change_rate ---
  (no fallback — ST1 是 5-member soft-vote ensemble)
============================================================
ST2: evidence_status
============================================================

--- class_distribution (final) ---
  No                                 257  (15.5% of active)
  Yes                               1402  (84.5% of active)
  N/A (ST1=No)                       341
  TOTAL                             2000

--- score_distribution (soft-vote prob, active rows only) ---
  <0.50                                0  (0.0%)
  0.50-0.60                           46  (2.8%)
  0.60-0.70                           51  (3.1%)
  0.70-0.80                           47  (2.8%)
  0.80-0.90                           59  (3.6%)
  >=0.90                            1456  (87.8%)
  median confidence               0.999907

--- fallback_change_rate ---
  (no fallback — ST2 是 5-member soft-vote ensemble)
============================================================
ST3: evidence_quality
============================================================

--- class_distribution (final) ---
  Clear                             1222  (87.2% of active)
  Not Clear                          180  (12.8% of active)
  N/A                                598
  TOTAL                             2000

--- score_distribution (soft-vote prob, active rows w/ score, N=1402) ---
  <0.50                                0  (0.0%)
  0.50-0.60                          121  (8.6%)
  0.60-0.70                          140  (10.0%)
  0.70-0.80                          142  (10.1%)
  0.80-0.90                          263  (18.8%)
  >=0.90                             736  (52.5%)
  median confidence               0.908295

--- fallback_change_rate ---
  (no fallback — ST3 是 5-member multitask soft-vote ensemble)
============================================================
ST4: verification_timeline
============================================================

--- class_distribution (final) ---
  already                            717  (43.2% of active)
  within_2_years                      82  (4.9% of active)
  between_2_and_5_years              633  (38.2% of active)
  more_than_5_years                  227  (13.7% of active)
  N/A                                341
  TOTAL                             2000

  (no score — ST4 是 codex 預測 + 14 列 na-fix;無機率分數)
