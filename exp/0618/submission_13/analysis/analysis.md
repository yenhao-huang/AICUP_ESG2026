============================================================
ST1: promise_status
============================================================

--- class_distribution (final) ---
  No                              406  (20.3%)
  Yes                            1594  (79.7%)
  TOTAL                          2000

--- score_distribution (soft-vote raw, N=all 2000) ---
  <0.50             0  (0.0%)
  0.50-0.60       130  (6.5%)
  0.60-0.70       149  (7.5%)
  0.70-0.80       201  (10.1%)
  0.80-0.90       398  (19.9%)
  >=0.90         1122  (56.1%)
  median confidence    0.917575

--- fallback_change_rate (Gemma, tau=0.7) ---
  total rows             2000
  kept BERT (conf>=tau)  1721
  escalated to Gemma      279
  changed by Gemma        113  (40.5% of escalated)
  overall change rate  5.65%
    Yes->No                89
    No->Yes                24

============================================================
ST2: evidence_status
============================================================

--- class_distribution (final) ---
  No                              251  (15.7% of active)
  Yes                            1343  (84.3% of active)
  N/A (ST1=No)                    406
  TOTAL                          2000

--- score_distribution (soft-vote, gated active N=1594) ---
  <0.50             0  (0.0%)
  0.50-0.60        45  (2.8%)
  0.60-0.70        48  (3.0%)
  0.70-0.80        46  (2.9%)
  0.80-0.90        59  (3.7%)
  >=0.90         1396  (87.6%)
  median confidence    0.999909

--- fallback_change_rate (Gemma, tau=0.9) ---
  total rows             2000
  kept BERT (conf>=tau)  1746
  escalated to Gemma      254
  changed by Gemma        122  (48.0% of escalated)
  overall change rate  6.10%
    Yes->No                78
    No->Yes                44

============================================================
ST3: evidence_quality
============================================================

--- class_distribution (final) ---
  Clear                          1099  (81.8% of active)
  Not Clear                       244  (18.2% of active)
  N/A                             657
  TOTAL                          2000

--- score_distribution (multitask, active rows w/ score) ---
  <0.50             2  (0.1%)
  0.50-0.60       106  (7.9%)
  0.60-0.70        89  (6.6%)
  0.70-0.80       122  (9.1%)
  0.80-0.90       229  (17.1%)
  >=0.90          795  (59.2%)
  median confidence    0.937302

(no fallback -- ST3 is bert_multitask)

============================================================
ST4: verification_timeline
============================================================

--- class_distribution (final) ---
  already                         688  (43.2% of active)
  between_2_and_5_years           544  (34.1% of active)
  more_than_5_years               278  (17.4% of active)
  within_2_years                   84  (5.3% of active)
  N/A                             406
  TOTAL                          2000

(no score section -- ST4 is codex add-context)
(no fallback -- codex add-context, v6 prompt, ST1-gated)
