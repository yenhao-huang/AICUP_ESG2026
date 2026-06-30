============================================================
ST1: promise_status
============================================================

--- class_distribution (final) ---
  No                              374  (18.7%)
  Yes                            1626  (81.3%)
  TOTAL                          2000

--- score_distribution (soft-vote raw, N=all 2000) ---
  <0.50             0  (0.0%)
  0.50-0.60       130  (6.5%)
  0.60-0.70       149  (7.5%)
  0.70-0.80       201  (10.1%)
  0.80-0.90       398  (19.9%)
  >=0.90         1122  (56.1%)
  median confidence    0.917575

--- fallback_change_rate (Gemma, tau=0.6) ---
  total rows             2000
  kept BERT (conf>=tau)  1870
  escalated to Gemma      130
  changed by Gemma         59  (45.4% of escalated)
  overall change rate  2.95%
    Yes->No                46
    No->Yes                13

============================================================
ST2: evidence_status
============================================================

--- class_distribution (final) ---
  No                              250  (15.4% of active)
  Yes                            1376  (84.6% of active)
  N/A (ST1=No)                    374
  TOTAL                          2000

--- score_distribution (soft-vote, gated active N=1626) ---
  <0.50             0  (0.0%)
  0.50-0.60        45  (2.8%)
  0.60-0.70        49  (3.0%)
  0.70-0.80        47  (2.9%)
  0.80-0.90        60  (3.7%)
  >=0.90         1425  (87.6%)
  median confidence    0.999908

(no fallback -- ST2 is soft-vote ensemble)

============================================================
ST3: evidence_quality
============================================================

--- class_distribution (final) ---
  Clear                          1129  (82.0% of active)
  Not Clear                       247  (18.0% of active)
  N/A                             624
  TOTAL                          2000

--- score_distribution (multitask, active rows w/ score) ---
  <0.50             2  (0.1%)
  0.50-0.60       106  (7.7%)
  0.60-0.70        90  (6.5%)
  0.70-0.80       127  (9.2%)
  0.80-0.90       233  (16.9%)
  >=0.90          818  (59.4%)
  median confidence    0.938610

(no fallback -- ST3 is bert_multitask)

============================================================
ST4: verification_timeline
============================================================

--- class_distribution (final) ---
  already                         720  (44.3% of active)
  between_2_and_5_years           543  (33.4% of active)
  more_than_5_years               278  (17.1% of active)
  within_2_years                   85  (5.2% of active)
  N/A                             374
  TOTAL                          2000

(no score section -- ST4 is codex add-context)
(no fallback -- codex add-context, v6 prompt, ST1-gated)
