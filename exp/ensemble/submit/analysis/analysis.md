============================================================
ST1: promise_status
============================================================

--- class_distribution (final) ---
  No                                 327  (16.4%)
  Yes                               1673  (83.7%)
  TOTAL                             2000

--- score_distribution (BERT raw, N=all 2000) ---
  <0.50                                0  (0.0%)
  0.50-0.60                          176  (8.8%)
  0.60-0.70                          184  (9.2%)
  0.70-0.80                          201  (10.1%)
  0.80-0.90                          377  (18.9%)
  >=0.90                            1062  (53.1%)
  median confidence               0.913374

--- fallback_change_rate (Gemma, tau=0.6) ---
  total rows                        2000
  BERT-only (conf>=tau)             1824
  Gemma fallback rows                176
  changed by Gemma                    74  (42.0% of fallback)
  overall change rate             3.70%
    Yes->No                           55
    No->Yes                           19

============================================================
ST2: evidence_status
============================================================

--- class_distribution (final) ---
  No                                 271  (16.2% of active)
  Yes                               1402  (83.8% of active)
  N/A (ST1=No)                       327
  TOTAL                             2000

--- score_distribution (BERT raw, active rows only, N=1673) ---
  <0.50                                0  (0.0%)
  0.50-0.60                           14  (0.8%)
  0.60-0.70                           12  (0.7%)
  0.70-0.80                           15  (0.9%)
  0.80-0.90                           32  (1.9%)
  >=0.90                            1600  (95.6%)
  median confidence               0.999951

--- fallback_change_rate (Gemma local no-rag, tau=0.9) ---
  total rows                        2000
  kept BERT (conf>=tau or N/A)      1927
  escalated to Gemma                  73
  changed by Gemma                    32  (43.8% of escalated)
  overall change rate             1.60%

============================================================
ST3: evidence_quality
============================================================

--- class_distribution (final) ---
  Clear                             1158  (82.6% of active)
  Not Clear                          244  (17.4% of active)
  N/A                                598
  TOTAL                             2000

--- score_distribution (BERT, active rows w/ score, N=1402) ---
  <0.50                                2  (0.1% of active w/ score)
  0.50-0.60                          110  (7.8% of active w/ score)
  0.60-0.70                           93  (6.6% of active w/ score)
  0.70-0.80                          128  (9.1% of active w/ score)
  0.80-0.90                          235  (16.8% of active w/ score)
  >=0.90                             834  (59.5% of active w/ score)
  median confidence               0.938823

--- fallback_change_rate ---
  (no fallback — ST3 is multitask BERT-only)

============================================================
ST4: verification_timeline
============================================================

--- class_distribution (final) ---
  already                            725  (43.3% of active)
  between_2_and_5_years              641  (38.3% of active)
  more_than_5_years                  228  (13.6% of active)
  within_2_years                      79  (4.7% of active)
  N/A (total)                        327
    of which ST1=No gated            327
    of which codex emitted N/A         0
  TOTAL                             2000

--- fallback_change_rate ---
  (codex-based ST4, gated by ST1; no BERT score)
