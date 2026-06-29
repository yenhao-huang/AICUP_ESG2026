After the submit run completes, generate `submit_N/analysis/analysis.md` with three sections per stage: **class_distribution**, **score_distribution** (BERT stages only), and **fallback_change_rate** (only if Gemma fallback is active for that stage).

```bash
mkdir -p exp/exp41/submit/submit_N/analysis
```

## Data sources per stage

| Stage | CSV | Label col | Score source |
|-------|-----|-----------|--------------|
| ST1 | `stage1/bert_opt1.csv` | `promise_status` | `score_yes`, `score_no` columns |
| ST2 | `stage2/bert.csv` | `evidence_status` | `postprocess_reason` (`score_yes=...;score_no=...`) |
| ST3 | `stage3/bert_cw_a1.csv` | `evidence_quality` | `evidence_quality_reason` (`score_clear=...;score_not_clear=...;score_misleading=...`) |
| ST4 | `stage4/*.csv` | `verification_timeline` | no score (rule-based / LLM) |

**Active-row logic:**
- ST2 active: `filter_passed == "yes"` ; N/A count = total − active
- ST3 active: rows where `evidence_quality` is a known label (Clear / Not Clear / Misleading) ; N/A count = total − active
- ST4 active: `stage4_filtered == "no"` and `verification_timeline` non-empty ; N/A count = total − active

## Fallback detection

A stage uses Gemma fallback if its `run.sh` calls `pred_by_bert_gemma.py` for that stage.
Read the fallback stats from `stage<N>/summary.json` (written by `pred_by_bert_gemma.py`):

```python
import json
s = json.load(open("stage2/summary.json"))
escalated   = s["escalated_rows"]       # rows sent to Gemma
changed     = s["changed_by_llm"]       # rows where Gemma disagreed with BERT
total_rows  = s["total_rows"]
tau         = s["threshold"]
# fallback_change_rate = changed / total_rows
# direction breakdown: iterate s["results"] where changed==true
```

## Score distribution bins

Always use these fixed bins (drop rows with conf < 0.50 into `<0.50`):

```
<0.50 | 0.50-0.60 | 0.60-0.70 | 0.70-0.80 | 0.80-0.90 | >=0.90
```

Extract max confidence per row:
- ST1: `max(float(score_yes), float(score_no))`
- ST2: `max(re.findall(r"score_(?:yes|no)=([0-9.eE+\-]+)", postprocess_reason))`
- ST3: `max(re.findall(r"score_(?:clear|not_clear|misleading)=([0-9.eE+\-]+)", evidence_quality_reason))`

For a soft-vote stage the per-row score is not in `postprocess_reason`; reconstruct it
from the members file (e.g. `stage2/tmp/softvote_raw.members.csv`): average each member's
`score_yes`/`score_no` per id, then take the max of the two averages.

**Bin over the GATED active rows, NOT all 2000.** This is the recurring mistake: a block
labelled `active rows only` but actually binned over every row (the N/A rows then inflate
`>=0.90` and skew the median). After extracting confidences, filter to that stage's active
set BEFORE binning:
- ST1: all 2000 rows (top of the cascade, every row active). Label `N=all 2000`.
- ST2: active = `filter_passed == "yes"` (ST1=Yes). Drop ST1=No / N/A rows. Label `gated active N=<count>`.
- ST3: active = rows whose final `evidence_quality` is a real label (Clear/Not Clear/Misleading); of those bin only rows with a parseable score (`% of active w/ score`).
- ST4: no score section.
Sanity-check: the ungated distribution must reproduce the prior run's numbers before you
trust the gated one.

## analysis.md format

Reference: `exp/exp41/submit/submit_2/analysis/analysis.md` (has both BERT-only and Gemma-fallback stages).

```
============================================================
ST1: promise_status
============================================================

--- class_distribution (final) ---
  No                                    345  (17.2%)
  Yes                                  1655  (82.8%)
  TOTAL                               2000

--- score_distribution (BERT raw, N=all 2000) ---
  <0.50                                   0  (0.0%)
  0.50-0.60                              12  (0.6%)
  0.60-0.70                              18  (0.9%)
  0.70-0.80                              21  (1.1%)
  0.80-0.90                              26  (1.3%)
  >=0.90                               1923  (96.2%)
  median confidence                   0.999813

--- fallback_change_rate (Gemma, tau=0.90) ---
  total rows                          2000
  BERT-only (conf>=0.90)              1923
  Gemma fallback rows                 77
  changed by Gemma                    41  (53.2% of fallback)
  overall change rate                 2.05%
    Yes→No                            26
    No→Yes                            15

============================================================
ST2: evidence_status
============================================================

--- class_distribution (final) ---
  No                                    240  (14.5% of active)
  Yes                                  1415  (85.5% of active)
  N/A (ST1=No)                        345
  TOTAL                               2000

--- score_distribution (BERT raw, active rows only) ---
  <0.50                                   0  (0.0%)
  0.50-0.60                              22  (1.3%)
  0.60-0.70                              28  (1.7%)
  0.70-0.80                              29  (1.8%)
  0.80-0.90                              44  (2.7%)
  >=0.90                               1532  (92.6%)
  median confidence                   0.999908

--- fallback_change_rate (Gemma, tau=0.8) ---
  total rows                          2000
  kept BERT (conf>=tau or N/A)        1921
  escalated to Gemma                  79
  changed by Gemma                    38  (48.1% of escalated)
  overall change rate                 1.90%
    Yes→No                            18
    No→Yes                            20

============================================================
ST3: evidence_quality
============================================================

--- class_distribution (final) ---
  Clear                                1199  (84.7% of active)
  Not Clear                             216  (15.3% of active)
  N/A                                 585
  TOTAL                               2000

--- score_distribution (BERT, active rows only) ---
  <0.50                                   2  (0.1% of active w/ score)
  0.50-0.60                             110  (7.8% of active w/ score)
  0.60-0.70                             116  (8.2% of active w/ score)
  0.70-0.80                             163  (11.5% of active w/ score)
  0.80-0.90                             263  (18.6% of active w/ score)
  >=0.90                                761  (53.8% of active w/ score)
  median confidence                   0.915515

============================================================
ST4: verification_timeline
============================================================

--- class_distribution (final) ---
  already                               721  (43.6% of active)
  between_2_and_5_years                 623  (37.6% of active)
  more_than_5_years                     228  (13.8% of active)
  within_2_years                         83  (5.0% of active)
  N/A                                 345
  TOTAL                               2000
```

Notes:
- If a stage has no Gemma fallback, replace the `--- fallback_change_rate ---` block with `(no Gemma fallback — STN is BERT-only)`.
- ST1 score_distribution label: `N=all 2000`; ST2/ST3: `active rows only`.
- ST3 pct label: `% of active w/ score` (rows where score was parseable).
- ST4 has no score section.

Write the file to `exp/exp41/submit/submit_N/analysis/analysis.md`.
