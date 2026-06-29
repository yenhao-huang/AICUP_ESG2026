---
name: submit
description: ESG contest submission preparation skill. Writes test run.sh, creates submit run.sh, verifies diff is exactly 2 lines (HERE + DATA), analyzes prediction confidence against data/predictions baselines, and generates submit_N/analysis/analysis.md with class_distribution, score_distribution, and fallback_change_rate.
---

# Submit

Invoke when the user says "上傳", "submit", "幫我做 submit", "準備提交", or asks to create a submit dir from a test dir in the ESG contest project.

## Invariants

- ROOT is always `/workspace/esg_contest`.
- Test dirs: `exp/exp41/submit/test_*/run.sh` — use `DATA="$ROOT/data/benchmarks/val.json"`.
- Submit dirs: `exp/exp41/submit/submit_N/run.sh` — use `DATA="$ROOT/data/raw_data/vpesg4k_test_2000.json"`.
- The **only** diff between a test and its submit `run.sh` must be exactly 2 lines:
  ```
  8c8   HERE="$ROOT/exp/exp41/submit/<test>"  →  HERE="$ROOT/exp/exp41/submit/submit_N"
  9c9   DATA="$ROOT/data/benchmarks/val.json" →  DATA="$ROOT/data/raw_data/vpesg4k_test_2000.json"
  ```
- Never touch checkpoints, thresholds, backends, or any other config when generating the submit script.

## Steps

### 1. Write test run.sh (if missing)

If the user provides a pipeline description and the test dir doesn't exist yet, create `exp/exp41/submit/test_<name>/run.sh` following the same structure as existing test scripts (see `test_1/run.sh`, `test_2/run.sh`). Always use `val.json` as DATA.

### 2. Find next submit number

```bash
ls exp/exp41/submit/ | grep '^submit_' | sort -t_ -k2 -n | tail -1
```

Increment by 1.

### 3. Generate submit run.sh

```bash
mkdir -p exp/exp41/submit/submit_N
sed \
  -e "s|exp/exp41/submit/<test_name>|exp/exp41/submit/submit_N|g" \
  -e "s|data/benchmarks/val.json|data/raw_data/vpesg4k_test_2000.json|g" \
  exp/exp41/submit/test_<name>/run.sh \
  > exp/exp41/submit/submit_N/run.sh
chmod +x exp/exp41/submit/submit_N/run.sh
```

### 4. Verify diff (mandatory)

```bash
diff exp/exp41/submit/test_<name>/run.sh exp/exp41/submit/submit_N/run.sh
```

Must show exactly 2 changed lines (HERE and DATA). If more lines differ, stop and investigate.

### 5. Analyze prediction confidence

After the test run completes, the stage CSVs land in `exp/exp41/submit/test_<name>/stage1/` and `stage2/`. Compare their confidence distribution against the baseline in `data/predictions/`.

**Find baseline CSVs:**
```bash
ls data/predictions/          # date-stamped subdirs (e.g. 0603, 0609)
ls data/predictions/<latest>/
```

**Run analysis inline (Python one-liner style):**
```python
import csv, statistics, re

def conf_stats(path):
    confs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            sy, sn = row.get("score_yes",""), row.get("score_no","")
            if sy and sn:
                confs.append(max(float(sy), float(sn)))
            else:
                # stage2: scores embedded in postprocess_reason
                m = re.findall(r"score_(?:yes|no)=([0-9.eE+-]+)", row.get("postprocess_reason",""))
                if m: confs.append(max(float(v) for v in m))
    if not confs: return {}
    return {
        "n": len(confs),
        "mean": round(statistics.mean(confs), 4),
        "median": round(statistics.median(confs), 4),
        "pct_below_090": round(sum(c < 0.90 for c in confs) / len(confs), 3),
        "pct_below_080": round(sum(c < 0.80 for c in confs) / len(confs), 3),
        "pct_below_070": round(sum(c < 0.70 for c in confs) / len(confs), 3),
    }
```

Compare test output vs baseline (same stage, same model family). Flag if:
- `pct_below_090` diverges from baseline by > 5 pp for ST1
- `pct_below_080` diverges by > 5 pp for ST2

## Analysis

See `references/analysis.md` for data sources, fallback detection, score bins, and output format.

Write results to `exp/exp41/submit/submit_N/analysis/analysis.md`.

## Output format

```
submit dir : exp/exp41/submit/submit_N/run.sh

diff:
  8c8  HERE  test_<name> → submit_N
  9c9  DATA  val.json    → vpesg4k_test_2000.json

confidence (ST1):
  baseline  mean=X  median=X  below_0.90=X%
  test run  mean=X  median=X  below_0.90=X%

verdict: [confidence looks normal | WARNING: X% rows below threshold T, Δ=+Ypp vs baseline]

analysis : exp/exp41/submit/submit_N/analysis/analysis.md
```
