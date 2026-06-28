# Gemma Submit Consistency

檢查目標：比較 `stage1_gemma.csv`、`stage2_gemma.csv` 與完整 submit 檔中相同 id 的結果是否一致。

檢查日期：2026-06-28

## Files

| stage | gemma subset | submit file | compared ids |
| --- | --- | --- | ---: |
| stage1 | `docs/check/check_submit_model_consistent/submit/gemma/stage1_gemma.csv` | `docs/check/check_submit_model_consistent/submit/gemma/stage1.csv` | 130 |
| stage2 | `docs/check/check_submit_model_consistent/submit/gemma/stage2_gemma.csv` | `docs/check/check_submit_model_consistent/submit/gemma/stage2.csv` | 118 |

兩個 stage 的 gemma subset id 都存在於完整 submit 檔：missing in submit = 0。

## Summary

| stage | compared prediction column | matched | mismatched | match rate |
| --- | --- | ---: | ---: | ---: |
| stage1 | `promise_status` | 130 | 0 | 100.00% |
| stage2 | `evidence_status` | 118 | 0 | 100.00% |
| stage2 | `filter_passed` | 118 | 0 | 100.00% |

完整共同欄位也完全一致：

```text
stage1 full_mismatch_columns {}
stage2 full_mismatch_columns {}
```

## Stage 1

比較欄位：`promise_status`

| stage1_gemma | submit | count |
| --- | --- | ---: |
| No | No | 85 |
| Yes | Yes | 45 |

判斷：`stage1_gemma.csv` 是完整 `stage1.csv` 中 gemma 相關題目的一致子集。

## Stage 2

比較欄位：`evidence_status`

| stage2_gemma | submit | count |
| --- | --- | ---: |
| No | No | 59 |
| Yes | Yes | 59 |

比較欄位：`filter_passed`

| stage2_gemma | submit | count |
| --- | --- | ---: |
| yes | yes | 118 |

判斷：`stage2_gemma.csv` 是完整 `stage2.csv` 中 gemma 相關題目的一致子集。

## Reproduce

```bash
python3 - <<'PY'
from pathlib import Path
import csv

root = Path(".")

comparisons = {
    "stage1": (
        "docs/check/check_submit_model_consistent/submit/gemma/stage1_gemma.csv",
        "docs/check/check_submit_model_consistent/submit/gemma/stage1.csv",
        ["promise_status"],
    ),
    "stage2": (
        "docs/check/check_submit_model_consistent/submit/gemma/stage2_gemma.csv",
        "docs/check/check_submit_model_consistent/submit/gemma/stage2.csv",
        ["evidence_status", "filter_passed"],
    ),
}

def read_csv(path):
    with (root / path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), {str(row["id"]): dict(row) for row in reader}

for stage, (subset_path, submit_path, pred_cols) in comparisons.items():
    subset_fields, subset = read_csv(subset_path)
    submit_fields, submit = read_csv(submit_path)
    common = sorted(set(subset) & set(submit), key=int)
    missing = sorted(set(subset) - set(submit), key=int)
    print(stage, "subset_rows", len(subset), "submit_rows", len(submit), "common", len(common), "missing_in_submit", len(missing))

    for col in pred_cols:
        mismatched = [row_id for row_id in common if subset[row_id].get(col, "") != submit[row_id].get(col, "")]
        print(stage, col, "mismatched", len(mismatched))

    full_mismatch_columns = {}
    for col in subset_fields:
        if col == "id" or col not in submit_fields:
            continue
        count = sum(1 for row_id in common if subset[row_id].get(col, "") != submit[row_id].get(col, ""))
        if count:
            full_mismatch_columns[col] = count
    print(stage, "full_mismatch_columns", full_mismatch_columns)
PY
```
