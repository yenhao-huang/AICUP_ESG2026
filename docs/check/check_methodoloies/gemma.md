# Gemma Subset Extraction Methodology

檢查目標：先從 submission 產物抽出與 gemma 有關的題目，輸出成 `stage1_gemma.csv`、`stage2_gemma.csv`，再用這兩個子集做後續一致性檢查。

## Extract Gemma Rows

Stage 1 來源：

```text
docs/check/check_submit_model_consistent/submission_12/stage1/bert_gemma.csv
```

抽取條件：

```text
source=gemma_fallback
or mode contains gemma
or model_family contains gemma
```

輸出：

```text
docs/check/check_submit_model_consistent/submit/gemma/stage1_gemma.csv
```

Stage 2 來源：

```text
docs/check/check_submit_model_consistent/submission_12/stage2/tmp/bert_gemma_raw.csv
```

抽取條件：

```text
prediction_source contains gemma
```

輸出：

```text
docs/check/check_submit_model_consistent/submit/gemma/stage2_gemma.csv
```

## Current Extracted Rows

```text
stage1_gemma_rows 130
stage2_gemma_rows 118
```

## Reproduce

```bash
python3 - <<'PY'
from pathlib import Path
import csv

root = Path(".")

extracts = [
    (
        "docs/check/check_submit_model_consistent/submission_12/stage1/bert_gemma.csv",
        "docs/check/check_submit_model_consistent/submit/gemma/stage1_gemma.csv",
        lambda row: row.get("source") == "gemma_fallback"
        or "gemma" in row.get("mode", "").lower()
        or "gemma" in row.get("model_family", "").lower(),
    ),
    (
        "docs/check/check_submit_model_consistent/submission_12/stage2/tmp/bert_gemma_raw.csv",
        "docs/check/check_submit_model_consistent/submit/gemma/stage2_gemma.csv",
        lambda row: "gemma" in row.get("prediction_source", "").lower(),
    ),
]

for source, output, keep in extracts:
    with (root / source).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if keep(row)]
        fields = reader.fieldnames or []
    with (root / output).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(output, len(rows))
PY
```
