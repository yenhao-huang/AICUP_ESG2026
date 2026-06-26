#!/usr/bin/env python3
"""Check submit/stage outputs against results/predict submit outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

STAGES = {
    "stage1": {
        "check": "docs/check/check_submit_model_consistent/submit/stage1/softvote_raw.csv",
        "results": "results/predict/stage1/ensemble/submit/softvote.csv",
        "prediction_columns": ("promise_status", "score_yes", "score_no", "raw_prediction"),
    },
    "stage2": {
        "check": "docs/check/check_submit_model_consistent/submit/stage2/softvote_raw.csv",
        "results": "results/predict/stage2/ensemble/submit/softvote.csv",
        "prediction_columns": (
            "evidence_status",
            "evidence_status_raw",
            "filter_passed",
            "prediction_source",
            "postprocess_reason",
        ),
    },
    "stage3": {
        "check": "docs/check/check_submit_model_consistent/submit/stage3/multitask.csv",
        "results": "results/predict/stage3/multitaskbert/submit/prediction.csv",
        "prediction_columns": (
            "evidence_quality",
            "evidence_quality_raw",
            "evidence_quality_source",
            "evidence_quality_reason",
        ),
    },
}


def read_csv(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = {str(row.get("id", "")): dict(row) for row in reader}
        return list(reader.fieldnames or []), rows


def compare_stage(stage: str, max_examples: int) -> dict[str, object]:
    config = STAGES[stage]
    check_path = ROOT / str(config["check"])
    results_path = ROOT / str(config["results"])
    check_cols, check_rows = read_csv(check_path)
    results_cols, results_rows = read_csv(results_path)

    check_ids = set(check_rows)
    results_ids = set(results_rows)
    common_ids = sorted(check_ids & results_ids, key=lambda value: int(value) if value.isdigit() else value)
    missing_in_results = sorted(check_ids - results_ids)
    extra_in_results = sorted(results_ids - check_ids)

    pred_mismatches: dict[str, list[dict[str, str]]] = {}
    for col in config["prediction_columns"]:
        examples = []
        for row_id in common_ids:
            check_value = check_rows[row_id].get(col, "")
            results_value = results_rows[row_id].get(col, "")
            if check_value != results_value:
                examples.append({"id": row_id, "check": check_value, "results": results_value})
                if len(examples) >= max_examples:
                    break
        if examples:
            pred_mismatches[str(col)] = examples

    full_mismatch_counts: dict[str, int] = {}
    for col in check_cols:
        if col == "id" or col not in results_cols:
            continue
        count = sum(1 for row_id in common_ids if check_rows[row_id].get(col, "") != results_rows[row_id].get(col, ""))
        if count:
            full_mismatch_counts[col] = count

    prediction_mismatch_count = sum(
        1
        for row_id in common_ids
        if any(
            check_rows[row_id].get(col, "") != results_rows[row_id].get(col, "")
            for col in config["prediction_columns"]
        )
    )
    id_mismatch_count = len(missing_in_results) + len(extra_in_results)
    prediction_aligned = id_mismatch_count == 0 and prediction_mismatch_count == 0

    return {
        "stage": stage,
        "check_path": str(check_path.relative_to(ROOT)),
        "results_path": str(results_path.relative_to(ROOT)),
        "check_rows": len(check_rows),
        "results_rows": len(results_rows),
        "matched_ids": len(common_ids),
        "id_mismatch_count": id_mismatch_count,
        "missing_in_results_first": missing_in_results[:max_examples],
        "extra_in_results_first": extra_in_results[:max_examples],
        "prediction_columns": list(config["prediction_columns"]),
        "prediction_aligned": prediction_aligned,
        "prediction_mismatch_rows": prediction_mismatch_count,
        "prediction_mismatch_examples": pred_mismatches,
        "full_csv_exact_aligned": id_mismatch_count == 0 and not full_mismatch_counts,
        "full_csv_mismatch_columns": full_mismatch_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    results = [compare_stage(stage, args.max_examples) for stage in stages]
    print(json.dumps(results if args.stage == "all" else results[0], ensure_ascii=False, indent=2))
    return 0 if all(result["prediction_aligned"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
