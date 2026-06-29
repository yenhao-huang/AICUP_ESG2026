#!/usr/bin/env python3
"""Post-process and evaluate ST4 Codex predictions for variant A1.

After a two-part run (split by disk-full at row 334), the output CSV has
rows 1-335 that passed stage1 marked as 'error_skipped'. This script:
1. Reads the output CSV.
2. For each row with stage4_error='error_skipped', looks up the raw JSON from
   the original raw dir and re-normalizes the label.
3. Writes the fixed CSV back.
4. Computes per-class F1 and Macro-F1 against val_public.json ground truth.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---- paths ----
OUTPUT_CSV = Path("/workspace/esg_contest/exp/agent_loop/claude/20260609T172829/loops/loops001/exp/A1/val_public_few_shot_boundary_v1.csv")
ORIG_RAW_DIR = Path("/workspace/esg_contest/exp/agent_loop/claude/20260609T172829/loops/loops001/exp/A1/raw")
TMP_RAW_DIR = Path("/tmp/A1_raw_remaining")
GT_JSON = Path("/workspace/esg_contest/data/benchmarks/val_public.json")

ALLOWED_TIMELINES = {"already", "within_2_years", "between_2_and_5_years", "more_than_5_years"}


def safe_file_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized[:80] or "row"


def extract_prediction_value(row):
    for key in ("verification_timeline", "label", "prediction", "answer"):
        if key in row:
            return row[key]
    if "response" in row:
        response = row["response"]
        if isinstance(response, str):
            try:
                return extract_prediction_value(json.loads(response))
            except json.JSONDecodeError:
                return response
        if isinstance(response, dict):
            return extract_prediction_value(response)
    return ""


def normalize_timeline(value):
    if isinstance(value, dict):
        value = extract_prediction_value(value)
    text = str(value or "").strip()
    if not text:
        return "N/A", "missing_codex_prediction"
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict):
        return normalize_timeline(loaded)
    lowered = text.lower()
    aliases = [
        ("between_2_and_5_years", ("between_2_and_5_years", "between 2 and 5", "2-5", "三年", "四年", "五年", "中期")),
        ("more_than_5_years", ("more_than_5_years", "more_than_5", "more than 5", "more than five", "5年以上", "長期", "2030", "2035", "2040", "2050")),
        ("within_2_years", ("within_2_years", "within 2", "兩年", "二年", "短期", "近期")),
        ("already", ("already", "已完成", "已實施", "每年", "定期", "現行", "持續執行")),
    ]
    hits = [label for label, keys in aliases if any(key.lower() in lowered for key in keys)]
    if len(set(hits)) == 1:
        return hits[0], ""
    if text in ALLOWED_TIMELINES:
        return text, ""
    return "N/A", "invalid_codex_label"


def main():
    # Load output CSV
    rows = []
    with OUTPUT_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(dict(row))

    fixed = 0
    still_skipped = 0
    for idx, row in enumerate(rows, start=1):
        if row.get("stage4_error") != "error_skipped":
            continue
        rid = row["id"]
        # Try orig raw dir
        raw_path = ORIG_RAW_DIR / f"{idx:04d}_{safe_file_part(rid)}.json"
        if not raw_path.exists():
            # Try tmp dir
            raw_path = TMP_RAW_DIR / f"{idx:04d}_{safe_file_part(rid)}.json"
        if raw_path.exists() and raw_path.stat().st_size > 0:
            try:
                raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
                label, error = normalize_timeline(raw_data.get("raw_prediction", ""))
                row["verification_timeline"] = label
                row["stage4_raw_timeline"] = label
                row["stage4_error"] = error
                fixed += 1
            except Exception as e:
                print(f"  WARNING: failed to parse {raw_path}: {e}", file=sys.stderr)
                still_skipped += 1
        else:
            still_skipped += 1

    print(f"Fixed {fixed} error_skipped rows. Still skipped: {still_skipped}.")

    # Write fixed CSV
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote fixed CSV to {OUTPUT_CSV}")

    # Eval
    with GT_JSON.open() as f:
        gt_list = json.load(f)
    gt = {str(r["id"]): r for r in gt_list}

    preds = {r["id"]: r for r in rows}

    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for id_, gt_r in gt.items():
        g = gt_r.get("verification_timeline", "")
        p = preds.get(id_, {}).get("verification_timeline", "")
        if not g:
            continue
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    labels = sorted(set(list(tp) + list(fn)))
    f1s = []
    print("\nPer-class results:")
    for label in labels:
        pr = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] > 0 else 0
        rc = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] > 0 else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0
        print(f"  {label}: P={pr:.4f} R={rc:.4f} F1={f1:.4f}  (TP={tp[label]}, FP={fp[label]}, FN={fn[label]})")
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    print(f"\nMacro-F1 (ST4): {macro_f1:.4f}")
    return macro_f1


if __name__ == "__main__":
    main()
