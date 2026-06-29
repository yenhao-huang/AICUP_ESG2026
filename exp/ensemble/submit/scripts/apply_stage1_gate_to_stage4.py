#!/usr/bin/env python3
"""Drift upstream N/A down into stage4 (verification_timeline).

Stage4 is only applicable when ST1 promise_status == "Yes". Per the cascade,
ST1 == "No" -> ST2/ST3/ST4 = N/A, so:

    ST1 promise_status == "No"  ->  verification_timeline = "N/A"

The stage4 codex handle predicts ALL 2000 rows independently (ungated, from
stage4/tmp/); this script reads the stage1 CSV and forces
verification_timeline = "N/A" for every id where ST1 == "No".

Usage
-----
    python .../submit_5/submit/scripts/apply_stage1_gate_to_stage4.py \
        --stage1 .../stage1/bert_focal_g3_w4.csv \
        --stage4 .../stage4/tmp/stage4_codex_predictions.csv \
        --output .../stage4/stage4_codex_gated.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]


def resolve(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True,
                        help="Stage1 CSV with id + promise_status columns.")
    parser.add_argument("--stage4", type=Path, required=True,
                        help="Stage4 CSV to patch (verification_timeline column).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path. Defaults to overwriting the stage4 CSV in place.")
    parser.add_argument("--stage1-col", default="promise_status")
    parser.add_argument("--stage4-col", default="verification_timeline")
    args = parser.parse_args()

    st1_path = resolve(args.stage1)
    st4_path = resolve(args.stage4)
    out_path = resolve(args.output) if args.output else st4_path

    no_ids: set[str] = set()
    with st1_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if str(row.get(args.stage1_col, "")).strip() == "No":
                no_ids.add(str(row["id"]).strip())
    print(f"Stage1 No ids : {len(no_ids)}", flush=True)

    with st4_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    n_patched = 0
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if rid in no_ids and str(row.get(args.stage4_col, "")).strip() != "N/A":
            row[args.stage4_col] = "N/A"
            n_patched += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Patched rows  : {n_patched}", flush=True)
    print(f"Output        : {out_path}", flush=True)

    remaining = sum(
        1 for r in rows
        if str(r.get("id", "")).strip() in no_ids
        and str(r.get(args.stage4_col, "")).strip() != "N/A"
    )
    if remaining:
        print(f"WARNING: {remaining} No-ids still not N/A after patch", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
