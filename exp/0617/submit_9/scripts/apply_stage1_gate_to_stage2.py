#!/usr/bin/env python3
"""Drift upstream N/A down into stage2 (evidence_status).

Stage2 is only applicable when ST1 promise_status == "Yes". Per the cascade,
ST1 == "No" -> ST2/ST3/ST4 = N/A, so:

    ST1 promise_status == "No"  ->  evidence_status = "N/A"

The stage2 soft-vote handle predicts a 2-class head (Yes/No) for ALL rows
(ungated); this script reads the stage1 CSV and forces evidence_status = "N/A"
for every id where ST1 == "No".

Usage
-----
    python .../submit_7/scripts/apply_stage1_gate_to_stage2.py \
        --stage1 .../stage1/softvote.csv \
        --stage2 .../stage2/tmp/softvote_raw.csv \
        --output .../stage2/softvote_gated.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def resolve(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True,
                        help="Stage1 CSV with id + promise_status columns.")
    parser.add_argument("--stage2", type=Path, required=True,
                        help="Stage2 CSV to patch (evidence_status column).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path. Defaults to overwriting the stage2 CSV in place.")
    parser.add_argument("--stage1-col", default="promise_status")
    parser.add_argument("--stage2-col", default="evidence_status")
    args = parser.parse_args()

    st1_path = resolve(args.stage1)
    st2_path = resolve(args.stage2)
    out_path = resolve(args.output) if args.output else st2_path

    no_ids: set[str] = set()
    with st1_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if str(row.get(args.stage1_col, "")).strip() == "No":
                no_ids.add(str(row["id"]).strip())
    print(f"Stage1 No ids : {len(no_ids)}", flush=True)

    with st2_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    n_patched = 0
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if rid in no_ids and str(row.get(args.stage2_col, "")).strip() != "N/A":
            row[args.stage2_col] = "N/A"
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
        and str(r.get(args.stage2_col, "")).strip() != "N/A"
    )
    if remaining:
        print(f"WARNING: {remaining} No-ids still not N/A after patch", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
