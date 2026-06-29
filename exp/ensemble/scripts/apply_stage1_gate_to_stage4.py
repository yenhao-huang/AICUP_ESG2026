#!/usr/bin/env python3
"""Apply stage1 gate to stage4 predictions.

For every row where stage1 promise_status == "No",
set verification_timeline = "N/A" in the stage4 CSV.

Usage
-----
    python exp/exp41/apply_stage1_gate_to_stage4.py \
        --stage1 exp/exp41/predictions/stage1/bert_opt1.csv \
        --stage4 exp/exp41/predictions/stage4/stage4_codex_predictions.csv \
        --output exp/exp41/predictions/stage4/stage4_codex_gated.csv

    # overwrite in place
    python exp/exp41/apply_stage1_gate_to_stage4.py \
        --stage1 exp/exp41/predictions/stage1/bert_opt1.csv \
        --stage4 exp/exp41/predictions/stage4/stage4_codex_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def resolve(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path,
                        default=Path("exp/exp41/submit/submit_2/stage1/bert_opt1.csv"),
                        help="Stage1 CSV with id + promise_status columns.")
    parser.add_argument("--stage4", type=Path,
                        default=Path("exp/exp41/predictions/stage4/stage4_codex_predictions.csv"),
                        help="Stage4 CSV to patch (verification_timeline column).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path. Defaults to exp/exp41/submit/submit_2/stage4/<stage4 filename>.")
    parser.add_argument("--stage1-col", default="promise_status")
    parser.add_argument("--stage4-col", default="verification_timeline")
    args = parser.parse_args()

    st1_path = resolve(args.stage1)
    st4_path = resolve(args.stage4)
    default_out = Path("exp/exp41/submit/submit_2/stage4") / st4_path.name
    out_path = resolve(args.output) if args.output else resolve(default_out)

    # Build set of ids where stage1 == No
    no_ids: set[str] = set()
    with st1_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if str(row.get(args.stage1_col, "")).strip() == "No":
                no_ids.add(str(row["id"]).strip())

    print(f"Stage1 No ids : {len(no_ids)}", flush=True)

    # Patch stage4
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

    # Sanity check
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
