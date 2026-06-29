#!/usr/bin/env python3
"""Drift upstream N/A down into stage3 (evidence_quality).

Stage3 is only applicable when the cascade allows it:

    ST1 promise_status == "Yes"  AND  ST2 evidence_status == "Yes"

Otherwise stage3 must be "N/A":
    - ST1 promise_status == "No"      -> ST2/ST3/ST4 = N/A  -> ST3 = "N/A"
    - ST1 == "Yes" but ST2 != "Yes"   -> ST3 not applicable -> ST3 = "N/A"

The stage3 codex handle predicts ALL 2000 rows independently (ungated, from
stage3/tmp/); this script reads the stage1 + stage2 CSVs and forces
evidence_quality = "N/A" for every id gated out by either upstream stage.

Usage
-----
    python .../submit_5/submit/scripts/apply_stage12_gate_to_stage3.py \
        --stage1 .../stage1/bert_focal_g3_w4.csv \
        --stage2 .../stage2/bert.csv \
        --stage3 .../stage3/tmp/stage3_codex_predictions_merge.csv \
        --output .../stage3/stage3_codex_gated.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def resolve(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def read_ids(path: Path, col: str, keep) -> set[str]:
    ids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if keep(str(row.get(col, "")).strip()):
                ids.add(str(row["id"]).strip())
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True,
                        help="Stage1 CSV with id + promise_status columns.")
    parser.add_argument("--stage2", type=Path, required=True,
                        help="Stage2 CSV with id + evidence_status columns.")
    parser.add_argument("--stage3", type=Path, required=True,
                        help="Stage3 CSV to patch (evidence_quality column).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path. Defaults to overwriting the stage3 CSV in place.")
    parser.add_argument("--stage1-col", default="promise_status")
    parser.add_argument("--stage2-col", default="evidence_status")
    parser.add_argument("--stage3-col", default="evidence_quality")
    args = parser.parse_args()

    st1_path = resolve(args.stage1)
    st2_path = resolve(args.stage2)
    st3_path = resolve(args.stage3)
    out_path = resolve(args.output) if args.output else st3_path

    st1_no = read_ids(st1_path, args.stage1_col, lambda v: v == "No")
    st2_not_yes = read_ids(st2_path, args.stage2_col, lambda v: v != "Yes")
    gated_ids = st1_no | st2_not_yes
    print(f"Stage1 No ids      : {len(st1_no)}", flush=True)
    print(f"Stage2 non-Yes ids : {len(st2_not_yes)}", flush=True)
    print(f"Gated ids (union)  : {len(gated_ids)}", flush=True)

    with st3_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    n_patched = 0
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if rid in gated_ids and str(row.get(args.stage3_col, "")).strip() != "N/A":
            row[args.stage3_col] = "N/A"
            n_patched += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Patched rows       : {n_patched}", flush=True)
    print(f"Output             : {out_path}", flush=True)

    remaining = sum(
        1 for r in rows
        if str(r.get("id", "")).strip() in gated_ids
        and str(r.get(args.stage3_col, "")).strip() != "N/A"
    )
    if remaining:
        print(f"WARNING: {remaining} gated ids still not N/A after patch", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
