#!/usr/bin/env python3
"""Merge, gate, and build submit artifacts."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Callable


SCORE_RE = re.compile(r"(score_yes|score_no)=([0-9.]+)")
SUBMISSION_FIELDS = ["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("id", "")): row for row in rows}


def stage1_confidence(row: dict[str, str]) -> float:
    values = []
    for col in ("score_yes", "score_no"):
        try:
            values.append(float(row.get(col, "") or "nan"))
        except ValueError:
            pass
    return max(values) if values else 0.0


def stage2_confidence(row: dict[str, str]) -> float:
    scores = {}
    for key, value in SCORE_RE.findall(row.get("postprocess_reason", "")):
        try:
            scores[key] = float(value)
        except ValueError:
            continue
    if scores:
        return max(scores.values())
    return 0.0


def merge_low_confidence(
    *,
    bert_path: Path,
    gemma_path: Path,
    output_path: Path,
    threshold: float,
    confidence_fn: Callable[[dict[str, str]], float],
    run_id: str | None = None,
    bert_source: str | None = None,
) -> None:
    bert_fields, bert_rows = read_csv(bert_path)
    _, gemma_rows = read_csv(gemma_path)
    gemma_by_id = by_id(gemma_rows)

    merged: list[dict[str, str]] = []
    for row in bert_rows:
        row_id = str(row.get("id", ""))
        if confidence_fn(row) < threshold and row_id in gemma_by_id:
            merged.append(dict(gemma_by_id[row_id]))
            continue
        out = dict(row)
        if run_id is not None and "run_id" in out:
            out["run_id"] = run_id
        if bert_source is not None and "source" in out:
            out["source"] = bert_source
        merged.append(out)

    write_csv(output_path, bert_fields, merged)


def gate_stage2(stage1_path: Path, stage2_path: Path, output_path: Path) -> None:
    fields, stage2_rows = read_csv(stage2_path)
    _, stage1_rows = read_csv(stage1_path)
    stage1 = by_id(stage1_rows)
    out_rows = []
    for row in stage2_rows:
        out = dict(row)
        if stage1.get(str(row.get("id", "")), {}).get("promise_status") != "Yes":
            out["evidence_status"] = "N/A"
        out_rows.append(out)
    write_csv(output_path, fields, out_rows)


def gate_stage3(stage1_path: Path, stage2_path: Path, stage3_path: Path, output_path: Path) -> None:
    fields, stage3_rows = read_csv(stage3_path)
    _, stage1_rows = read_csv(stage1_path)
    _, stage2_rows = read_csv(stage2_path)
    stage1 = by_id(stage1_rows)
    stage2 = by_id(stage2_rows)
    out_rows = []
    for row in stage3_rows:
        row_id = str(row.get("id", ""))
        out = dict(row)
        if (
            stage1.get(row_id, {}).get("promise_status") != "Yes"
            or stage2.get(row_id, {}).get("evidence_status") != "Yes"
        ):
            out["evidence_quality"] = "N/A"
        out_rows.append(out)
    write_csv(output_path, fields, out_rows)


def gate_stage4(stage1_path: Path, stage4_path: Path, output_path: Path) -> None:
    fields, stage4_rows = read_csv(stage4_path)
    _, stage1_rows = read_csv(stage1_path)
    stage1 = by_id(stage1_rows)
    out_rows = []
    for row in stage4_rows:
        out = dict(row)
        if stage1.get(str(row.get("id", "")), {}).get("promise_status") != "Yes":
            out["verification_timeline"] = "N/A"
        out_rows.append(out)
    write_csv(output_path, fields, out_rows)


def build_submission(stage1_path: Path, stage2_path: Path, stage3_path: Path, stage4_path: Path, output_path: Path) -> None:
    _, stage1_rows = read_csv(stage1_path)
    _, stage2_rows = read_csv(stage2_path)
    _, stage3_rows = read_csv(stage3_path)
    _, stage4_rows = read_csv(stage4_path)
    stage1 = by_id(stage1_rows)
    stage2 = by_id(stage2_rows)
    stage3 = by_id(stage3_rows)
    stage4 = by_id(stage4_rows)

    ids = sorted(set(stage1) | set(stage2) | set(stage3) | set(stage4), key=lambda value: int(value) if value.isdigit() else value)
    rows = []
    for row_id in ids:
        rows.append(
            {
                "id": row_id,
                "promise_status": stage1.get(row_id, {}).get("promise_status", ""),
                "verification_timeline": stage4.get(row_id, {}).get("verification_timeline", ""),
                "evidence_status": stage2.get(row_id, {}).get("evidence_status", ""),
                "evidence_quality": stage3.get(row_id, {}).get("evidence_quality", ""),
            }
        )
    write_csv(output_path, SUBMISSION_FIELDS, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("merge-stage1")
    p.add_argument("--bert", required=True, type=Path)
    p.add_argument("--gemma", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--run-id")
    p.add_argument("--bert-source")

    p = sub.add_parser("merge-stage2")
    p.add_argument("--bert", required=True, type=Path)
    p.add_argument("--gemma", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=0.7)

    p = sub.add_parser("gate-stage2")
    p.add_argument("--stage1", required=True, type=Path)
    p.add_argument("--stage2", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)

    p = sub.add_parser("gate-stage3")
    p.add_argument("--stage1", required=True, type=Path)
    p.add_argument("--stage2", required=True, type=Path)
    p.add_argument("--stage3", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)

    p = sub.add_parser("gate-stage4")
    p.add_argument("--stage1", required=True, type=Path)
    p.add_argument("--stage4", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)

    p = sub.add_parser("build-submission")
    p.add_argument("--stage1", required=True, type=Path)
    p.add_argument("--stage2", required=True, type=Path)
    p.add_argument("--stage3", required=True, type=Path)
    p.add_argument("--stage4", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "merge-stage1":
        merge_low_confidence(
            bert_path=args.bert,
            gemma_path=args.gemma,
            output_path=args.output,
            threshold=args.threshold,
            confidence_fn=stage1_confidence,
            run_id=args.run_id,
            bert_source=args.bert_source,
        )
    elif args.command == "merge-stage2":
        merge_low_confidence(
            bert_path=args.bert,
            gemma_path=args.gemma,
            output_path=args.output,
            threshold=args.threshold,
            confidence_fn=stage2_confidence,
        )
    elif args.command == "gate-stage2":
        gate_stage2(args.stage1, args.stage2, args.output)
    elif args.command == "gate-stage3":
        gate_stage3(args.stage1, args.stage2, args.stage3, args.output)
    elif args.command == "gate-stage4":
        gate_stage4(args.stage1, args.stage4, args.output)
    elif args.command == "build-submission":
        build_submission(args.stage1, args.stage2, args.stage3, args.stage4, args.output)


if __name__ == "__main__":
    main()
