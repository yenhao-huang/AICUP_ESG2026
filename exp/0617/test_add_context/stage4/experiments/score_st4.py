#!/usr/bin/env python3
"""Stage 4 verification_timeline Macro-F1 scorer for the test_add_context probe.

Compares a prediction CSV (id + verification_timeline) against a benchmark JSON
holding GT `verification_timeline`. The benchmark here is the Yes-gated promise
subset (val_yes*.json), so every row is a promise and the live label space is
the four timeline classes:

    already / within_2_years / between_2_and_5_years / more_than_5_years

Full coverage: every benchmark row is scored. A missing or invalid prediction is
folded to a never-correct sentinel so it counts against the score rather than
being silently dropped. GT labels are used ONLY for this offline scoring, never
as model input.

Usage:
    python score_st4.py --benchmark ../data/val_yes.100.json --pred ../preds/codex/vanilla_100_codex.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import f1_score

LABELS = ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years"]
_MISSING = "__missing__"


def load_benchmark(path: Path) -> dict[str, str]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(r.get("id", "")): str(r.get("verification_timeline", "")).strip() for r in rows}


def load_predictions(path: Path) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return {str(r.get("id", "")): str(r.get("verification_timeline", "")).strip()
                for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--output", type=Path, help="Optional JSON metrics dump.")
    args = ap.parse_args()

    gt = load_benchmark(args.benchmark)
    pred = load_predictions(args.pred)

    y_true: list[str] = []
    y_pred: list[str] = []
    missing = 0
    invalid = 0
    for rid, g in gt.items():
        if g not in LABELS:
            continue  # skip rows whose GT is outside the 4 live classes (e.g. N/A)
        p = pred.get(rid)
        if p is None:
            missing += 1
            p = _MISSING
        elif p not in LABELS:
            invalid += 1
            p = _MISSING
        y_true.append(g)
        y_pred.append(p)

    macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    per = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0

    result = {
        "benchmark": str(args.benchmark),
        "pred": str(args.pred),
        "n_scored": len(y_true),
        "missing_pred": missing,
        "invalid_pred": invalid,
        "macro_f1": round(macro, 4),
        "accuracy": round(acc, 4),
        "per_class_f1": {lab: round(float(f), 4) for lab, f in zip(LABELS, per)},
        "gt_dist": dict(Counter(y_true)),
        "pred_dist": dict(Counter(y_pred)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
