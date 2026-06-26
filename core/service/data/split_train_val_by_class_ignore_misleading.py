#!/usr/bin/env python3
"""Stratified train/val split for an ESG Stage 3 JSON dataset.

This variant removes rows whose label matches ``--ignore-labels`` before
splitting. Stage 3 has very few ``evidence_quality == "Misleading"`` rows, so
the default behavior is to ignore those examples and build train/val splits from
the remaining labels.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict


def load_records(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"[error] {path} is not a JSON list of records (got {type(data).__name__})")
    return data


def filter_records(records, label_key, ignore_labels):
    ignore = set(ignore_labels)
    kept, ignored = [], []
    for record in records:
        if str(record.get(label_key, "__missing__")) in ignore:
            ignored.append(record)
        else:
            kept.append(record)
    return kept, ignored


def stratified_split(records, label_key, val_ratio, seed):
    """Return (train, val) lists, stratified by ``label_key``."""
    buckets = defaultdict(list)
    for r in records:
        label = r.get(label_key, "__missing__")
        buckets[str(label)].append(r)

    if "__missing__" in buckets:
        sys.stderr.write(
            f"[warn] {len(buckets['__missing__'])} rows have no '{label_key}'; "
            "they are split as their own class.\n"
        )

    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(buckets):
        rows = buckets[label][:]
        rng.shuffle(rows)
        n_val = round(len(rows) * val_ratio)
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])

    # shuffle the merged splits so classes are interleaved (order-independent
    # downstream, but tidier for inspection / batching)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def dist(records, label_key):
    return dict(sorted(Counter(str(r.get(label_key, "__missing__")) for r in records).items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Input JSON (list of records).")
    ap.add_argument("--train-out", required=True, help="Output path for the train split.")
    ap.add_argument("--val-out", required=True, help="Output path for the val split.")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Per-class fraction held out for val (default 0.2).")
    ap.add_argument("--label-key", default="evidence_quality", help="Field to stratify on (default evidence_quality for ST3).")
    ap.add_argument(
        "--ignore-labels",
        default="Misleading",
        help="Comma-separated label values excluded before splitting (default 'Misleading'). Pass '' to disable.",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible shuffling (default 42).")
    args = ap.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        sys.exit(f"[error] --val-ratio must be in (0,1), got {args.val_ratio}")

    ignore_labels = [c.strip() for c in args.ignore_labels.split(",") if c.strip()]

    records = load_records(args.input)
    kept, ignored = filter_records(records, args.label_key, ignore_labels)
    if not kept:
        sys.exit(f"[error] no records left after ignoring labels: {ignore_labels}")

    train, val = stratified_split(kept, args.label_key, args.val_ratio, args.seed)

    for path, rows in ((args.train_out, train), (args.val_out, val)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    total = len(records)
    kept_total = len(kept)
    if ignore_labels:
        print(f"ignore labels: {ignore_labels}")
    print(f"input  : {total} rows  dist={dist(records, args.label_key)}")
    print(f"ignored: {len(ignored)} rows ({len(ignored)/total:.1%})  dist={dist(ignored, args.label_key)}")
    print(f"kept   : {kept_total} rows ({kept_total/total:.1%})  dist={dist(kept, args.label_key)}")
    print(f"train  : {len(train)} rows ({len(train)/kept_total:.1%} of kept)  dist={dist(train, args.label_key)} -> {args.train_out}")
    print(f"val    : {len(val)} rows ({len(val)/kept_total:.1%} of kept)  dist={dist(val, args.label_key)} -> {args.val_out}")


if __name__ == "__main__":
    main()
