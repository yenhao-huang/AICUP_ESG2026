#!/usr/bin/env python3
"""Stratified train/val split for an ESG ST3 JSON dataset, balanced by class.

Splits a list-of-records JSON file into a train file and a val file so that each
class of the label field (default ``evidence_quality`` for ST3) keeps the SAME
proportion in both splits ("公平地切開"). For ST3 the label space over the
applicable rows (``evidence_status == "Yes"``) is Clear / Not Clear / Misleading;
``Misleading`` is extremely scarce (only a couple of rows in VeriPromiseESG4K),
so a naive per-class ratio would put a fractional / rounded count into val and
make the held-out ST3 Macro-F1 (used to pick ``best_multitask_st3.pt``) jump
between 0 and 1 on a single row.

Misleading-in-val policy
------------------------
``--force-val-classes`` (default ``Misleading``) names label values that are
moved *entirely* into val instead of being stratified. With the default, every
``evidence_quality == "Misleading"`` row goes to val and none is seen in
training, so the ST3 head effectively trains on Clear / Not Clear and the
Misleading rows are kept purely as an evaluation signal. The remaining classes
(Clear, Not Clear, and the non-applicable ``""`` / ``N/A`` buckets whose
``evidence_status`` is not ``Yes`` and which ST3 training masks out anyway) are
stratified by ``--val-ratio`` so both splits stay representative.

Method
------
1. Pull every record whose label is in ``--force-val-classes`` straight into val.
2. Group the rest by label value (rows missing the label go to a ``__missing__``
   bucket and are split the same way, with a warning).
3. Shuffle each group with a fixed ``--seed`` (deterministic, reproducible).
4. Take ``round(n_class * --val-ratio)`` rows of each remaining class into val,
   the rest into train. Splitting per class guarantees the val ratio holds
   within every class, not just overall.
5. Write ``--train-out`` and ``--val-out`` (same record schema as input), and
   print the per-class distribution of each split.

Example
-------
    D=exp/integrated_stage_predictions/0614/submit/submit_5/stage3
    .venv/bin/python $D/split_train_val_by_class.py \\
      --input $D/data/vpesg_4k_train_1000_add_val.json \\
      --train-out $D/data/vpesg_4k_train_1000_add_val.train.json \\
      --val-out   $D/data/vpesg_4k_train_1000_add_val.val.json \\
      --label-key evidence_quality --force-val-classes Misleading \\
      --val-ratio 0.2 --seed 42
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


def stratified_split(records, label_key, val_ratio, seed, force_val_classes):
    """Return (train, val) lists, stratified by ``label_key``.

    Rows whose label is in ``force_val_classes`` are placed entirely in val and
    excluded from the per-class ratio split.
    """
    force_val = set(force_val_classes)
    forced, rest = [], []
    for r in records:
        if str(r.get(label_key, "__missing__")) in force_val:
            forced.append(r)
        else:
            rest.append(r)

    buckets = defaultdict(list)
    for r in rest:
        label = r.get(label_key, "__missing__")
        buckets[str(label)].append(r)

    if "__missing__" in buckets:
        sys.stderr.write(
            f"[warn] {len(buckets['__missing__'])} rows have no '{label_key}'; "
            "they are split as their own class.\n"
        )

    rng = random.Random(seed)
    train, val = [], list(forced)
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
        "--force-val-classes", default="Misleading",
        help="Comma-separated label values moved entirely into val (default 'Misleading'). "
             "Pass '' to disable and stratify every class.",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible shuffling (default 42).")
    args = ap.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        sys.exit(f"[error] --val-ratio must be in (0,1), got {args.val_ratio}")

    force_val_classes = [c.strip() for c in args.force_val_classes.split(",") if c.strip()]

    records = load_records(args.input)
    train, val = stratified_split(records, args.label_key, args.val_ratio, args.seed, force_val_classes)

    for path, rows in ((args.train_out, train), (args.val_out, val)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    total = len(records)
    if force_val_classes:
        print(f"force-val classes (all -> val): {force_val_classes}")
    print(f"input : {total} rows  dist={dist(records, args.label_key)}")
    print(f"train : {len(train)} rows ({len(train)/total:.1%})  dist={dist(train, args.label_key)} -> {args.train_out}")
    print(f"val   : {len(val)} rows ({len(val)/total:.1%})  dist={dist(val, args.label_key)} -> {args.val_out}")


if __name__ == "__main__":
    main()
