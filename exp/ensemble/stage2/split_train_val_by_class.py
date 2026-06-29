#!/usr/bin/env python3
"""Stratified train/val split for an ESG ST2 JSON dataset, balanced by class.

Splits a list-of-records JSON file into a train file and a val file so that each
class of the label field (default ``evidence_status`` for ST2) keeps the SAME
proportion in both splits ("公平地切開"). For ST2 the scarce class is the
``No``-evidence label; a naive random split can starve it in val and make the
held-out Macro-F1 (used to pick ``best_st2.pt``) noisy. Per-class slicing keeps
both splits representative of every label, including the ``N/A`` rows whose
``promise_status`` is ``No`` and which ST2 training filters out anyway.

Why ``evidence_status`` and not ``promise_status``
--------------------------------------------------
ST2 only trains on ``promise_status == "Yes"`` rows and predicts
``evidence_status`` (Yes/No). Stratifying on ``evidence_status`` is what keeps
the Yes/No balance stable across the train/val split. ``N/A`` rows (promise==No)
form their own bucket and are split with the same ratio, so the file stays a
faithful superset usable by ``train_bert.py --stage st2``.

Method
------
1. Group every record by its label value (rows missing the label go to a
   separate ``__missing__`` bucket and are split the same way, with a warning).
2. Shuffle each group with a fixed ``--seed`` (deterministic, reproducible).
3. Take ``round(n_class * --val-ratio)`` rows of each class into val, the rest
   into train. Splitting per class guarantees the val ratio holds within every
   class, not just overall.
4. Write ``--train-out`` and ``--val-out`` (same record schema as input), and
   print the per-class distribution of each split.

Example
-------
    D=exp/integrated_stage_predictions/0614/submit/submit_5/stage2
    .venv/bin/python $D/split_train_val_by_class.py \\
      --input $D/data/mix_a2_b3_add_val.json \\
      --train-out $D/data/mix_a2_b3_add_val.train.json \\
      --val-out   $D/data/mix_a2_b3_add_val.val.json \\
      --label-key evidence_status --val-ratio 0.2 --seed 42
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
    ap.add_argument("--label-key", default="evidence_status", help="Field to stratify on (default evidence_status for ST2).")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible shuffling (default 42).")
    args = ap.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        sys.exit(f"[error] --val-ratio must be in (0,1), got {args.val_ratio}")

    records = load_records(args.input)
    train, val = stratified_split(records, args.label_key, args.val_ratio, args.seed)

    for path, rows in ((args.train_out, train), (args.val_out, val)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    total = len(records)
    print(f"input : {total} rows  dist={dist(records, args.label_key)}")
    print(f"train : {len(train)} rows ({len(train)/total:.1%})  dist={dist(train, args.label_key)} -> {args.train_out}")
    print(f"val   : {len(val)} rows ({len(val)/total:.1%})  dist={dist(val, args.label_key)} -> {args.val_out}")


if __name__ == "__main__":
    main()
