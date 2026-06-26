"""Stratified train/val split of the train+val mix by `evidence_status`.

Splits a record list into a train part (out-a) and a val part (out-b), holding
out `--val-ratio` of EACH `evidence_status` class (Yes / No / N/A / "") to val so
both parts mirror the global distribution. Stratification is per class: each
class bucket is shuffled with a fixed seed and cut at the requested ratio.

Note (data-use): this only reads `evidence_status` to bucket records for
splitting. It is NOT used as a model/prompt/feature input. Each output record
keeps all original fields unchanged.

Usage
-----
    # default val_ratio 0.2 (train 0.8 / val 0.2), seed 42
    python exp/eval_train_in_gemma4_st12/split_by_evidence_status.py

    # custom ratio / key / inputs / outputs
    python exp/eval_train_in_gemma4_st12/split_by_evidence_status.py \
        --input  exp/eval_train_in_gemma4_st12/data/vpesg4k_train_val_mix_2000.json \
        --out-a  exp/eval_train_in_gemma4_st12/data/vpesg4k_train_val_mix_2000.train.json \
        --out-b  exp/eval_train_in_gemma4_st12/data/vpesg4k_train_val_mix_2000.val.json \
        --val-ratio 0.2 --key evidence_status --seed 42
"""

import argparse
import json
import pathlib
import random
from collections import Counter, defaultdict

_HERE = pathlib.Path(__file__).resolve().parent
_DEFAULT_INPUT = _HERE / "data" / "vpesg4k_train_val_mix_2000.json"
_DEFAULT_OUT_A = _HERE / "data" / "vpesg4k_train_val_mix_2000.train.json"
_DEFAULT_OUT_B = _HERE / "data" / "vpesg4k_train_val_mix_2000.val.json"


def stratified_split(records, key, val_ratio, seed):
    """Return (train_part, val_part), each preserving the per-`key` proportion.

    Records are bucketed by str(record[key]); each bucket is shuffled with `seed`
    and round(len * val_ratio) per class go to val, the rest to train.
    """
    buckets = defaultdict(list)
    for r in records:
        buckets[str(r.get(key, ""))].append(r)

    rng = random.Random(seed)
    train_part, val_part = [], []
    for cls in sorted(buckets):
        items = buckets[cls][:]
        rng.shuffle(items)
        n_val = round(len(items) * val_ratio)
        val_part.extend(items[:n_val])
        train_part.extend(items[n_val:])

    rng.shuffle(train_part)
    rng.shuffle(val_part)
    return train_part, val_part


def _dist(records, key):
    return dict(Counter(str(r.get(key, "")) for r in records))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(_DEFAULT_INPUT))
    ap.add_argument("--out-a", default=str(_DEFAULT_OUT_A))
    ap.add_argument("--out-b", default=str(_DEFAULT_OUT_B))
    ap.add_argument("--key", default="evidence_status",
                    help="field to stratify on (default: evidence_status)")
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="fraction of each class held out to val / out-b (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        ap.error("--val-ratio must be in (0, 1)")

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    train_part, val_part = stratified_split(records, args.key, args.val_ratio, args.seed)

    for path, part in ((args.out_a, train_part), (args.out_b, val_part)):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False, indent=2)

    n = len(records)
    print(f"[split] input={args.input} n={n} key={args.key} val_ratio={args.val_ratio} seed={args.seed}")
    print(f"[split] global dist : {_dist(records, args.key)}")
    print(f"[split] train n={len(train_part)} -> {args.out_a}")
    print(f"[split]   train dist : {_dist(train_part, args.key)}")
    print(f"[split] val   n={len(val_part)} -> {args.out_b}")
    print(f"[split]   val dist   : {_dist(val_part, args.key)}")


if __name__ == "__main__":
    main()
