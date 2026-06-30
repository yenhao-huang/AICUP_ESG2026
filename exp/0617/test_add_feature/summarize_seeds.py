"""Aggregate multi-seed paired vanilla-vs-multitask runs and test for noise.

Reads results/seeds/train_{vanilla,multitask}_s{seed}.json (written by
train_st3_feature.py) and reports, per metric:
  - per-seed vanilla / multitask / paired delta (multitask - vanilla)
  - mean +- std of vanilla and multitask (the per-model seed spread = "noise band")
  - mean / median paired delta, and a sign test (#seeds where delta > 0)

Conclusion rule: the Method-A effect is "not just fluctuation" when the paired
delta is positive on a clear majority of seeds AND the mean paired delta exceeds
~0 by more than the within-model seed std is small enough to make the sign test
significant. We report the numbers; the README states the verdict.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st


def load_runs(d):
    runs = {}  # seed -> {mode -> metrics}
    for fp in glob.glob(os.path.join(d, "train_*_s*.json")):
        m = re.search(r"train_(vanilla|multitask)_s(\d+)\.json$", os.path.basename(fp))
        if not m:
            continue
        mode, seed = m.group(1), int(m.group(2))
        runs.setdefault(seed, {})[mode] = json.load(open(fp))
    return runs


def col(runs, seeds, mode, key):
    return [runs[s][mode][key] for s in seeds]


def fmt(xs):
    return f"{st.mean(xs):.4f}±{(st.pstdev(xs) if len(xs)>1 else 0.0):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "results", "seeds"))
    args = ap.parse_args()

    runs = load_runs(args.dir)
    seeds = sorted(s for s, mm in runs.items() if "vanilla" in mm and "multitask" in mm)
    if not seeds:
        print(f"no paired seed runs found under {args.dir}")
        return
    print(f"paired seeds (n={len(seeds)}): {seeds}\n")

    summary = {"seeds": seeds, "metrics": {}}
    for key, label in [("val_macro_f1", "macroF1(3c)"), ("val_nc_f1", "NC-F1")]:
        van = col(runs, seeds, "vanilla", key)
        mt = col(runs, seeds, "multitask", key)
        deltas = [m - v for m, v in zip(mt, van)]
        n_pos = sum(d > 0 for d in deltas)

        print(f"### {label}")
        print(f"{'seed':>6s} {'vanilla':>9s} {'multitask':>10s} {'Δ(mt-van)':>11s}")
        for s, v, m, dlt in zip(seeds, van, mt, deltas):
            print(f"{s:>6d} {v:9.4f} {m:10.4f} {dlt:+11.4f}")
        print(f"  vanilla   mean±std = {fmt(van)}   (per-seed spread = noise band)")
        print(f"  multitask mean±std = {fmt(mt)}")
        print(f"  paired Δ  mean = {st.mean(deltas):+.4f} | median = {st.median(deltas):+.4f} "
              f"| std = {(st.pstdev(deltas) if len(deltas)>1 else 0.0):.4f}")
        print(f"  sign test: {n_pos}/{len(seeds)} seeds have Δ>0\n")

        summary["metrics"][key] = {
            "label": label,
            "vanilla": dict(zip(map(str, seeds), van)),
            "multitask": dict(zip(map(str, seeds), mt)),
            "delta": dict(zip(map(str, seeds), deltas)),
            "vanilla_mean": st.mean(van), "vanilla_std": st.pstdev(van) if len(van) > 1 else 0.0,
            "multitask_mean": st.mean(mt), "multitask_std": st.pstdev(mt) if len(mt) > 1 else 0.0,
            "delta_mean": st.mean(deltas), "delta_median": st.median(deltas),
            "delta_std": st.pstdev(deltas) if len(deltas) > 1 else 0.0,
            "n_pos": n_pos, "n": len(seeds),
        }

    out = os.path.join(args.dir, "summary.json")
    json.dump(summary, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()
