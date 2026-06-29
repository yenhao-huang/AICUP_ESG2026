#!/usr/bin/env python3
"""Analyse per-member confidence & disagreement from a soft-vote members.csv.

members.csv schema: id, member, <one score column per class>, pred
Works for binary (ST1/ST2: score_no/score_yes) and multiclass (ST3:
Clear/Not Clear/Misleading). Confidence of a row = the probability of the
predicted class = max over the score columns.

Usage:
  python analyze_members.py --members <members.csv> [--label STAGE_NAME]
"""
from __future__ import annotations
import argparse, csv, itertools, statistics as st
from collections import defaultdict, Counter


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** .5
    db = sum((y - mb) ** 2 for y in b) ** .5
    return cov / (da * db) if da and db else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.members)))
    cols = rows[0].keys()
    score_cols = [c for c in cols if c not in ("id", "member", "pred")]
    by = defaultdict(list)            # member -> [(id, {cls:score}, pred)]
    for r in rows:
        m = r["member"]
        sc = {c: float(r[c]) for c in score_cols}
        by[m].append((r["id"], sc, r["pred"]))
    short = {m: m.split("seed")[-1] and ("seed" + m.split("seed")[-1]) or m for m in by}

    n_per = len(next(iter(by.values())))
    print(f"\n===== {args.label or args.members} =====")
    print(f"members={len(by)}  rows/member={n_per}  classes={score_cols}")

    def pct(xs, f): return 100 * sum(1 for x in xs if f(x)) / len(xs)
    print(f"\n{'model':<10} {'meanConf':>9} {'medConf':>8} {'std':>6} {'≥.9':>6} {'.7-.9':>6} {'<.7':>6}  dist")
    for m in sorted(by):
        recs = by[m]
        conf = [max(sc.values()) for _, sc, _ in recs]
        dist = dict(sorted(Counter(p for *_, p in recs).items()))
        print(f"{short[m]:<10} {st.mean(conf):>9.3f} {st.median(conf):>8.3f} {st.pstdev(conf):>6.3f} "
              f"{pct(conf, lambda x: x>=.9):>5.1f}% {pct(conf, lambda x: .7<=x<.9):>5.1f}% "
              f"{pct(conf, lambda x: x<.7):>5.1f}%  {dist}")

    # disagreement
    ids = [i for i, _, _ in next(iter(by.values()))]
    preds_by_id = defaultdict(list)
    for m in by:
        for i, _, p in by[m]:
            preds_by_id[i].append(p)
    distinct = Counter(len(set(preds_by_id[i])) for i in ids)
    agree = distinct[1]
    print(f"\n全員一致: {agree}/{len(ids)} ({100*agree/len(ids):.1f}%)  分歧: {len(ids)-agree} ({100*(len(ids)-agree)/len(ids):.1f}%)")
    print("分歧程度(一列出現幾種不同標籤): " +
          "  ".join(f"{k}種={v}" for k, v in sorted(distinct.items())))

    # binary vote split
    if set(score_cols) == {"score_no", "score_yes"}:
        split = Counter()
        for i in ids:
            y = sum(1 for p in preds_by_id[i] if p == "Yes")
            split[(y, len(preds_by_id[i]) - y)] += 1
        print("Yes:No 票數分布: " + "  ".join(f"{a}:{b}={split[(a,b)]}" for a, b in
              sorted(split, key=lambda k: -k[0])))

    # pairwise corr on predicted-positive prob (binary) / on each class avg (else skip)
    if "score_yes" in score_cols:
        pj = {m: {i: dict(sc)["score_yes"] for i, sc, _ in by[m]} for m in by}
        print("\nP(Yes) 兩兩相關:")
        for a, b in itertools.combinations(sorted(by), 2):
            va = [pj[a][i] for i in ids]; vb = [pj[b][i] for i in ids]
            print(f"  {short[a]:<9} vs {short[b]:<9}: {corr(va, vb):.3f}")


if __name__ == "__main__":
    main()
