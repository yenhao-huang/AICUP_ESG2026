#!/usr/bin/env python3
"""Score ST3 multitask predictions against a val set: accuracy + Macro-F1.

Reproduces the training selection metric ``best_val_st3_f1`` for the multitask
ST3 head: a FIXED 3-class {Clear, Not Clear, Misleading} Macro-F1 computed over
the subset where ground-truth ``evidence_quality`` is one of those 3 classes.

Rationale: the multitask ST3 head only emits {Clear, Not Clear, Misleading};
rows whose GT is ``N/A`` or blank are the cascade gate's responsibility
(ST1=No -> N/A, ST2=No -> ""), not the ST3 head's, so they are excluded from the
head's own accuracy/F1. The full-402 accuracy is also reported for context.

GT ``evidence_quality`` is used ONLY for offline scoring here, never as model
input (the prediction CSV was produced data-only from the ``data`` field).

Usage:
    python score_st3_subset.py --gold <val.json> --pred <pred.csv> [--output <metrics.json>]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

LABELS = ["Clear", "Not Clear", "Misleading"]


def load_gold(p: Path) -> dict[str, str]:
    return {str(r.get("id")): (r.get("evidence_quality") or "") for r in json.load(open(p))}


def load_pred(p: Path) -> dict[str, str]:
    return {r["id"]: r["evidence_quality"] for r in csv.DictReader(open(p))}


def load_conf(p: Path) -> dict[str, float]:
    """Winning-class confidence = max(softmax) parsed from evidence_quality_reason."""
    out: dict[str, float] = {}
    for r in csv.DictReader(open(p)):
        reason = r.get("evidence_quality_reason", "") or ""
        try:
            d = dict(kv.split("=") for kv in reason.split(";"))
            out[r["id"]] = max(float(v) for v in d.values())
        except Exception:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True, type=Path, help="Val JSON with id + evidence_quality GT.")
    ap.add_argument("--pred", required=True, type=Path, help="Stage3 prediction CSV.")
    ap.add_argument("--output", type=Path, default=None, help="Optional metrics JSON output.")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    pred = load_pred(args.pred)
    conf = load_conf(args.pred)

    # 純 3 類有效子集 (模型負責範圍)
    sub_ids = [i for i in gold if gold[i] in LABELS]
    y = [gold[i] for i in sub_ids]
    yp = [pred.get(i, "") for i in sub_ids]

    acc = accuracy_score(y, yp)
    macro = f1_score(y, yp, labels=LABELS, average="macro", zero_division=0)
    per = f1_score(y, yp, labels=LABELS, average=None, zero_division=0)

    # 對照: 全 402 / 排除 N/A 的 accuracy
    all_ids = list(gold)
    acc_all = accuracy_score([gold[i] for i in all_ids], [pred.get(i, "") for i in all_ids])
    no_na = [i for i in gold if gold[i] != "N/A"]
    acc_no_na = accuracy_score([gold[i] for i in no_na], [pred.get(i, "") for i in no_na])

    res = {
        "pred": str(args.pred),
        "n_total": len(gold),
        "n_eval_3cls_subset": len(sub_ids),
        "accuracy_3cls_subset": round(acc, 4),
        "macro_f1_3cls": round(macro, 4),
        "per_class_f1": {k: round(float(v), 4) for k, v in zip(LABELS, per)},
        "accuracy_all_402": round(acc_all, 4),
        "accuracy_exclude_na": round(acc_no_na, 4),
    }

    print("=" * 60)
    print(f"pred: {args.pred.name}")
    print(f"純3類有效子集 (n={len(sub_ids)}):")
    print(f"  Accuracy   = {acc*100:.1f}%  ({sum(a==b for a,b in zip(y,yp))}/{len(sub_ids)})")
    print(f"  Macro-F1   = {macro:.4f}   (固定3類 Clear/Not Clear/Misleading)")
    for k, v in zip(LABELS, per):
        print(f"    {k:<11} F1={v:.4f}")
    print(f"對照 accuracy: 全402={acc_all*100:.1f}%  排除N/A={acc_no_na*100:.1f}%")

    # 信心區間 accuracy (純3類子集, 累積) + 佔總預測比例 (分母=全部預測數)
    if conf:
        total_pred = len(conf)  # 模型實際做的預測總數 (裸模型 = 全 402)
        print(f"信心區間 accuracy (純3類子集 acc, 累積) | 佔總預測 (n_all/{total_pred}):")
        print(f"    {'threshold':<8} {'子集n':>5} {'子集acc':>8}   {'全部n':>6} {'佔總預測':>8}")
        for tag, lo, hi in [("< 0.6", 0, 0.6), ("< 0.7", 0, 0.7), ("< 0.8", 0, 0.8),
                            ("< 0.9", 0, 0.9), (">= 0.9", 0.9, 2.0)]:
            sel = [i for i in sub_ids if lo <= conf.get(i, -1) < hi]
            n_all = sum(1 for v in conf.values() if lo <= v < hi)
            pct_all = n_all / total_pred if total_pred else 0
            if sel:
                c = sum(gold[i] == pred.get(i, "") for i in sel)
                print(f"    {tag:<8} {len(sel):>5} {c/len(sel)*100:>7.1f}%   "
                      f"{n_all:>6} {pct_all*100:>7.1f}%")
        res["confidence_bins"] = {
            tag: {"n_all": sum(1 for v in conf.values() if lo <= v < hi),
                  "pct_total_pred": round(sum(1 for v in conf.values() if lo <= v < hi) / total_pred, 4)}
            for tag, lo, hi in [("lt_0.6", 0, 0.6), ("lt_0.7", 0, 0.7), ("lt_0.8", 0, 0.8),
                                ("lt_0.9", 0, 0.9), ("ge_0.9", 0.9, 2.0)]
        }
    print("=" * 60)

    if args.output:
        args.output.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"metrics -> {args.output}")


if __name__ == "__main__":
    main()
