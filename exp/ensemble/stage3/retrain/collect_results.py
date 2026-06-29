#!/usr/bin/env python3
"""Collect ST3 multitask task-weight sweep results into one sorted table.

Reads every ``*.json`` written by ``train_multitaskbert_stage3.py --output`` in
``--results-dir`` and prints a table sorted by best ST3 val Macro-F1, with the
ST1/ST2 val F1 at the same (best) epoch for context. Also writes a machine-
readable ``summary.json`` next to the inputs.

Usage:
    .venv/bin/python collect_results.py \\
      --results-dir exp/integrated_stage_predictions/0614/submit/submit_5/stage3/retrain/results
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def best_epoch_val_f1(rec: dict) -> dict:
    """Return the per-task val_f1 dict at rec['best_epoch'], or {} if unavailable."""
    be = rec.get("best_epoch")
    for em in rec.get("epoch_metrics", []):
        if em.get("epoch") == be:
            return em.get("val_f1", {}) or {}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="Directory of per-combo result JSONs.")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        if os.path.basename(path) == "summary.json":
            continue
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        tw = rec.get("task_weights", {})
        vf = best_epoch_val_f1(rec)
        rows.append({
            "tag": os.path.splitext(os.path.basename(path))[0],
            "task_weights": tw,
            "tw_str": ",".join(str(tw.get(t, "")) for t in ("st1", "st2", "st3")),
            "best_epoch": rec.get("best_epoch"),
            "best_val_st3_f1": rec.get("best_val_st3_f1"),
            "val_st1_f1": vf.get("st1"),
            "val_st2_f1": vf.get("st2"),
            "val_st3_f1": vf.get("st3"),
            "checkpoint": (rec.get("checkpoints") or {}).get("multitask"),
        })

    rows.sort(key=lambda r: (r["best_val_st3_f1"] is None, -(r["best_val_st3_f1"] or 0.0)))

    if not rows:
        print(f"[warn] no result JSONs found in {args.results_dir}")
        return

    hdr = f"{'task_weights(st1,st2,st3)':28s} {'epoch':>5s} {'ST3 F1':>8s} {'ST1 F1':>8s} {'ST2 F1':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def f(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) else "  nan "
        print(f"{r['tw_str']:28s} {str(r['best_epoch']):>5s} "
              f"{f(r['best_val_st3_f1']):>8s} {f(r['val_st1_f1']):>8s} {f(r['val_st2_f1']):>8s}")

    best = rows[0]
    print(f"\nBEST: task_weights={best['tw_str']}  ST3 val Macro-F1={best['best_val_st3_f1']}  "
          f"epoch={best['best_epoch']}\n      ckpt={best['checkpoint']}")

    out = os.path.join(args.results_dir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"sorted_by": "best_val_st3_f1", "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {out}")


if __name__ == "__main__":
    main()
