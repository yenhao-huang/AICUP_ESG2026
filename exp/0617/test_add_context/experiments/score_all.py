#!/usr/bin/env python3
"""Score every exp_*.csv in a preds dir against a benchmark JSON and rank by the
GT-gated 2-class (Clear/Not Clear) Macro-F1 — the metric the context ablations move
(the 2-class prompt never emits N/A, so full-coverage F1 is dominated by N/A rows).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
for p in ROOT.parents:
    if (p / "core" / "analysis" / "score_st3_full_coverage.py").exists():
        sys.path.insert(0, str(p))
        break
from core.analysis.score_st3_full_coverage import load_benchmark, load_predictions, score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--preds-dir", type=Path, required=True)
    ap.add_argument("--pattern", default="exp_*.csv")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    gt = load_benchmark(args.benchmark)
    rows = []
    for csv_path in sorted(glob.glob(str(args.preds_dir / args.pattern))):
        name = Path(csv_path).stem
        for pfx in ("exp_", "full_", "pp_", "win_", "m_"):
            if name.startswith(pfx):
                name = name[len(pfx):]
        s = score(gt, load_predictions(Path(csv_path)))
        rows.append({
            "method": name,
            "gtgated_2cls_macro_f1": round(s["gtgated_2cls_macro_f1"], 4),
            "full_coverage_macro_f1": round(s["full_coverage_macro_f1"], 4),
            "f1_clear": round(s["per_class_f1"]["Clear"], 4),
            "f1_not_clear": round(s["per_class_f1"]["Not Clear"], 4),
            "n_2cls": s["n_gtgated_2cls"],
            "missing_pred": s["missing_pred"],
        })
    rows.sort(key=lambda r: r["gtgated_2cls_macro_f1"], reverse=True)

    print(f"benchmark: {args.benchmark}  ({len(gt)} rows)")
    print(f"{'rank':<5}{'method':<24}{'2cls_F1':<10}{'full_F1':<10}{'F1_Clear':<10}{'F1_NotClr':<11}{'n2':<5}{'miss':<5}")
    for i, r in enumerate(rows, 1):
        print(f"{i:<5}{r['method']:<24}{r['gtgated_2cls_macro_f1']:<10}{r['full_coverage_macro_f1']:<10}"
              f"{r['f1_clear']:<10}{r['f1_not_clear']:<11}{r['n_2cls']:<5}{r['missing_pred']:<5}")
    top3 = [r["method"] for r in rows[:3]]
    print("\nTOP3:", " ".join(top3))
    if args.output:
        args.output.write_text(json.dumps({"ranking": rows, "top3": top3}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
