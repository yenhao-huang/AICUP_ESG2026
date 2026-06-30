#!/usr/bin/env python3
"""Collect Stage 4 variant *.score.json files into one comparison table.

Reads preds/codex/<variant>_<set>_<backend>.score.json for the standard variant
set and prints a Macro-F1 / accuracy / per-class table with deltas vs vanilla.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = ["vanilla", "add_context", "add_page_abstract", "add_promise", "add_image", "all"]
CLASSES = ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="100")
    ap.add_argument("--backend", default="codex")
    args = ap.parse_args()

    preds = Path(__file__).resolve().parents[1] / "preds" / "codex"
    rows = {}
    for v in VARIANTS:
        f = preds / f"{v}_{args.set}_{args.backend}.score.json"
        if f.exists():
            rows[v] = json.loads(f.read_text(encoding="utf-8"))

    if not rows:
        print("(no score.json files found yet)")
        return

    base = rows.get("vanilla", {}).get("macro_f1")
    hdr = f"{'variant':<14}{'macroF1':>9}{'Δvanilla':>10}{'acc':>8}  " + "".join(f"{c[:10]:>12}" for c in CLASSES)
    print(hdr)
    print("-" * len(hdr))
    for v in VARIANTS:
        r = rows.get(v)
        if not r:
            continue
        mf = r["macro_f1"]
        d = f"{mf - base:+.4f}" if base is not None else "   n/a"
        pc = r.get("per_class_f1", {})
        line = f"{v:<14}{mf:>9.4f}{d:>10}{r['accuracy']:>8.4f}  " + "".join(f"{pc.get(c, 0):>12.4f}" for c in CLASSES)
        print(line)
    print()
    for v in VARIANTS:
        r = rows.get(v)
        if r and (r.get("missing_pred") or r.get("invalid_pred")):
            print(f"  note {v}: missing={r['missing_pred']} invalid={r['invalid_pred']} n_scored={r['n_scored']}")


if __name__ == "__main__":
    main()
