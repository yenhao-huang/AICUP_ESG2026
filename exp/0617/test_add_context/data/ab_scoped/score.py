#!/usr/bin/env python3
"""Score the current-vs-scoped prompt A/B on the 100 gated val rows.

Reads the codex raw dumps under raw_cur/ and raw_scoped/, normalizes each to a
label, and reports the Clear/Not Clear distribution plus per-class F1, macro-F1,
and accuracy against the GT in val_ctxhit.json. Run after the A/B shards finish.

Usage:
  /workspace/esg_contest/.venv/bin/python \
    exp/integrated_stage_predictions/0616/test_add_context/stage3/data/ab_scoped/score.py
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(ROOT))

from core.human.predict.stage3.pred_by_codex import normalize_evidence_quality  # noqa: E402

HERE = Path(__file__).resolve().parent
VAL = HERE.parent / "val_ctxhit.json"            # ../val_ctxhit.json (GT + ids)
SUBSET = HERE / "val100.json"                    # the 100 gated rows scored

gold = {str(r["id"]): r["evidence_quality"] for r in json.loads(VAL.read_text(encoding="utf-8"))}
sub_ids = [str(r["id"]) for r in json.loads(SUBSET.read_text(encoding="utf-8"))]


def load(raw_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in glob.glob(str(raw_dir / "shard_*" / "*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        label, _ = normalize_evidence_quality(r.get("raw_prediction", ""))
        out[str(r["id"])] = label
    return out


def report(pred: dict[str, str], name: str) -> None:
    ids = [i for i in sub_ids if i in pred]
    dist = Counter(pred[i] for i in ids)

    def f1(label: str):
        tp = sum(1 for i in ids if gold[i] == label and pred[i] == label)
        fp = sum(1 for i in ids if gold[i] != label and pred[i] == label)
        fn = sum(1 for i in ids if gold[i] == label and pred[i] != label)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        return (2 * P * R / (P + R) if P + R else 0.0), P, R

    acc = sum(1 for i in ids if gold[i] == pred[i]) / len(ids) if ids else 0.0
    fc, fn = f1("Clear"), f1("Not Clear")
    macro = (fc[0] + fn[0]) / 2
    print(f"[{name}] n={len(ids)}  pred dist={dict(dist)}")
    print(f"   accuracy={acc:.3f}  macro-F1(2cls)={macro:.3f}")
    print(f"   Clear     F1={fc[0]:.3f} (P={fc[1]:.2f} R={fc[2]:.2f})")
    print(f"   Not Clear F1={fn[0]:.3f} (P={fn[1]:.2f} R={fn[2]:.2f})")


print("gold(100):", dict(Counter(gold[i] for i in sub_ids)))
report(load(HERE / "raw_cur"), "current prompt")
report(load(HERE / "raw_scoped"), "scoped prompt")
