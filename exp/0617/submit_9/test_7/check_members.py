#!/usr/bin/env python3
"""Independent per-member check: each ensemble member on ITS OWN held-out val.

Each member was trained on its own seed split, so the ONLY leakage-free signal
for that member is its OWN val.json (the ~20% this seed held out). This script,
for one stage, finds every member checkpoint and evaluates member seed<S> on
  data/ensemble/seed<S>/<name>.val.json
reporting per-member coverage + raw-head Macro-F1 on the gold labels in that file.

This is a fully INDEPENDENT per-model test -- no cross-member alignment, no soft
vote. (The global vpesg4k_val_1000 is NOT used here: it is folded into every
member's training set via the `_add_val` pool, so scoring on it leaks.)

Run via test_7/run_test.sh, or directly:
  .venv/bin/python .../test_7/check_members.py --stage 1 \
    --ckpt-glob '/models/ensemble_st1_a3_b1_focal_g3_w4_seed*/best_st1.pt'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from core.human.predict.stage1 import pred_by_bert as s1
from core.human.predict.stage2 import pred_by_bert as s2
from core.human.predict.stage3 import pred_by_bert_multitask as s3
from core.human.predict.stage3.pred_by_bert import LABEL_BY_ID as ST3_LABEL_BY_ID

MODEL = "hfl/chinese-roberta-wwm-ext-large"
ENS = "exp/integrated_stage_predictions/0615/ensemble"


def run_s1(rows, ckpt, dev, bs, ml):
    preds = s1.predict_rows(rows=rows, model_name=MODEL, checkpoint_path=ckpt, device=dev,
                            batch_size=bs, max_len=ml, text_col="data", local_files_only=True)
    return {r["id"]: r["promise_status"] for r in preds}


def run_s2(rows, ckpt, dev, bs, ml):
    preds = s2.predict_rows(rows, MODEL, ckpt, dev, bs, ml, text_mode="data", local_files_only=True)
    return {rid: r["evidence_status"] for rid, r in preds.items()}


def run_s3(rows, ckpt, dev, bs, ml):
    preds = s3.predict_rows(rows, MODEL, ckpt, dev, bs, ml, local_files_only=True, nc_tau=None)
    return {rid: r["evidence_quality"] for rid, r in preds.items()}


# per stage: runner, gold field, the classes the head can emit, default own-val template
STAGE_CFG = {
    1: dict(run=run_s1, target="promise_status", head={"No", "Yes"},
            val_tmpl=f"{ENS}/stage1/data/ensemble/seed{{seed}}/a3_b1_add_val.val.json"),
    2: dict(run=run_s2, target="evidence_status", head={"No", "Yes"},
            val_tmpl=f"{ENS}/stage2/data/ensemble/seed{{seed}}/mix_a2_b3_add_val.val.json"),
    3: dict(run=run_s3, target="evidence_quality", head={"Clear", "Not Clear", "Misleading"},
            val_tmpl=f"{ENS}/stage3/data/ensemble/seed{{seed}}/vpesg_4k_train_1000_add_val.val.json"),
}


def macro_f1(pairs):
    """pairs = [(gold, pred), ...]; macro over labels present in gold."""
    labels = sorted({g for g, _ in pairs})
    per = {}
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[lab] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return (sum(per.values()) / len(per) if per else 0.0), per


def parse_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--ckpt-glob", required=True, help="Glob for this stage's member checkpoints.")
    ap.add_argument("--val-template", default=None,
                    help="Own-val path with {seed} placeholder (default: this stage's ensemble val split).")
    ap.add_argument("--expect", type=int, default=5, help="Expected member count (default 5).")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="Smoke-test row limit per member.")
    args = ap.parse_args()

    cfg = STAGE_CFG[args.stage]
    tmpl = args.val_template or cfg["val_tmpl"]
    dev = parse_device(args.device)
    ckpts = [Path(p) for p in sorted(glob.glob(args.ckpt_glob))]

    print(f"\n========== STAGE {args.stage}  (each member on its OWN val) ==========")
    print(f"glob    : {args.ckpt_glob}")
    print(f"val tmpl: {tmpl}")
    print(f"members : {len(ckpts)} (expected {args.expect})")

    fails = []
    if len(ckpts) != args.expect:
        fails.append(f"member count {len(ckpts)} != expected {args.expect}")

    print(f"\n{'member':<48} {'seed':>6} {'val':>5} {'miss':>5} {'app':>4}  {'F1':>7}  dist")
    f1s_all = []
    for ckpt in ckpts:
        m = re.search(r"seed(\d+)", str(ckpt))
        if not m:
            fails.append(f"cannot parse seed from {ckpt}"); continue
        seed = m.group(1)
        val_path = ROOT / tmpl.format(seed=seed)
        if not val_path.is_file():
            fails.append(f"seed{seed}: missing own val {val_path}")
            print(f"{ckpt.parent.name:<48} {seed:>6}   MISSING VAL {val_path}")
            continue
        rows = json.loads(val_path.read_text(encoding="utf-8"))
        if args.limit is not None:
            rows = rows[: args.limit]
        val_ids = [str(r["id"]) for r in rows]
        gold = {str(r["id"]): str(r.get(cfg["target"], "")) for r in rows}

        pred = cfg["run"](rows, ckpt, dev, args.batch_size, args.max_len)
        missing = set(val_ids) - set(pred)
        if missing:
            fails.append(f"seed{seed}: {len(missing)} val rows not predicted")

        # honest Macro-F1 on rows whose gold is a class the head can emit
        pairs = [(gold[i], pred[i]) for i in val_ids if i in pred and gold[i] in cfg["head"]]
        f1, _ = macro_f1(pairs) if pairs else (float("nan"), {})
        if f1 == f1:
            f1s_all.append(f1)
        dist = dict(sorted(Counter(pred[i] for i in val_ids if i in pred).items()))
        f1str = f"{f1:.4f}" if f1 == f1 else "    n/a"
        print(f"{ckpt.parent.name:<48} {seed:>6} {len(val_ids):>5} {len(missing):>5} {len(pairs):>4}  {f1str:>7}  {dist}")

    if f1s_all:
        mean = sum(f1s_all) / len(f1s_all)
        print(f"\nper-member own-val Macro-F1: mean={mean:.4f}  min={min(f1s_all):.4f}  max={max(f1s_all):.4f}  (n={len(f1s_all)})")

    ok = not fails
    print(f"\nRESULT stage {args.stage}: {'PASS' if ok else 'FAIL'}")
    for fmsg in fails:
        print(f"  - {fmsg}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
