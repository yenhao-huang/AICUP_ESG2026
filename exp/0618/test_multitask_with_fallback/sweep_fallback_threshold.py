#!/usr/bin/env python3
"""Sweep the multitask->codex fallback confidence threshold on the val set.

Policy: multitask (w0_2_0_3_0_5) is the primary ST3 predictor. For each row whose
multitask winning-confidence < T, replace its label with the codex add-context
prediction; otherwise keep multitask. Score every T on the SAME 3-class subset
口徑 (gold in {Clear, Not Clear, Misleading}, n=271) used for the 0.5209 baseline,
so the numbers are directly comparable.

T=0.0  -> all multitask (baseline 0.5209 / 86.0%)
T=1.01 -> all codex     (0.4867 / 83.0%)

Paths are self-contained (derived from __file__), so no $DIR needed.
GT is used ONLY for offline scoring, never as model input.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
GOLD = HERE / "data" / "vpesg_4k_train_1000_add_val.val.json"
MT = HERE / "preds" / "st3_val_pred.csv"                       # multitask (has softmax conf)
CODEX = HERE / "preds" / "codex" / "add_context_val_codex.csv"  # codex add-context
LABELS = ["Clear", "Not Clear", "Misleading"]


def load_gold():
    return {str(r.get("id")): (r.get("evidence_quality") or "") for r in json.load(open(GOLD))}


def load_mt():
    """id -> (pred_label, winning_confidence)."""
    out = {}
    for r in csv.DictReader(open(MT)):
        reason = r.get("evidence_quality_reason", "") or ""
        try:
            d = dict(kv.split("=") for kv in reason.split(";"))
            conf = max(float(v) for v in d.values())
        except Exception:
            conf = 1.0
        out[r["id"]] = (r["evidence_quality"], conf)
    return out


def load_codex():
    return {r["id"]: r["evidence_quality"] for r in csv.DictReader(open(CODEX))}


def main():
    gold = load_gold()
    mt = load_mt()
    codex = load_codex()

    sub_ids = [i for i in gold if gold[i] in LABELS]   # 271
    y = [gold[i] for i in sub_ids]

    def eval_at(T):
        merged = {}
        for i in mt:
            pred, conf = mt[i]
            merged[i] = codex.get(i, pred) if conf < T else pred
        yp = [merged[i] for i in sub_ids]
        acc = accuracy_score(y, yp)
        macro = f1_score(y, yp, labels=LABELS, average="macro", zero_division=0)
        per = f1_score(y, yp, labels=LABELS, average=None, zero_division=0)
        n_fb = sum(1 for i in sub_ids if mt[i][1] < T)        # fallback 數 (子集內)
        return acc, macro, per, n_fb

    grid = [0.0, 0.55, 0.57, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
    n_sub = len(sub_ids)
    print(f"fallback sweep | val 純3類子集 n={n_sub} | multitask 為主, conf<T 換 codex add-context")
    print(f"{'T':>5} {'fallback':>10} {'Accuracy':>9} {'Macro-F1':>9} {'Clear':>7} {'NotClr':>7}  note")
    best = (-1, None)
    rows = []
    for T in grid:
        acc, macro, per, n_fb = eval_at(T)
        note = ""
        if T == 0.0: note = "= 全 multitask (baseline)"
        elif T == 1.01: note = "= 全 codex"
        rows.append((T, n_fb, acc, macro, per))
        if 0.0 < T < 1.01 and macro > best[0]:
            best = (macro, T)
        print(f"{T:>5.2f} {n_fb:>4}/{n_sub:<5} {acc*100:>7.1f}% {macro:>9.4f} "
              f"{per[0]:>7.3f} {per[1]:>7.3f}  {note}")
    base_macro = rows[0][3]
    print(f"\nbaseline(全multitask) Macro-F1={base_macro:.4f}")
    if best[1] is not None:
        bm, bt = best
        delta = bm - base_macro
        verdict = "↑ fallback 有幫助" if delta > 0 else "↓ 沒有 threshold 勝過純 multitask"
        print(f"best fallback T={bt:.2f} Macro-F1={bm:.4f} (Δ={delta:+.4f}) {verdict}")


if __name__ == "__main__":
    main()
