#!/usr/bin/env python3
"""Offline ST1 Macro-F1 scorer: merged/scored CSV predictions vs GT promise_status.

GT labels are used for OFFLINE SCORING ONLY. Runtime path is data-only.
Usage: score_csv.py <pred_csv> <eval_json> [conf_threshold_for_base_routing_count]
Prints JSON: macro_f1, per-class F1 (Yes/No), n, escalation count if applicable.
"""
import csv, json, sys
from sklearn.metrics import f1_score, precision_recall_fscore_support

pred_csv = sys.argv[1]
eval_json = sys.argv[2]

gt = {}
for r in json.load(open(eval_json)):
    gt[str(r["id"])] = r["promise_status"]

rows = list(csv.DictReader(open(pred_csv, encoding="utf-8-sig")))
y_true, y_pred = [], []
src_counter = {}
for r in rows:
    rid = str(r["id"]).strip()
    if rid not in gt:
        continue
    y_true.append(gt[rid])
    y_pred.append(r["promise_status"].strip())
    s = r.get("source", "")
    src_counter[s] = src_counter.get(s, 0) + 1

labels = ["Yes", "No"]
macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
p, rc, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
out = {
    "n": len(y_true),
    "macro_f1": round(float(macro), 6),
    "f1_Yes": round(float(f[0]), 6),
    "f1_No": round(float(f[1]), 6),
    "support": {"Yes": int(sup[0]), "No": int(sup[1])},
    "sources": src_counter,
}
print(json.dumps(out, ensure_ascii=False))
