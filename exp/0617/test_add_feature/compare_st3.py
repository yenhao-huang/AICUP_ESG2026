"""Compare three ST3 models on the val_1000 ST3-applicable subset.

Models (all loaded as a plain 3-class BertClassifier; the multitask checkpoint's
aux_head.* keys are ignored via strict=False — inference uses the main head only):

  ref        /models/esg_contest/exp23_train_json_st2_st3_st4_large/best_st3.pt
  vanilla    Method-A control (retrained, no aux heads)
  multitask  Method A (aux verifiability heads)

Eval population = build_subtask_samples(val, "st3"): rows with GT
evidence_status == "Yes" and a non-blank evidence_quality, 3-class
{Clear, Not Clear, Misleading}. This isolates the ST3 model's clarity
discrimination (the N/A class is cascade-driven and not produced by these models).

Usage:
  .venv/bin/python compare_st3.py \
      --val data/raw_data/vpesg4k_val_1000.json \
      --ref /models/esg_contest/exp23_train_json_st2_st3_st4_large/best_st3.pt \
      --vanilla /models/test_add_feature_0617/vanilla/best_st3.pt \
      --multitask /models/test_add_feature_0617/multitask/best_st3.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _REPO)
from core.train.train_bert import (  # noqa: E402
    INV_LABEL_MAPS,
    LABEL_MAPS,
    BertClassifier,
    build_subtask_samples,
    load_data,
)

ST3_NUM_LABELS = len(LABEL_MAPS["st3"])  # 3
CLASSES = [INV_LABEL_MAPS["st3"][i] for i in range(ST3_NUM_LABELS)]  # Clear, Not Clear, Misleading


class _DS(torch.utils.data.Dataset):
    def __init__(self, samples, tok, max_len):
        self.s, self.tok, self.max_len = samples, tok, max_len

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        text, label = self.s[i]
        enc = self.tok(text, max_length=self.max_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long)}


@torch.no_grad()
def predict(ckpt, loader, pretrain, device):
    model = BertClassifier(pretrain, ST3_NUM_LABELS).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # aux_head.* expected to be unexpected for the multitask checkpoint.
    unexpected = [k for k in unexpected if not k.startswith("aux_head")]
    if missing or unexpected:
        print(f"  [warn] {ckpt}: missing={missing} unexpected={unexpected}")
    model.eval()
    preds, labels = [], []
    for b in loader:
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        preds.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(b["label"].tolist())
    return preds, labels


def per_class(labels, preds):
    rep = classification_report(labels, preds, labels=list(range(ST3_NUM_LABELS)),
                                target_names=CLASSES, zero_division=0, output_dict=True)
    macro = f1_score(labels, preds, average="macro", zero_division=0)
    # 2-class (Clear vs Not Clear) macro, excluding the single Misleading row.
    pair = [(l, p) for l, p in zip(labels, preds) if l in (0, 1)]
    lp = [l for l, _ in pair]; pp = [min(p, 1) for _, p in pair]
    macro_2c = f1_score(lp, pp, average="macro", zero_division=0)
    return {
        "macro_f1_3class": macro,
        "macro_f1_2class_clear_nc": macro_2c,
        "accuracy": accuracy_score(labels, preds),
        "per_class": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                          "f1": rep[c]["f1-score"], "support": rep[c]["support"]}
                      for c in CLASSES},
        "pred_dist": {CLASSES[i]: preds.count(i) for i in range(ST3_NUM_LABELS)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="data/raw_data/vpesg4k_val_1000.json")
    ap.add_argument("--ref", default="/models/esg_contest/exp23_train_json_st2_st3_st4_large/best_st3.pt")
    ap.add_argument("--vanilla", default="/models/test_add_feature_0617/vanilla/best_st3.pt")
    ap.add_argument("--multitask", default="/models/test_add_feature_0617/multitask/best_st3.pt")
    ap.add_argument("--pretrain", default="hfl/chinese-roberta-wwm-ext-large")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.pretrain)
    samples = build_subtask_samples(load_data(args.val), "st3")
    from collections import Counter
    print(f"val ST3-applicable: {len(samples)} dist="
          f"{ {INV_LABEL_MAPS['st3'][k]: v for k, v in Counter(l for _, l in samples).items()} }")
    loader = DataLoader(_DS(samples, tok, args.max_len), batch_size=16, shuffle=False, num_workers=4)

    models = {"ref(exp23)": args.ref, "vanilla": args.vanilla, "multitask": args.multitask}
    results = {}
    labels_ref = None
    for name, ckpt in models.items():
        if not os.path.exists(ckpt):
            print(f"[skip] {name}: missing {ckpt}")
            continue
        print(f"\n>>> {name}: {ckpt}")
        preds, labels = predict(ckpt, loader, args.pretrain, device)
        labels_ref = labels
        results[name] = per_class(labels, preds)

    # ── Comparison table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"{'model':12s} {'macroF1':>8s} {'2cF1':>7s} {'acc':>6s} "
          f"{'Clear-F1':>9s} {'NC-P':>6s} {'NC-R':>6s} {'NC-F1':>6s} {'Mis-F1':>6s} {'NCpred':>7s}")
    print("-" * 92)
    for name, r in results.items():
        nc = r["per_class"]["Not Clear"]; cl = r["per_class"]["Clear"]; mis = r["per_class"]["Misleading"]
        print(f"{name:12s} {r['macro_f1_3class']:8.4f} {r['macro_f1_2class_clear_nc']:7.4f} "
              f"{r['accuracy']:6.3f} {cl['f1']:9.4f} {nc['precision']:6.3f} {nc['recall']:6.3f} "
              f"{nc['f1']:6.3f} {mis['f1']:6.3f} {r['pred_dist']['Not Clear']:7d}")
    print("=" * 92)

    # Deltas vs ref and vanilla.
    def delta(a, b, key="macro_f1_3class"):
        return results[a][key] - results[b][key] if a in results and b in results else None
    if "multitask" in results and "vanilla" in results:
        print(f"\nMethod-A effect (multitask - vanilla): "
              f"macroF1 {delta('multitask','vanilla'):+.4f} | "
              f"NC-F1 {results['multitask']['per_class']['Not Clear']['f1'] - results['vanilla']['per_class']['Not Clear']['f1']:+.4f} | "
              f"NC-P {results['multitask']['per_class']['Not Clear']['precision'] - results['vanilla']['per_class']['Not Clear']['precision']:+.4f}")
    if "multitask" in results and "ref(exp23)" in results:
        print(f"multitask - ref(exp23): macroF1 {delta('multitask','ref(exp23)'):+.4f}")
    if "vanilla" in results and "ref(exp23)" in results:
        print(f"vanilla   - ref(exp23): macroF1 {delta('vanilla','ref(exp23)'):+.4f}")

    out = args.output or os.path.join(os.path.dirname(__file__), "results", "compare_st3.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"val": args.val, "n_eval": len(samples), "models": models, "results": results},
              open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\ncompare -> {out}")


if __name__ == "__main__":
    main()
