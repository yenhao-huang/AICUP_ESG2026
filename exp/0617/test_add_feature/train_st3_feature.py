"""Method A — verifiability-aware Stage 3 (evidence_quality) training.

Trains either of two models on the SAME data/recipe so the only difference is the
auxiliary supervision:

  --mode vanilla     plain BertClassifier (3-class Clear/Not Clear/Misleading).
                     Control: isolates the effect of retraining vs. exp23.
  --mode multitask   shared encoder + main 3-class head + auxiliary binary heads
                     predicting the data-only verifiability signals from
                     verifiability_features.py (Method A). Inference uses ONLY the
                     main head, so the saved checkpoint is load-compatible with a
                     plain BertClassifier (aux_head.* is ignored at eval).

Data-only: model input is d["data"]; aux targets are regex-derived from d["data"]
(no annotation field). Label space matches exp23 (gated_mis 3-class, rows with
evidence_status == "Yes" and a non-blank evidence_quality).

Example:
  .venv/bin/python train_st3_feature.py --mode multitask \
      --train data/raw_data/vpesg_4k_train_1000.json \
      --val   data/raw_data/vpesg4k_val_1000.json \
      --model-dir /models/test_add_feature_0617/multitask --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# Reuse the canonical data gating / model / helpers (no infra change).
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _REPO)
from core.train.train_bert import (  # noqa: E402
    INV_LABEL_MAPS,
    LABEL_MAPS,
    BertClassifier,
    build_subtask_samples,
    compute_class_weights,
    load_data,
    set_seed,
)

from verifiability_features import AUX_NAMES, NUM_AUX, extract_vector  # noqa: E402

ST3_NUM_LABELS = len(LABEL_MAPS["st3"])  # 3: Clear / Not Clear / Misleading


# ── Model ────────────────────────────────────────────────────────────────────────

class MultiTaskBertClassifier(nn.Module):
    """Shared encoder + main ST3 head (`classifier`) + auxiliary head (`aux_head`).

    `bert` and `classifier` are named identically to BertClassifier so the saved
    state_dict loads into a plain BertClassifier with strict=False at eval time.
    """

    def __init__(self, pretrain_model, num_labels, num_aux, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrain_model)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.aux_head = nn.Linear(hidden, num_aux)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        return self.classifier(pooled), self.aux_head(pooled)


# ── Data ─────────────────────────────────────────────────────────────────────────

class ST3FeatureDataset(Dataset):
    """(text, main_label) from core gating + regex-derived aux target vector."""

    def __init__(self, samples, tokenizer, max_len):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "aux": torch.tensor(extract_vector(text), dtype=torch.float),
        }


# ── Eval ─────────────────────────────────────────────────────────────────────────

def _forward_logits(model, input_ids, attention_mask, is_mt):
    out = model(input_ids, attention_mask)
    return out[0] if is_mt else out


@torch.no_grad()
def evaluate(model, loader, device, is_mt):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        logits = _forward_logits(
            model, batch["input_ids"].to(device), batch["attention_mask"].to(device), is_mt
        )
        preds.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(batch["label"].tolist())
    macro = f1_score(labels, preds, average="macro", zero_division=0)
    # Not-Clear (index 1) F1 specifically — the P3 target.
    nc_f1 = f1_score(labels, preds, average=None, labels=[1], zero_division=0)[0]
    return macro, nc_f1, preds, labels


# ── Train ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vanilla", "multitask"], required=True)
    ap.add_argument("--train", default="data/raw_data/vpesg_4k_train_1000.json")
    ap.add_argument("--val", default="data/raw_data/vpesg4k_val_1000.json")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pretrain", default="hfl/chinese-roberta-wwm-ext-large")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aux-lambda", type=float, default=0.5,
                    help="Weight of the auxiliary BCE loss (multitask only).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=None, help="Metrics JSON path.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    is_mt = args.mode == "multitask"
    set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"=== Method A train | mode={args.mode} | device={device} | seed={args.seed} ===")
    print(f"pretrain={args.pretrain} max_len={args.max_len} bs={args.batch_size} "
          f"ga={args.grad_accum} lr={args.lr} epochs={args.epochs}")
    if is_mt:
        print(f"aux heads ({NUM_AUX}): {AUX_NAMES} | aux_lambda={args.aux_lambda}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
    train_samples = build_subtask_samples(load_data(args.train), "st3")  # gated_mis 3-class
    val_samples = build_subtask_samples(load_data(args.val), "st3")
    from collections import Counter
    print(f"train ST3-applicable: {len(train_samples)} dist={dict(Counter(l for _, l in train_samples))}")
    print(f"val   ST3-applicable: {len(val_samples)} dist={dict(Counter(l for _, l in val_samples))}")

    train_dl = DataLoader(ST3FeatureDataset(train_samples, tokenizer, args.max_len),
                          batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(ST3FeatureDataset(val_samples, tokenizer, args.max_len),
                        batch_size=args.batch_size, shuffle=False, num_workers=4)

    if is_mt:
        model = MultiTaskBertClassifier(args.pretrain, ST3_NUM_LABELS, NUM_AUX, args.dropout).to(device)
    else:
        model = BertClassifier(args.pretrain, ST3_NUM_LABELS, args.dropout).to(device)

    class_w = compute_class_weights(train_samples, ST3_NUM_LABELS).to(device)
    main_crit = nn.CrossEntropyLoss(weight=class_w)
    aux_crit = nn.BCEWithLogitsLoss()
    print(f"main loss: weighted CE weights={[round(w.item(),3) for w in class_w]}")

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = (len(train_dl) // args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(steps * args.warmup_ratio), steps)

    best_path = os.path.join(args.model_dir, "best_st3.pt")
    best_macro, best_epoch, history = -1.0, None, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_main, run_aux = 0.0, 0.0
        optim.zero_grad()
        for step, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            if is_mt:
                logits, aux_logits = model(ids, mask)
                l_main = main_crit(logits, labels)
                l_aux = aux_crit(aux_logits, batch["aux"].to(device))
                loss = (l_main + args.aux_lambda * l_aux) / args.grad_accum
                run_aux += l_aux.item()
            else:
                logits = model(ids, mask)
                l_main = main_crit(logits, labels)
                loss = l_main / args.grad_accum
            loss.backward()
            run_main += l_main.item()
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_dl):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step(); optim.zero_grad()

        macro, nc_f1, _, _ = evaluate(model, val_dl, device, is_mt)
        n = len(train_dl)
        msg = (f"epoch {epoch}/{args.epochs} | main_loss={run_main/n:.4f}"
               + (f" aux_loss={run_aux/n:.4f}" if is_mt else "")
               + f" | val macroF1={macro:.4f} val NC-F1={nc_f1:.4f}")
        print(msg)
        history.append({"epoch": epoch, "val_macro_f1": macro, "val_nc_f1": nc_f1,
                        "train_main_loss": run_main / n,
                        "train_aux_loss": (run_aux / n) if is_mt else None})
        if macro > best_macro:
            best_macro, best_epoch = macro, epoch
            torch.save(model.state_dict(), best_path)
            print(f"  saved best -> {best_path} (epoch={epoch}, val_macro_f1={macro:.4f})")

    # Final report on best checkpoint.
    state = torch.load(best_path, map_location=device, weights_only=False)
    if is_mt:
        model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    macro, nc_f1, preds, labels = evaluate(model, val_dl, device, is_mt)
    present = sorted(set(labels) | set(preds))
    names = [INV_LABEL_MAPS["st3"][i] for i in present]
    report = classification_report(labels, preds, labels=present, target_names=names,
                                   zero_division=0, digits=4)
    print(f"\n=== best epoch {best_epoch} | val macroF1={macro:.4f} | NC-F1={nc_f1:.4f} ===")
    print(report)

    metrics = {
        "mode": args.mode, "best_epoch": best_epoch,
        "val_macro_f1": macro, "val_nc_f1": nc_f1,
        "aux_lambda": args.aux_lambda if is_mt else None,
        "aux_names": AUX_NAMES if is_mt else None,
        "best_checkpoint": best_path,
        "train": args.train, "val": args.val,
        "history": history,
        "classification_report": classification_report(
            labels, preds, labels=present, target_names=names,
            zero_division=0, output_dict=True),
    }
    out = args.output or os.path.join(
        os.path.dirname(__file__), "results", f"train_{args.mode}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(metrics, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"metrics -> {out}")


if __name__ == "__main__":
    main()
