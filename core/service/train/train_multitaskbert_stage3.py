"""
train_multitaskbert_stage3.py — ST3-focused multitask BERT (joint ST1+ST2+ST3).

A single shared encoder (hfl/chinese-roberta-wwm-ext-large by default) with three
classification heads:

    head_st1: promise_status   (No / Yes)          — always applicable
    head_st2: evidence_status  (No / Yes)           — masked when promise_status != Yes
    head_st3: evidence_quality (Clear/Not Clear/Misleading) — masked when evidence_status != Yes

Every row carries all three labels; non-applicable task labels are set to an
ignore_index so CrossEntropyLoss skips them (no gradient), exactly mirroring the
task-dependency gate (ST2⊂promise=Yes, ST3⊂evidence=Yes). The total loss is the
task-weighted sum of the three per-task losses:

    L = λ_st1·L_st1 + λ_st2·L_st2 + λ_st3·L_st3

The motivation is ST3: the shared encoder gets extra supervision from the (much
larger) ST1/ST2 signals so the rare ST3 minority classes (Not Clear / Misleading)
get a better representation than a single-stage ST3 fine-tune.

Each task loss independently supports ce / weighted_ce / manual_ce, mirroring
core/train/train_bert.py. Best checkpoint is selected by **ST3 val Macro-F1**.

Outputs (under --model-dir):
  - best_multitask_st3.pt : full multitask state_dict (bert + 3 heads)
  - best_st3.pt           : BertClassifier-compatible (bert.* + classifier.* from
                            head_st3) → loads directly in
                            core/human/predict/stage3/pred_by_bert.py and the
                            0613 stage3 eval, so it is comparable with exp23 / a1 / a5.

Usage (exp23-style recipe, ST3-weighted):
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
      core/train/train_multitaskbert_stage3.py \
      --model large \
      --train-path data/raw_data/vpesg_4k_train_1000.json \
      --val-path data/benchmarks/test.json \
      --model-dir /models/multitask_stage3/mt_st123 \
      --st1-loss weighted_ce --st2-loss weighted_ce --st3-loss weighted_ce \
      --task-weights 0.2,0.3,0.5 \
      --output exp/integrated_stage_predictions/0613/multitask_train_stage3/results/mt_st123_train.json
"""

import argparse
import json
import os
import pathlib
import random
from collections import Counter

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, classification_report
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ── Config (reuse the project bert.yml for model resolution + defaults) ─────────
_CFG_PATH = pathlib.Path(__file__).resolve().parents[3] / "configs" / "train" / "bert.yml"
with open(_CFG_PATH) as f:
    _CFG = yaml.safe_load(f)
CONFIGS = _CFG["models"]
SEED_DEFAULT = _CFG.get("seed", 42)
MAX_LEN_DEFAULT = _CFG.get("max_len", 512)
EPOCHS_DEFAULT = _CFG.get("epochs", 5)

# ST3 head is 3-class (no reserved N/A) so best_st3.pt matches pred_by_bert stage3.
ST1_LABEL_MAP = {"No": 0, "Yes": 1}
ST2_LABEL_MAP = {"No": 0, "Yes": 1}
ST3_LABEL_MAP = {"Clear": 0, "Not Clear": 1, "Misleading": 2}
ST3_INV = {v: k for k, v in ST3_LABEL_MAP.items()}
ST1_INV = {v: k for k, v in ST1_LABEL_MAP.items()}
ST2_INV = {v: k for k, v in ST2_LABEL_MAP.items()}
INV = {"st1": ST1_INV, "st2": ST2_INV, "st3": ST3_INV}
NUM_LABELS = {"st1": 2, "st2": 2, "st3": 3}
TASKS = ("st1", "st2", "st3")

# Per-task ignore_index for non-applicable rows (distinct, all < 0).
IGNORE = {"st1": -100, "st2": -101, "st3": -102}

LOSS_MODES = ("ce", "weighted_ce", "manual_ce")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Data ────────────────────────────────────────────────────────────────────────

def load_data(path):
    with open(path) as f:
        return json.load(f)


def build_multitask_samples(data):
    """Each row -> (text, {st1,st2,st3 label or task IGNORE}). Applicability:
    ST2 applicable iff promise_status==Yes; ST3 iff evidence_status==Yes."""
    samples = []
    for d in data:
        text = d.get("data", "")
        ps = d.get("promise_status")
        es = d.get("evidence_status")
        eq = d.get("evidence_quality")
        if ps not in ST1_LABEL_MAP:
            continue  # ST1 label is the anchor; skip rows without it
        st1 = ST1_LABEL_MAP[ps]
        st2 = ST2_LABEL_MAP[es] if (ps == "Yes" and es in ST2_LABEL_MAP) else IGNORE["st2"]
        st3 = ST3_LABEL_MAP[eq] if (es == "Yes" and eq in ST3_LABEL_MAP) else IGNORE["st3"]
        samples.append((text, {"st1": st1, "st2": st2, "st3": st3}))
    return samples


class MTDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.samples = samples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, labels = self.samples[idx]
        enc = self.tok(text, max_length=self.max_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        for t in TASKS:
            item[t] = torch.tensor(labels[t], dtype=torch.long)
        return item


# ── Model ───────────────────────────────────────────────────────────────────────

class MultiTaskBertST3(nn.Module):
    def __init__(self, pretrain_model, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrain_model)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # head names chosen so head_st3 -> classifier remap is trivial for export
        self.head_st1 = nn.Linear(hidden, NUM_LABELS["st1"])
        self.head_st2 = nn.Linear(hidden, NUM_LABELS["st2"])
        self.head_st3 = nn.Linear(hidden, NUM_LABELS["st3"])
        self._heads = {"st1": self.head_st1, "st2": self.head_st2, "st3": self.head_st3}

    def forward(self, input_ids, attention_mask, tasks=TASKS):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]          # [CLS], matches BertClassifier
        pooled = self.dropout(pooled)
        return {t: self._heads[t](pooled) for t in tasks}


# ── Loss ────────────────────────────────────────────────────────────────────────

def compute_class_weights(labels, num_labels, ignore_index):
    counts = [0] * num_labels
    for l in labels:
        if l != ignore_index and 0 <= l < num_labels:
            counts[l] += 1
    total = sum(counts)
    return [total / (num_labels * c) if c > 0 else 1.0 for c in counts]


def manual_class_split_ce(logits, labels, weights, ignore_index):
    """Σ_c w_c · mean_CE(class c). Masked (ignore_index) rows match no class."""
    total = None
    for c, w in enumerate(weights):
        mask = labels == c
        if mask.sum() == 0:
            continue
        ce = F.cross_entropy(logits[mask], labels[mask])
        term = w * ce
        total = term if total is None else total + term
    if total is None:
        return logits.sum() * 0.0
    return total


class TaskLoss:
    def __init__(self, mode, weights, ignore_index, device):
        self.mode = mode
        self.ignore_index = ignore_index
        self.weights = weights
        if mode == "weighted_ce":
            w = torch.tensor(weights, dtype=torch.float, device=device)
            self.ce = nn.CrossEntropyLoss(weight=w, ignore_index=ignore_index)
        elif mode == "ce":
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:  # manual_ce
            self.ce = None

    def __call__(self, logits, labels):
        if (labels != self.ignore_index).sum() == 0:
            return logits.sum() * 0.0  # no applicable rows in this batch
        if self.mode == "manual_ce":
            return manual_class_split_ce(logits, labels, self.weights, self.ignore_index)
        return self.ce(logits, labels)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = {t: [] for t in TASKS}
    golds = {t: [] for t in TASKS}
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        logits = model(ids, am)
        for t in TASKS:
            lab = batch[t]
            p = logits[t].argmax(-1).cpu().numpy()
            for pi, li in zip(p, lab.numpy()):
                if li != IGNORE[t]:
                    preds[t].append(int(pi))
                    golds[t].append(int(li))
    out = {}
    for t in TASKS:
        if golds[t]:
            out[t] = f1_score(golds[t], preds[t],
                              labels=list(range(NUM_LABELS[t])),
                              average="macro", zero_division=0)
        else:
            out[t] = float("nan")
    out["_preds"], out["_golds"] = preds, golds
    return out


# ── Checkpoint export ───────────────────────────────────────────────────────────

def export_best_st3(model, path):
    """Save a BertClassifier-compatible checkpoint: bert.* + classifier.* from
    head_st3, so core/human/predict/stage3/pred_by_bert.py loads it directly."""
    sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if k.startswith("bert."):
            out[k] = v
    out["classifier.weight"] = sd["head_st3.weight"]
    out["classifier.bias"] = sd["head_st3.bias"]
    torch.save(out, path)


# ── Train ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large", choices=list(CONFIGS.keys()))
    ap.add_argument("--train-path", required=True)
    ap.add_argument("--val-path", default=None,
                    help="External val JSON for best-ST3 selection. If unset, use --val-ratio.")
    ap.add_argument("--val-ratio", type=float, default=0.0,
                    help="Held-out fraction for selection if --val-path unset. 0 = no split (best=last epoch).")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output", default=None, help="Result JSON path.")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--max-len", type=int, default=MAX_LEN_DEFAULT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    ap.add_argument("--grad-accum", type=int, default=None, help="Override config grad accum.")
    # per-task loss mode
    ap.add_argument("--st1-loss", default="weighted_ce", choices=LOSS_MODES)
    ap.add_argument("--st2-loss", default="weighted_ce", choices=LOSS_MODES)
    ap.add_argument("--st3-loss", default="weighted_ce", choices=LOSS_MODES)
    # manual per-task class weights, e.g. --st3-class-weights 1,8,30
    ap.add_argument("--st1-class-weights", default=None)
    ap.add_argument("--st2-class-weights", default=None)
    ap.add_argument("--st3-class-weights", default=None)
    # task loss weights λ_st1,λ_st2,λ_st3
    ap.add_argument("--task-weights", default="0.2,0.3,0.5",
                    help="Comma λ for st1,st2,st3 in the summed loss.")
    ap.add_argument("--no-epoch-saves", action="store_true")
    ap.add_argument("--st3-drop-misleading", action="store_true",
                    help="Train ST3 head as 2 classes (Clear/Not Clear) only; "
                         "Misleading rows are masked out (excluded from the loss). "
                         "Pass 2 values to --st3-class-weights (e.g. 1,8).")
    args = ap.parse_args()

    if args.st3_drop_misleading:
        # Mutate the ST3 label space in place so build_multitask_samples() maps
        # Misleading rows to IGNORE (eq not in ST3_LABEL_MAP) and the head/loss
        # use 2 classes. ST3_INV is the same object referenced by INV["st3"].
        ST3_LABEL_MAP.pop("Misleading", None)
        ST3_INV.clear()
        ST3_INV.update({v: k for k, v in ST3_LABEL_MAP.items()})
        NUM_LABELS["st3"] = len(ST3_LABEL_MAP)
        print(f"[st3] drop-misleading: ST3 head -> {NUM_LABELS['st3']} classes {ST3_LABEL_MAP}")

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = CONFIGS[args.model]
    pretrain = cfg["pretrain_model"]
    batch_size = args.batch_size or cfg.get("batch_size", 8)
    grad_accum = args.grad_accum or cfg.get("grad_accum", 2)
    lr = float(cfg.get("lr", 1e-5))

    task_w = {t: float(x) for t, x in zip(TASKS, args.task_weights.split(","))}
    loss_mode = {"st1": args.st1_loss, "st2": args.st2_loss, "st3": args.st3_loss}
    manual_cw = {
        "st1": args.st1_class_weights, "st2": args.st2_class_weights, "st3": args.st3_class_weights,
    }

    os.makedirs(args.model_dir, exist_ok=True)
    print(f"Device: {device}  Model: {pretrain}")
    print(f"Tasks: {TASKS}  task_weights={task_w}")
    print(f"Loss modes: {loss_mode}")

    # ── Data
    data = load_data(args.train_path)
    samples = build_multitask_samples(data)
    print(f"Train rows: {len(data)}  multitask samples: {len(samples)}")

    val_samples = None
    if args.val_path:
        val_samples = build_multitask_samples(load_data(args.val_path))
        train_samples = samples
        print(f"External val: {args.val_path}  val samples: {len(val_samples)}")
    elif args.val_ratio and args.val_ratio > 0:
        idx = list(range(len(samples)))
        random.Random(args.seed).shuffle(idx)
        n_val = int(len(idx) * args.val_ratio)
        val_idx = set(idx[:n_val])
        train_samples = [s for i, s in enumerate(samples) if i not in val_idx]
        val_samples = [samples[i] for i in val_idx]
        print(f"Internal split: train {len(train_samples)} / val {len(val_samples)}")
    else:
        train_samples = samples
        print("No validation split — best checkpoint = last epoch.")

    for t in TASKS:
        dist = Counter(l[t] for _, l in train_samples if l[t] != IGNORE[t])
        named = {INV[t].get(k, k): v for k, v in sorted(dist.items())}
        print(f"  {t} applicable train dist: {named}")

    tokenizer = AutoTokenizer.from_pretrained(pretrain)
    train_loader = DataLoader(MTDataset(train_samples, tokenizer, args.max_len),
                              batch_size=batch_size, shuffle=True)
    val_loader = (DataLoader(MTDataset(val_samples, tokenizer, args.max_len),
                             batch_size=batch_size, shuffle=False)
                  if val_samples else None)

    # ── Loss objects (class weights from train labels)
    task_losses = {}
    for t in TASKS:
        labels_t = [l[t] for _, l in train_samples]
        if manual_cw[t]:
            weights = [float(x) for x in manual_cw[t].split(",")]
        else:
            weights = compute_class_weights(labels_t, NUM_LABELS[t], IGNORE[t])
        task_losses[t] = TaskLoss(loss_mode[t], weights, IGNORE[t], device)
        print(f"  {t} loss={loss_mode[t]} weights={[round(w,4) for w in weights]}")

    # ── Model / optim
    model = MultiTaskBertST3(pretrain, dropout=args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=max(1, total_steps))

    best_st3 = -1.0
    best_epoch = None
    epoch_metrics = []
    best_full = os.path.join(args.model_dir, "best_multitask_st3.pt")
    best_st3_path = os.path.join(args.model_dir, "best_st3.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        running = 0.0
        per_task_run = {t: 0.0 for t in TASKS}
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            logits = model(ids, am)
            loss = 0.0
            for t in TASKS:
                lt = task_losses[t](logits[t], batch[t].to(device))
                per_task_run[t] += float(lt.detach())
                loss = loss + task_w[t] * lt
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            running += float(loss.detach())
        avg = running / max(1, len(train_loader))
        pt = {t: round(per_task_run[t] / max(1, len(train_loader)), 4) for t in TASKS}

        rec = {"epoch": epoch, "train_loss": round(avg, 4), "train_task_loss": pt}
        if val_loader is not None:
            ev = evaluate(model, val_loader, device)
            rec["val_f1"] = {t: round(ev[t], 4) for t in TASKS}
            sel = ev["st3"]
            msg = " | ".join(f"{t} F1={ev[t]:.4f}" for t in TASKS)
            print(f"Epoch {epoch}/{args.epochs} | loss={avg:.4f} {pt} | {msg}")
            if sel >= best_st3:
                best_st3 = sel
                best_epoch = epoch
                torch.save(model.state_dict(), best_full)
                export_best_st3(model, best_st3_path)
                print(f"  → new best ST3 F1={sel:.4f}  saved best_st3.pt")
        else:
            print(f"Epoch {epoch}/{args.epochs} | loss={avg:.4f} {pt}")
            # no val: keep last epoch as best
            best_epoch = epoch
            torch.save(model.state_dict(), best_full)
            export_best_st3(model, best_st3_path)
        if not args.no_epoch_saves:
            torch.save(model.state_dict(), os.path.join(args.model_dir, f"epoch_mt_{epoch:03d}.pt"))
        epoch_metrics.append(rec)

    # ── Final train-set report (ST3)
    final = evaluate(model, train_loader, device)
    print("\n=== Final train Macro-F1 ===")
    for t in TASKS:
        print(f"  {t}: {final[t]:.4f}")
    if final["_golds"]["st3"]:
        print(classification_report(final["_golds"]["st3"], final["_preds"]["st3"],
              labels=[0, 1, 2], target_names=["Clear", "Not Clear", "Misleading"],
              zero_division=0, digits=3))

    result = {
        "model": pretrain,
        "tasks": list(TASKS),
        "task_weights": task_w,
        "loss_modes": loss_mode,
        "selection": ("val_path:" + args.val_path) if args.val_path else
                     (f"val_ratio:{args.val_ratio}" if args.val_ratio else "last_epoch"),
        "best_epoch": best_epoch,
        "best_val_st3_f1": best_st3 if val_loader is not None else None,
        "epoch_metrics": epoch_metrics,
        "checkpoints": {"multitask": best_full, "st3_compat": best_st3_path},
    }
    out_path = args.output or os.path.join(args.model_dir, "train_mt_st3.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nBest epoch: {best_epoch}  best_st3.pt → {best_st3_path}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
