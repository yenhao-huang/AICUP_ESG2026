"""
BERT training for VeriPromiseESG 2026.

Usage:
    python train_bert.py --model baseline   # hfl/chinese-roberta-wwm-ext (102M)
    python train_bert.py --model large      # hfl/chinese-roberta-wwm-ext-large (330M)
"""

import argparse
import json
import os
import pathlib
import random
from collections import Counter

import warnings

# Suppress HuggingFace Hub warnings about disabled discussions
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore", message=".* Discussions are disabled.*")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, classification_report
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ── Config ─────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = pathlib.Path(__file__).parents[3] / "configs" / "train" / "bert.yml"

def _load_yaml_config(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

_CFG = _load_yaml_config(_DEFAULT_CONFIG)

TRAIN_PATH = _CFG["train_path"]
CONFIGS    = _CFG["models"]
SEED       = _CFG["seed"]
MAX_LEN    = _CFG["max_len"]
EPOCHS     = _CFG["epochs"]
DEVICE     = torch.device("cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"))

LABEL_MAPS = {
    "st1": {"Yes": 1, "No": 0},
    "st2": {"Yes": 1, "No": 0},
    "st3": {"Clear": 0, "Not Clear": 1, "Misleading": 2},
    # Gate-free 3-class ST3 (loop001). Canonical index order: 0=Clear, 1=Not Clear, 2=N/A.
    # Used only when build_subtask_samples is called with st3_label_space="3class_na".
    "st3_3class_na": {"Clear": 0, "Not Clear": 1, "N/A": 2},
    # Clean 2-class gated ST3 (loop002). Canonical index order: 0=Clear, 1=Not Clear.
    # Used only when build_subtask_samples is called with st3_label_space="2class".
    # N/A is NOT learned (supplied at runtime by the predicted ST1/ST2 cascade gate);
    # Misleading is folded per st3_mis_policy (drop / nc).
    "st3_2class": {"Clear": 0, "Not Clear": 1},
    "st4": {"already": 0, "within_2_years": 1, "between_2_and_5_years": 2, "more_than_5_years": 3},
}
INV_LABEL_MAPS = {k: {v: k2 for k2, v in m.items()} for k, m in LABEL_MAPS.items()}
TASK_WEIGHTS = {"st1": 0.20, "st2": 0.30, "st3": 0.35, "st4": 0.15}

# ── Data ───────────────────────────────────────────────────────────────────────

def load_data(path):
    with open(path) as f:
        return json.load(f)


def build_subtask_samples(data, subtask, st3_label_space="gated_mis", st3_mis_policy="drop"):
    """Build (text, label) samples for a subtask.

    Model input text is ALWAYS d["data"] (data-only compliance).

    st3_label_space (only affects subtask == "st3"):
        "gated_mis"  : legacy gated 3-class {Clear:0, Not Clear:1, Misleading:2}.
                       Keeps ONLY rows with evidence_status == "Yes" and a
                       non-blank evidence_quality. Byte-for-byte unchanged from
                       the original behavior; this is the default.
        "3class_na"  : gate-free 3-class {Clear:0, Not Clear:1, N/A:2}.
                       Keeps ALL rows. Blank evidence_quality (promise_status==No
                       OR evidence_status==No) -> N/A(2). Misleading is folded
                       per st3_mis_policy.
        "2class"     : clean gated 2-class {Clear:0, Not Clear:1} (loop002).
                       Keeps ONLY rows with evidence_status == "Yes" (the existing
                       ST3-applicable subset, same gate as gated_mis). No N/A class
                       (N/A is supplied at runtime by the predicted cascade gate,
                       NOT learned). Misleading is folded per st3_mis_policy.
    st3_mis_policy (affects st3_label_space in {"3class_na", "2class"}):
        "drop" : drop Misleading rows (default).
        "nc"   : map Misleading -> Not Clear(1).
    """
    samples = []
    if subtask == "st3" and st3_label_space == "2class":
        label_map = LABEL_MAPS["st3_2class"]
        for d in data:
            # Same ST3-applicable gate as gated_mis: evidence_status == "Yes".
            if d.get("evidence_status") != "Yes":
                continue
            text = d["data"]
            eq = d.get("evidence_quality") or ""
            if eq == "Clear":
                label = label_map["Clear"]
            elif eq == "Not Clear":
                label = label_map["Not Clear"]
            elif eq == "Misleading":
                if st3_mis_policy == "nc":
                    label = label_map["Not Clear"]
                else:  # drop
                    continue
            else:
                # blank or unexpected evidence_quality on an applicable row: skip.
                continue
            samples.append((text, label))
        return samples

    if subtask == "st3" and st3_label_space == "3class_na":
        label_map = LABEL_MAPS["st3_3class_na"]
        for d in data:
            text = d["data"]
            eq = d.get("evidence_quality") or ""
            if eq == "Clear":
                label = label_map["Clear"]
            elif eq == "Not Clear":
                label = label_map["Not Clear"]
            elif eq == "Misleading":
                if st3_mis_policy == "nc":
                    label = label_map["Not Clear"]
                else:  # drop
                    continue
            elif eq == "":
                # blank evidence_quality: promise_status==No OR evidence_status==No
                label = label_map["N/A"]
            else:
                continue
            samples.append((text, label))
        return samples

    label_map = LABEL_MAPS[subtask]
    for d in data:
        if subtask == "st1":
            label_str = d["promise_status"]
            text = d["data"]
        elif subtask == "st2":
            if d["promise_status"] != "Yes":
                continue
            label_str = d["evidence_status"]
            text = d["data"]
        elif subtask == "st3":
            if d["evidence_status"] != "Yes":
                continue
            label_str = d["evidence_quality"]
            if not label_str:
                continue
            text = d["data"]
        elif subtask == "st4":
            if d["promise_status"] != "Yes":
                continue
            label_str = d["verification_timeline"]
            if not label_str:
                continue
            text = d["data"]
        if label_str not in label_map:
            continue
        samples.append((text, label_map[label_str]))
    return samples


class ESGDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ── Model ──────────────────────────────────────────────────────────────────────

class BertClassifier(nn.Module):
    def __init__(self, pretrain_model, num_labels, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrain_model)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))


# ── Training ───────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(samples, num_labels):
    counts = [0] * num_labels
    for _, label in samples:
        counts[label] += 1
    total = len(samples)
    weights = [total / (num_labels * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float)


def manual_class_split_ce(logits, labels, weights_per_class):
    """Loss = Σ_c w_c × CE(samples of class c).

    weights_per_class: list/tensor of length num_classes.
    Implements the class-split formulation explicitly (vs. nn.CrossEntropyLoss
    weight= which is per-sample reweighting). Batches that happen to contain
    no samples of a given class simply skip that term.
    """
    loss = torch.zeros(1, device=logits.device, requires_grad=False)
    # make it a leaf-compatible accumulation
    total = None
    for cls_idx, w in enumerate(weights_per_class):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        ce = F.cross_entropy(logits[mask], labels[mask])
        term = w * ce
        total = term if total is None else total + term
    return total if total is not None else loss.squeeze()


def focal_loss(logits, labels, alpha, gamma):
    """Multi-class Focal Loss: FL(p_t) = -α_t * (1-p_t)^γ * log(p_t).

    alpha: per-class weight tensor of shape (num_classes,) on correct device.
    gamma: focusing parameter (>= 0). gamma=0 reduces to weighted CE.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    # Gather true-class probability and log-probability
    pt = probs.gather(1, labels.view(-1, 1)).squeeze(1)
    log_pt = log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
    alpha_t = alpha[labels]
    loss = -alpha_t * ((1.0 - pt) ** gamma) * log_pt
    return loss.mean()


def asl_loss(logits, labels, gamma_neg, gamma_pos, alpha=None):
    """Asymmetric Focal Loss for multi-class via per-class binary cross-entropy.

    Each class output is treated independently (sigmoid, not softmax).
    gamma_neg: focusing exponent for negative (absent) class predictions.
    gamma_pos: focusing exponent for positive (present) class predictions.
    alpha: optional per-class weight tensor applied to the correct-class term.
    """
    num_classes = logits.shape[1]
    targets = F.one_hot(labels, num_classes).float()
    probs = torch.sigmoid(logits)
    log_p   = torch.log(probs + 1e-8)
    log_1mp = torch.log(1.0 - probs + 1e-8)
    focal_pos = (1.0 - probs) ** gamma_pos
    focal_neg = probs ** gamma_neg
    # Per-element losses
    loss_pos = -focal_pos * log_p   * targets
    loss_neg = -focal_neg * log_1mp * (1.0 - targets)
    loss = (loss_pos + loss_neg).sum(dim=1)
    if alpha is not None:
        loss = loss * alpha[labels]
    return loss.mean()


def oversample_minority(samples, target_ratio=1.0, cap=0, seed=42):
    """Random-oversample minority classes toward the majority class count.

    For each class with fewer than `target_ratio * majority_count` samples,
    draw additional copies *with replacement* until it reaches the target.
    Originals are always kept; only duplicates are added.

    target_ratio: desired minority/majority count ratio (1.0 = full balance).
    cap: max duplication multiplier per class (e.g. cap=10 means a class may grow
         to at most 10x its original size); 0 = unlimited. Guards against blowing
         up a class that has only a handful of original samples.
    """
    rng = random.Random(seed)
    by_label = {}
    for s in samples:
        by_label.setdefault(s[1], []).append(s)
    if not by_label:
        return list(samples)

    majority = max(len(g) for g in by_label.values())
    target = int(round(majority * target_ratio))

    out = []
    for group in by_label.values():
        out.extend(group)
        n = len(group)
        if n == 0 or n >= target:
            continue
        need = target - n
        if cap and need > n * (cap - 1):
            need = n * (cap - 1)
        if need > 0:
            out.extend(rng.choices(group, k=need))
    rng.shuffle(out)
    return out


def stratified_sample(samples, max_samples, seed):
    if not max_samples or len(samples) <= max_samples:
        return list(samples)

    rng = random.Random(seed)
    by_label = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)
    for group in by_label.values():
        rng.shuffle(group)

    total = len(samples)
    quotas = {}
    remainders = []
    allocated = 0
    for label, group in by_label.items():
        exact = max_samples * len(group) / total
        quota = min(len(group), int(exact))
        quotas[label] = quota
        allocated += quota
        remainders.append((exact - quota, label))

    for _, label in sorted(remainders, reverse=True):
        if allocated >= max_samples:
            break
        if quotas[label] < len(by_label[label]):
            quotas[label] += 1
            allocated += 1

    selected = []
    for label, group in by_label.items():
        selected.extend(group[: quotas[label]])
    rng.shuffle(selected)
    return selected


def split_train_val(samples, val_ratio, seed):
    if val_ratio <= 0:
        return list(samples), []

    rng = random.Random(seed)
    by_label = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)

    train_samples, val_samples = [], []
    for group in by_label.values():
        rng.shuffle(group)
        if len(group) <= 1:
            train_samples.extend(group)
            continue
        val_count = max(1, int(round(len(group) * val_ratio)))
        val_count = min(val_count, len(group) - 1)
        val_samples.extend(group[:val_count])
        train_samples.extend(group[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def evaluate_model(model, data_loader, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(input_ids, attention_mask)
            if criterion is not None:
                total_loss += criterion(logits, labels).item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(data_loader) if criterion is not None and data_loader else None
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss": avg_loss,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "predictions": all_preds,
        "labels": all_labels,
    }


def train_subtask(
    subtask,
    train_data,
    tokenizer,
    cfg,
    val_ratio=0.2,
    max_samples=None,
    epochs=None,
    max_len=None,
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_epoch_ckpts=True,
    loss_mode="weighted_ce",
    manual_weights=None,
    from_checkpoint=None,
    external_val_data=None,
    focal_gamma=2.0,
    asl_gamma_neg=4.0,
    asl_gamma_pos=1.0,
    oversample=False,
    oversample_target=1.0,
    oversample_cap=0,
    st3_label_space="gated_mis",
    st3_mis_policy="drop",
):
    pretrain_model = cfg["pretrain_model"]
    model_dir = cfg["model_dir"]
    batch_size = cfg["batch_size"]
    grad_accum = cfg["grad_accum"]
    lr = float(cfg["lr"])
    if epochs is None:
        epochs = EPOCHS
    if max_len is None:
        max_len = MAX_LEN

    print(f"\n{'='*60}")
    print(f"Training subtask: {subtask.upper()}")
    print(f"{'='*60}")

    use_st3_3class_na = subtask == "st3" and st3_label_space == "3class_na"
    use_st3_2class = subtask == "st3" and st3_label_space == "2class"
    if use_st3_3class_na:
        label_key = "st3_3class_na"
    elif use_st3_2class:
        label_key = "st3_2class"
    else:
        label_key = subtask
    if use_st3_3class_na:
        print(f"ST3 label space: 3class_na (0=Clear,1=Not Clear,2=N/A) | Misleading policy: {st3_mis_policy}")
    elif use_st3_2class:
        print(f"ST3 label space: 2class (0=Clear,1=Not Clear) gated evidence_status==Yes | Misleading policy: {st3_mis_policy}")

    train_samples = build_subtask_samples(
        train_data, subtask, st3_label_space=st3_label_space, st3_mis_policy=st3_mis_policy
    )
    train_samples = stratified_sample(train_samples, max_samples, SEED)
    num_labels = len(LABEL_MAPS[label_key])
    if external_val_data is not None:
        # Use an external held-out file as val; do not split train.
        val_samples = build_subtask_samples(
            external_val_data, subtask, st3_label_space=st3_label_space, st3_mis_policy=st3_mis_policy
        )
        train_samples = list(train_samples)
    else:
        train_samples, val_samples = split_train_val(train_samples, val_ratio, SEED)

    if oversample:
        before = Counter(l for _, l in train_samples)
        train_samples = oversample_minority(train_samples, oversample_target, oversample_cap, SEED)
        after = Counter(l for _, l in train_samples)
        print(
            f"Oversample (train split only): {dict(before)} -> {dict(after)} "
            f"(target_ratio={oversample_target}, cap={oversample_cap})"
        )

    train_label_counts = Counter(l for _, l in train_samples)
    val_label_counts = Counter(l for _, l in val_samples)
    print(f"Train samples: {len(train_samples)} | Labels: {num_labels} | Dist: {dict(train_label_counts)}")
    print(f"Val samples: {len(val_samples)} | Dist: {dict(val_label_counts)}")

    train_ds = ESGDataset(train_samples, tokenizer, max_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dl = None
    if val_samples:
        val_ds = ESGDataset(val_samples, tokenizer, max_len)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = BertClassifier(pretrain_model, num_labels).to(DEVICE)
    if from_checkpoint:
        model.load_state_dict(torch.load(from_checkpoint, map_location=DEVICE, weights_only=False))
        print(f"Loaded checkpoint → {from_checkpoint}")

    _manual_w = None
    _focal_alpha = None
    _focal_gamma = focal_gamma
    _asl_gamma_neg = asl_gamma_neg
    _asl_gamma_pos = asl_gamma_pos
    _asl_alpha = None

    if loss_mode == "ce":
        criterion = nn.CrossEntropyLoss()
        print("Loss: plain CrossEntropyLoss")
    elif loss_mode == "manual_ce":
        if manual_weights is None:
            manual_weights = compute_class_weights(train_samples, num_labels).tolist()
        _manual_w = manual_weights
        criterion = None
        print(f"Loss: manual_class_split_ce  weights={[round(w,4) for w in _manual_w]}")
    elif loss_mode == "focal":
        if manual_weights is None:
            manual_weights = compute_class_weights(train_samples, num_labels).tolist()
        _focal_alpha = torch.tensor(manual_weights, dtype=torch.float, device=DEVICE)
        criterion = None
        print(f"Loss: focal  gamma={_focal_gamma}  alpha={[round(w,4) for w in manual_weights]}")
    elif loss_mode == "asl":
        if manual_weights is None:
            manual_weights = compute_class_weights(train_samples, num_labels).tolist()
        _asl_alpha = torch.tensor(manual_weights, dtype=torch.float, device=DEVICE)
        criterion = None
        print(f"Loss: asl  gamma_neg={_asl_gamma_neg}  gamma_pos={_asl_gamma_pos}  alpha={[round(w,4) for w in manual_weights]}")
    else:  # weighted_ce (default)
        class_weights = compute_class_weights(train_samples, num_labels).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Loss: weighted CrossEntropyLoss  weights={[round(w.item(),4) for w in class_weights]}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = (len(train_dl) // grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    best_path = os.path.join(model_dir, f"best_{subtask}.pt")
    best_val_macro_f1 = -1.0
    best_epoch = None
    epoch_metrics = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            if _manual_w is not None:
                loss = manual_class_split_ce(logits, labels, _manual_w) / grad_accum
            elif _focal_alpha is not None:
                loss = focal_loss(logits, labels, _focal_alpha, _focal_gamma) / grad_accum
            elif _asl_alpha is not None:
                loss = asl_loss(logits, labels, _asl_gamma_neg, _asl_gamma_pos, _asl_alpha) / grad_accum
            else:
                loss = criterion(logits, labels) / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_dl):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_dl)
        if save_epoch_ckpts:
            epoch_path = os.path.join(model_dir, f"epoch_{subtask}_{epoch:03d}.pt")
            torch.save(model.state_dict(), epoch_path)
            print(f"Saved epoch model → {epoch_path}")
        else:
            epoch_path = None

        epoch_result = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "checkpoint": epoch_path,
        }
        if val_dl is not None:
            # Pass criterion=None for focal/asl modes (loss tracking not needed for val)
            val_result = evaluate_model(model, val_dl, criterion)
            epoch_result.update(
                {
                    "val_loss": val_result["loss"],
                    "val_micro_f1": val_result["micro_f1"],
                    "val_macro_f1": val_result["macro_f1"],
                }
            )
            val_loss_str = f"{val_result['loss']:.4f}" if val_result["loss"] is not None else "n/a"
            print(
                f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | "
                f"Val loss: {val_loss_str} | "
                f"Val micro-F1: {val_result['micro_f1']:.4f} | "
                f"Val macro-F1: {val_result['macro_f1']:.4f}"
            )
            if val_result["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_result["macro_f1"]
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
                print(f"Saved best model → {best_path} (epoch={epoch}, val_macro_f1={best_val_macro_f1:.4f})")
        else:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")
        epoch_metrics.append(epoch_result)

    if val_dl is None:
        torch.save(model.state_dict(), best_path)
        best_epoch = epochs
        print(f"Saved model → {best_path}")
    else:
        model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=False))
        print(f"Loaded best model from epoch {best_epoch} for final train metrics")

    # Evaluate micro-F1 on training set
    train_result = evaluate_model(model, train_dl, criterion)
    inv_map = INV_LABEL_MAPS[label_key]
    present_labels = sorted(set(train_result["labels"] + train_result["predictions"]))
    target_names = [inv_map[i] for i in present_labels]
    print(f"\nTrain Micro-F1: {train_result['micro_f1']:.4f}")
    print(f"Train Macro-F1: {train_result['macro_f1']:.4f}")
    print(
        classification_report(
            train_result["labels"],
            train_result["predictions"],
            labels=present_labels,
            target_names=target_names,
            zero_division=0,
        )
    )

    final_val = None
    if val_dl is not None:
        final_val = evaluate_model(model, val_dl, criterion)
        print(f"Best Val Macro-F1: {final_val['macro_f1']:.4f} (epoch={best_epoch})")

    return {
        "train_micro_f1": train_result["micro_f1"],
        "train_macro_f1": train_result["macro_f1"],
        "best_val_macro_f1": final_val["macro_f1"] if final_val else None,
        "best_val_micro_f1": final_val["micro_f1"] if final_val else None,
        "best_epoch": best_epoch,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_label_distribution": dict(train_label_counts),
        "val_label_distribution": dict(val_label_counts),
        "epoch_metrics": epoch_metrics,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(CONFIGS.keys()), default=_CFG.get("model", "large"))
    parser.add_argument("--stage", choices=["st1", "st2", "st3", "st4"], help="Train only one stage")
    parser.add_argument("--train-path", default=TRAIN_PATH, help="Override training JSON path")
    parser.add_argument("--output", help="Override result JSON path")
    parser.add_argument("--model-dir", help="Override checkpoint output directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Stratified validation ratio for best checkpoint selection (ignored if --val-path is set)")
    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help=(
            "Optional external validation JSON (e.g. data/raw_data/vpesg4k_val_1000.json). "
            "When set, the entire --train-path is used for training (no internal split) "
            "and this file provides the held-out val set for best-epoch selection."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Stratified sample cap after building subtask samples")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--max-len", type=int, default=None, help="Override config max_len")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--lr", type=float, default=None, help="Override per-model lr")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-model batch_size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override per-model grad_accum")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup steps as fraction of total steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--no-epoch-saves", action="store_true", help="Skip per-epoch checkpoint saves; keep best only")
    parser.add_argument(
        "--loss",
        choices=["weighted_ce", "ce", "manual_ce", "focal", "asl"],
        default="weighted_ce",
        help=(
            "Loss function. "
            "weighted_ce: inverse-frequency class weights (default). "
            "ce: plain CrossEntropyLoss. "
            "manual_ce: Loss = Σ_c w_c × CE(class-c samples); set weights with --class-weights. "
            "focal: Focal Loss with per-class alpha from --class-weights and gamma from --focal-gamma. "
            "asl: Asymmetric Focal Loss with --asl-gamma-neg / --asl-gamma-pos."
        ),
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help=(
            "Comma-separated per-class weights for --loss manual_ce/focal/asl, ordered by class index. "
            "For st1/st2 (No=0, Yes=1): e.g. '4.4,1.0'. "
            "If omitted, falls back to inverse-frequency weights."
        ),
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help=(
            "Random-oversample minority classes in the TRAIN split toward the majority "
            "class count (val split untouched). For st3 this lifts Not Clear toward Clear. "
            "Pair with --loss ce since the data is rebalanced by sampling."
        ),
    )
    parser.add_argument(
        "--oversample-target",
        type=float,
        default=1.0,
        help="Target minority/majority ratio for --oversample (1.0 = full balance).",
    )
    parser.add_argument(
        "--oversample-cap",
        type=int,
        default=0,
        help="Max per-class duplication multiplier for --oversample (0 = unlimited).",
    )
    parser.add_argument(
        "--st3-label-space",
        choices=["gated_mis", "3class_na", "2class"],
        default="gated_mis",
        help=(
            "ST3 label space (only affects --stage st3). "
            "gated_mis: legacy gated {Clear,Not Clear,Misleading}, keeps only "
            "evidence_status==Yes rows (default, backward compatible). "
            "3class_na: gate-free {Clear:0,Not Clear:1,N/A:2}, keeps ALL rows, "
            "blank evidence_quality -> N/A; the model learns the N/A boundary. "
            "2class: clean gated {Clear:0,Not Clear:1}, keeps only "
            "evidence_status==Yes rows (same gate as gated_mis), no N/A class "
            "(N/A supplied by the runtime cascade gate); Misleading folded per "
            "--st3-mis-policy."
        ),
    )
    parser.add_argument(
        "--st3-mis-policy",
        choices=["drop", "nc"],
        default="drop",
        help=(
            "Misleading folding policy for --st3-label-space 3class_na / 2class. "
            "drop: drop Misleading rows (default). nc: map Misleading -> Not Clear."
        ),
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focusing gamma for --loss focal (default 2.0).")
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0, help="Negative gamma for --loss asl (default 4.0).")
    parser.add_argument("--asl-gamma-pos", type=float, default=1.0, help="Positive gamma for --loss asl (default 1.0).")
    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint to load before training (fine-tune from C1).",
    )
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    cfg = dict(cfg)
    if args.model_dir:
        cfg["model_dir"] = args.model_dir
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        cfg["grad_accum"] = args.grad_accum
    global SEED
    seed = args.seed if args.seed is not None else SEED
    SEED = seed
    set_seed(seed)
    os.makedirs(cfg["model_dir"], exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Model: {cfg['pretrain_model']}")
    print(f"Batch size: {cfg['batch_size']} (grad accum: {cfg['grad_accum']}, effective: {cfg['batch_size'] * cfg['grad_accum']})")
    if args.val_path:
        print(f"Validation: external file {args.val_path} (ignoring --val-ratio)")
    else:
        print(f"Validation ratio: {args.val_ratio}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")

    train_data = load_data(args.train_path)
    print(f"Train records: {len(train_data)}")

    external_val_data = None
    if args.val_path:
        external_val_data = load_data(args.val_path)
        print(f"External val: {args.val_path} ({len(external_val_data)} records)")

    tokenizer = AutoTokenizer.from_pretrained(cfg["pretrain_model"])

    effective_val_ratio = 0.0 if external_val_data is not None else args.val_ratio
    results = {}
    stages = [args.stage] if args.stage else ["st1", "st2", "st3", "st4"]
    for subtask in stages:
        results[subtask] = train_subtask(
            subtask,
            train_data,
            tokenizer,
            cfg,
            val_ratio=effective_val_ratio,
            max_samples=args.max_samples,
            epochs=args.epochs,
            max_len=args.max_len,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            save_epoch_ckpts=not args.no_epoch_saves,
            loss_mode=args.loss,
            manual_weights=[float(w) for w in args.class_weights.split(",")] if args.class_weights else None,
            from_checkpoint=args.from_checkpoint,
            external_val_data=external_val_data,
            focal_gamma=args.focal_gamma,
            asl_gamma_neg=args.asl_gamma_neg,
            asl_gamma_pos=args.asl_gamma_pos,
            oversample=args.oversample,
            oversample_target=args.oversample_target,
            oversample_cap=args.oversample_cap,
            st3_label_space=args.st3_label_space,
            st3_mis_policy=args.st3_mis_policy,
        )

    print("\n" + "=" * 60)
    print("Final Results (Train Micro-F1)")
    print("=" * 60)
    weighted_sum = 0
    for st, f1 in results.items():
        score = f1["best_val_macro_f1"] if f1["best_val_macro_f1"] is not None else f1["train_micro_f1"]
        print(f"{st.upper()}: {score:.4f} (weight={TASK_WEIGHTS[st]})")
        weighted_sum += score * TASK_WEIGHTS[st]
    if len(results) == len(TASK_WEIGHTS):
        print(f"\nWeighted Score: {weighted_sum:.4f}")
    else:
        print(f"\nPartial Weighted Contribution: {weighted_sum:.4f}")

    out_path = args.output or f"/tmp2/howard/side_project/esg_contest/results/train/bert_{args.model}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "subtasks": results,
                "selection_metric": "val_macro_f1" if (args.val_ratio > 0 or args.val_path) else "train_micro_f1",
                "val_path": args.val_path,
                "weighted_score": weighted_sum if len(results) == len(TASK_WEIGHTS) else None,
                "partial_weighted_contribution": weighted_sum,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
