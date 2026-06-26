#!/usr/bin/env python3
"""Run Stage 3 evidence_quality inference from one multitask BERT checkpoint.

This predictor always runs the Stage 3 head for every input row. It does not
read a Stage 2 CSV and never rewrites rows to N/A based on cascade gating.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from core.service.predict.stage3 import schema  # noqa: E402


FIELDNAMES = list(schema.OUTPUT_COLUMNS)
ST1_CLASSES = 2
ST2_CLASSES = 2
ST3_LABELS = {
    2: {0: "Clear", 1: "Not Clear"},
    3: {0: "Clear", 1: "Not Clear", 2: "Misleading"},
}
ST3_SCORE_KEYS = {
    2: ("score_clear", "score_not_clear"),
    3: ("score_clear", "score_not_clear", "score_misleading"),
}
MODEL_ALIASES = {
    "chinese-roberta-wwm-ext-large": "hfl/chinese-roberta-wwm-ext-large",
    "chinese-roberta-wwm-ext": "hfl/chinese-roberta-wwm-ext",
    "roberta-large": "hfl/chinese-roberta-wwm-ext-large",
    "roberta-base": "hfl/chinese-roberta-wwm-ext",
}


class TextDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        enc = self.tokenizer(
            str(row.get("data", "")),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "id": str(row.get("id", index + 1)),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


class MultiTaskBertClassifier(nn.Module):
    """Shared BERT encoder with ST1, ST2, and ST3 classification heads."""

    def __init__(self, model_name: str, st3_classes: int, dropout: float = 0.1, local_files_only: bool = True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            _fast_init=False,
            local_files_only=local_files_only,
        )
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_st1 = nn.Linear(hidden, ST1_CLASSES)
        self.head_st2 = nn.Linear(hidden, ST2_CLASSES)
        self.head_st3 = nn.Linear(hidden, st3_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        return self.head_st3(pooled)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def resolve_model_name(value: str) -> str:
    return MODEL_ALIASES.get(value, value)


def parse_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    source = resolve_path(path)
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8-sig") as f:
            return [dict(row) for row in csv.DictReader(f)]
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{source} must contain a JSON list or CSV rows")
    return [dict(row) for row in rows]


def read_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "model_state"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    if "head_st3.weight" not in checkpoint and "classifier.weight" in checkpoint:
        checkpoint = dict(checkpoint)
        checkpoint["head_st3.weight"] = checkpoint.pop("classifier.weight")
        if "classifier.bias" in checkpoint:
            checkpoint["head_st3.bias"] = checkpoint.pop("classifier.bias")
    return checkpoint


def st3_class_count(state: dict[str, torch.Tensor]) -> int:
    if "head_st3.weight" not in state:
        raise RuntimeError(
            "Checkpoint has no 'head_st3.weight' or 'classifier.weight'; expected a multitask or exported ST3 checkpoint."
        )
    n_classes = int(state["head_st3.weight"].shape[0])
    if n_classes not in ST3_LABELS:
        raise RuntimeError(f"Unsupported ST3 head size {n_classes}; expected 2 or 3.")
    return n_classes


def load_multitask_checkpoint(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing, _unexpected = model.load_state_dict(state, strict=False)
    required = [key for key in missing if key.startswith(("bert.", "head_st3."))]
    if required:
        raise RuntimeError(f"Checkpoint is missing required ST3 weights: {required[:8]}")


def disable_transformers_conversion() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import transformers.safetensors_conversion as safetensors_conversion

        safetensors_conversion.auto_conversion = lambda *args, **kwargs: (None, None)
    except Exception:
        pass


def decide_label(prob: torch.Tensor, nc_tau: float | None) -> int:
    if nc_tau is None:
        return int(torch.argmax(prob).item())
    if float(prob[1]) > nc_tau:
        return 1
    if prob.shape[0] == 2:
        return 0
    return 0 if float(prob[0]) >= float(prob[2]) else 2


def predict_rows(
    *,
    rows: list[dict[str, Any]],
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    max_len: int,
    local_files_only: bool,
    nc_tau: float | None,
) -> list[dict[str, str]]:
    disable_transformers_conversion()
    state = read_state_dict(checkpoint_path, device)
    n_classes = st3_class_count(state)
    labels = ST3_LABELS[n_classes]
    score_keys = ST3_SCORE_KEYS[n_classes]
    print(f"[st3] multitask head: {n_classes} classes {list(labels.values())}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
    model = MultiTaskBertClassifier(model_name, n_classes, local_files_only=local_files_only).to(device)
    load_multitask_checkpoint(model, state)
    model.eval()

    dataset = TextDataset(rows, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions: list[dict[str, str]] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=-1).cpu()
            for row_id, prob in zip(batch["id"], probs):
                pred_id = decide_label(prob, nc_tau)
                reason = ";".join(f"{score_keys[i]}={float(prob[i]):.8f}" for i in range(n_classes))
                predictions.append(
                    {
                        "id": str(row_id),
                        "evidence_quality": labels[pred_id],
                        "evidence_quality_raw": str(pred_id),
                        "evidence_quality_source": "bert_multitask",
                        "evidence_quality_reason": reason,
                    }
                )
    return predictions


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Input JSON/CSV containing id and data text.")
    parser.add_argument("--output", type=Path, required=True, help="Output Stage 3 CSV path.")
    parser.add_argument("--finetune-path", type=Path, required=True, help="Multitask ST3 checkpoint.")
    parser.add_argument("--model", default="hfl/chinese-roberta-wwm-ext-large")
    parser.add_argument("--nc-tau", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="pred_by_multitask_stage3")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = load_rows(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]

    predictions = predict_rows(
        rows=rows,
        model_name=resolve_model_name(args.model),
        checkpoint_path=args.finetune_path,
        device=parse_device(args.device),
        batch_size=args.batch_size,
        max_len=args.max_len,
        local_files_only=args.local_files_only,
        nc_tau=args.nc_tau,
    )

    write_csv(args.output, predictions)
    print(
        json.dumps(
            {
                "stage": "stage3",
                "mode": "finetune_multitask_st3",
                "run_id": args.run_id,
                "output": str(resolve_path(args.output)),
                "rows": len(predictions),
                "predicted": len(predictions),
                "finetune_path": str(resolve_path(args.finetune_path)),
                "label_dist": dict(sorted(Counter(row["evidence_quality"] for row in predictions).items())),
                "stage2_gate": "disabled",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
