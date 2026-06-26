#!/usr/bin/env python3
"""Run independent Stage 2 BERT evidence_status prediction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
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

from core.service.predict.stage2 import schema


FIELDNAMES = list(schema.OUTPUT_COLUMNS)
LABEL_BY_ID = {0: "No", 1: "Yes"}
MODEL_ALIASES = {
    "chinese-roberta-wwm-ext-large": "hfl/chinese-roberta-wwm-ext-large",
    "chinese-roberta-wwm-ext": "hfl/chinese-roberta-wwm-ext",
    "roberta-large": "hfl/chinese-roberta-wwm-ext-large",
    "roberta-base": "hfl/chinese-roberta-wwm-ext",
}


class TextDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_len: int, text_mode: str):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_mode = text_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        text = build_text(row, self.text_mode)
        enc = self.tokenizer(
            text,
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


class BertClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1, local_files_only: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            _fast_init=False,
            local_files_only=local_files_only,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))


def resolve_model_name(value: str) -> str:
    return MODEL_ALIASES.get(value, value)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_rows(path: Path) -> list[dict[str, Any]]:
    source = resolve_path(path)
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8-sig") as f:
            return [dict(row) for row in csv.DictReader(f)]
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{source} must contain a JSON list or CSV rows")
    return [dict(row) for row in rows]


def load_gate(path: Path | None, column: str) -> dict[str, str]:
    if path is None:
        return {}
    return {str(row.get("id", "")): str(row.get(column, "")) for row in load_rows(path)}


def build_text(row: dict[str, Any], text_mode: str) -> str:
    if text_mode == "train_compat":
        return str(row.get("data", "")) + "[SEP]" + str(row.get("promise_string", ""))
    return str(row.get("data", ""))


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(checkpoint)


def predict_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    max_len: int,
    text_mode: str,
    local_files_only: bool,
) -> dict[str, dict[str, str]]:
    local_files_only = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import transformers.safetensors_conversion as safetensors_conversion

        safetensors_conversion.auto_conversion = lambda *args, **kwargs: (None, None)
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
    model = BertClassifier(model_name, num_labels=2, local_files_only=local_files_only).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    dataset = TextDataset(rows, tokenizer, max_len, text_mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions: dict[str, dict[str, str]] = {}
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=-1).cpu()
            pred_ids = torch.argmax(probs, dim=-1).tolist()
            for row_id, pred_id, prob in zip(batch["id"], pred_ids, probs):
                label = LABEL_BY_ID[int(pred_id)]
                predictions[str(row_id)] = {
                    "evidence_status": label,
                    "evidence_status_raw": str(int(pred_id)),
                    "filter_passed": "yes",
                    "prediction_source": "bert",
                    "postprocess_reason": f"score_no={float(prob[0]):.8f};score_yes={float(prob[1]):.8f}",
                }
    return predictions


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finetune-path", type=Path, required=True)
    parser.add_argument("--stage1-csv", type=Path, default=None, help="Optional Stage 1 gate CSV.")
    parser.add_argument("--stage1-gate-col", default="promise_status")
    parser.add_argument("--model", default="hfl/chinese-roberta-wwm-ext-large")
    parser.add_argument("--text-mode", choices=["data", "train_compat"], default="data")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = load_rows(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]
    gate = load_gate(args.stage1_csv, args.stage1_gate_col)
    passed_rows = [row for row in rows if not gate or gate.get(str(row.get("id", ""))) == "Yes"]
    preds = predict_rows(
        passed_rows,
        resolve_model_name(args.model),
        args.finetune_path,
        parse_device(args.device),
        args.batch_size,
        args.max_len,
        args.text_mode,
        args.local_files_only,
    )

    output_rows = []
    for row in rows:
        row_id = str(row.get("id", ""))
        if gate and gate.get(row_id) != "Yes":
            output_rows.append(
                {
                    "id": row_id,
                    "evidence_status": "N/A",
                    "evidence_status_raw": "",
                    "filter_passed": "no",
                    "prediction_source": "stage1_filter",
                    "postprocess_reason": "promise_status_not_yes",
                }
            )
        else:
            output_rows.append({"id": row_id, **preds[row_id]})

    write_csv(args.output, output_rows)
    print(json.dumps({"stage": "stage2", "output": str(args.output), "rows": len(output_rows), "predicted": len(preds)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
