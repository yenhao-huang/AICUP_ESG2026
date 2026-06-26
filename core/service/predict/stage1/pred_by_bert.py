#!/usr/bin/env python3
"""Run Stage 1 promise_status inference and write the prediction CSV."""

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

from core.service.predict.stage1 import schema


LABEL_BY_ID = {0: "No", 1: "Yes"}
FIELDNAMES = list(schema.OUTPUT_COLUMNS)
MODEL_ALIASES = {
    "chinese-roberta-wwm-ext-large": "hfl/chinese-roberta-wwm-ext-large",
    "chinese-roberta-wwm-ext": "hfl/chinese-roberta-wwm-ext",
    "roberta-large": "hfl/chinese-roberta-wwm-ext-large",
    "roberta-base": "hfl/chinese-roberta-wwm-ext",
}


class Stage1Dataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, text_col: str, max_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        enc = self.tokenizer(
            str(row.get(self.text_col, "")),
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


class BertStage1Classifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1, local_files_only: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            _fast_init=False,
            local_files_only=local_files_only,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))


def resolve_model_name(value: str) -> str:
    return MODEL_ALIASES.get(value, value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            return [dict(row) for row in csv.DictReader(f)]
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list or CSV rows")
    return rows


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(checkpoint)


def predict_rows(
    *,
    rows: list[dict[str, Any]],
    model_name: str,
    checkpoint_path: Path | None,
    device: torch.device,
    batch_size: int,
    max_len: int,
    text_col: str,
    local_files_only: bool,
) -> list[dict[str, str]]:
    local_files_only = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import transformers.safetensors_conversion as safetensors_conversion

        safetensors_conversion.auto_conversion = lambda *args, **kwargs: (None, None)
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = BertStage1Classifier(model_name, local_files_only=local_files_only).to(device)
    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, device)
    model.eval()

    dataset = Stage1Dataset(rows, tokenizer, text_col, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions: list[dict[str, str]] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1).cpu()
            pred_ids = torch.argmax(probs, dim=-1).tolist()
            for row_id, pred_id, prob in zip(batch["id"], pred_ids, probs):
                score_no = float(prob[0])
                score_yes = float(prob[1])
                predictions.append(
                    {
                        "id": str(row_id),
                        "promise_status": LABEL_BY_ID[int(pred_id)],
                        "score_yes": f"{score_yes:.8f}",
                        "score_no": f"{score_no:.8f}",
                        "raw_prediction": str(int(pred_id)),
                    }
                )
    return predictions


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        row_id = row.get("id", "").strip()
        status = row.get("promise_status", "").strip()
        if not row_id:
            errors.append(f"row {index}: id is empty")
        elif row_id in seen_ids:
            errors.append(f"row {index}: duplicate id {row_id}")
        seen_ids.add(row_id)
        if status not in {"Yes", "No"}:
            errors.append(f"row {index}: invalid promise_status {status!r}")
    return errors


def parse_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Input JSON/CSV containing id and data text.")
    parser.add_argument("--output", type=Path, required=True, help="Output Stage 1 CSV path.")
    parser.add_argument(
        "--model",
        default="hfl/chinese-roberta-wwm-ext-large",
        help="Backbone model or alias: chinese-roberta-wwm-ext-large, chinese-roberta-wwm-ext.",
    )
    parser.add_argument("--finetune-path", type=Path, default=None, help="Optional finetuned checkpoint path.")
    parser.add_argument("--mode", choices=["finetune", "no_finetune"], default=None)
    parser.add_argument("--text-col", default="data")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--local-files-only", action="store_true", help="Load model/tokenizer from local cache only.")
    args = parser.parse_args()

    if args.mode == "finetune" and args.finetune_path is None:
        parser.error("--mode finetune requires --finetune-path")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = resolve_model_name(args.model)
    mode = args.mode or ("finetune" if args.finetune_path else "no_finetune")
    rows = load_rows(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]

    predictions = predict_rows(
        rows=rows,
        model_name=model_name,
        checkpoint_path=args.finetune_path,
        device=parse_device(args.device),
        batch_size=args.batch_size,
        max_len=args.max_len,
        text_col=args.text_col,
        local_files_only=args.local_files_only,
    )

    run_id = args.run_id or f"stage1_{Path(model_name).name}_{mode}"
    for row in predictions:
        row.update(
            {
                "model_family": model_name,
                "mode": mode,
                "finetune_path": str(args.finetune_path or ""),
                "prompt_id": "",
                "prompt_path": "",
                "run_id": run_id,
                "source": "model_inference",
            }
        )

    write_csv(args.output, predictions)
    errors = validate_rows(predictions)
    summary = {
        "output": str(args.output),
        "rows": len(predictions),
        "model": model_name,
        "mode": mode,
        "finetune_path": str(args.finetune_path or ""),
        "validation": {"ok": not errors, "errors": errors},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
