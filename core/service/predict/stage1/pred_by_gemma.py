#!/usr/bin/env python3
"""Predict Stage 1 directly with the Gemma ST1+ST2 JSON adapter.

Every input row is sent to Gemma. There is no BERT, fallback, confidence gate,
or cascade gate in this predictor.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from core.service.predict.stage1 import schema  # noqa: E402
from core.service.train import train_gemma4 as T  # noqa: E402

DEFAULT_GEMMA_BASE = "models/gemma/base/unsloth-gemma-4-12b"
DEFAULT_GEMMA_ADAPTER = "models/submission/st12_fallback/gemma4_st12_mix"
VALID_LABELS = {"Yes", "No"}


def resolve_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_rows(path: Path) -> list[dict[str, Any]]:
    source = resolve_path(path)
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8-sig") as f:
            return [dict(r) for r in csv.DictReader(f)]
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{source} must contain a JSON list or CSV rows")
    return [dict(r) for r in rows]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(schema.OUTPUT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_model(base: str, adapter: str | None, device: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter or base)
    common = dict(quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": device})
    try:
        model = AutoModelForCausalLM.from_pretrained(base, **common)
    except (ValueError, KeyError, TypeError):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(base, **common)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def parse_json_output(raw: str) -> dict[str, Any]:
    cut = raw.split("<end_of_turn>")[0].strip()
    try:
        obj = json.loads(cut)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", cut, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
    return {}


@torch.no_grad()
def predict_row(text: str, tokenizer, model, device: str, max_new_tokens: int) -> tuple[dict[str, Any], str]:
    prompt = T.PROMPT_TEMPLATE.format(system=T.SYSTEM, data=text or "")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [i for i in {tokenizer.eos_token_id, eot} if i is not None and i >= 0]
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=eos_ids or tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
    return parse_json_output(raw), raw.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gemma-base", default=DEFAULT_GEMMA_BASE)
    parser.add_argument("--gemma-adapter", default=DEFAULT_GEMMA_ADAPTER)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--text-col", default="data")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default="gemma_st1_direct")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    adapter = None if args.no_adapter else args.gemma_adapter
    rows = load_rows(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"[gemma-st1] rows={len(rows)} base={args.gemma_base} adapter={adapter} device={device}")
    tokenizer, model = build_model(args.gemma_base, adapter, device)

    out_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        rid = str(row.get("id", index))
        obj, raw = predict_row(str(row.get(args.text_col, "")), tokenizer, model, device, args.max_new_tokens)
        label = str(obj.get("promise_status") or "").strip()
        label = label if label in VALID_LABELS else "No"
        out_rows.append(
            {
                "id": rid,
                "promise_status": label,
                "score_yes": "",
                "score_no": "",
                "model_family": str(args.gemma_base),
                "mode": "gemma_direct",
                "finetune_path": "" if adapter is None else str(adapter),
                "prompt_id": "",
                "prompt_path": "",
                "run_id": args.run_id,
                "source": "gemma_direct",
                "raw_prediction": raw,
            }
        )
        if index % 20 == 0 or index == len(rows):
            print(f"[gemma-st1] {index}/{len(rows)}", flush=True)

    write_csv(args.output, out_rows)
    print(json.dumps({
        "stage": 1,
        "rows": len(out_rows),
        "output": str(args.output),
        "label_dist": dict(sorted(Counter(r["promise_status"] for r in out_rows).items())),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
