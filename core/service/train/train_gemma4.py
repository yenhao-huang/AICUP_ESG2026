"""
QLoRA SFT for /models/unsloth-gemma-4-12b (Gemma4Unified, ~12B).

Multitask ST1+ST2 generative head: the model reads ONLY the raw `data` field and
emits a single JSON object:

    {"promise_status": ..., "promise_str": ..., "evidence_status": ..., "evidence_str": ...}

Data-use compliance
-------------------
- Model input is the raw `data` field only.
- promise_string / evidence_string are used solely as SFT *targets* (labels);
  they never enter the prompt. At inference the model sees only `data`.

Stack: trl SFTTrainer + peft LoRA + bitsandbytes 4-bit (QLoRA).

Usage
-----
    # Inspect the exact formatted input/target without loading the model:
    python core/train/train_gemma4.py --config configs/train/gemma4.yml --dry-run --num-show 3

    # Train (run only when the GPU is free; the script does NOT free GPUs):
    python core/train/train_gemma4.py --config configs/train/gemma4.yml
"""

import argparse
import json
import pathlib

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT = pathlib.Path(__file__).parents[3]
_DEFAULT_CONFIG = _ROOT / "configs" / "train" / "gemma4.yml"

# ── Prompt (data-only input) ────────────────────────────────────────────────────

SYSTEM = """你是一位專業的 ESG 永續報告分析師。
你會收到一段 ESG 報告原文（data）。請依序完成兩項判斷，並只輸出一個 JSON 物件：

1. promise_status：這段文字是否包含明確的企業未來行動承諾。
   - "Yes"：有明確行動計劃／量化目標／時程，或「承諾、致力、預計、自X年起」等意向性陳述。
   - "No"：只是描述事實、現況、背景或第三方聲明，無未來目標或行動意向。
2. promise_str：若 promise_status 為 "Yes"，從原文中「逐字」抽出構成承諾的句子；若為 "No"，輸出空字串 ""。
3. evidence_status：該承諾是否有具體的支持證據或可驗證的行動細節。
   - "Yes"：原文有數字、可量化指標、明確行動步驟或已實施措施作為佐證。
   - "No"：只有方向性／意向性描述，缺乏具體佐證。
   - "N/A"：promise_status 為 "No"（非承諾，不適用）。
4. evidence_str：若 evidence_status 為 "Yes"，逐字抽出作為證據的句子；否則輸出空字串 ""。

規則：
- promise_str 與 evidence_str 必須是 data 原文中存在的片段，不可改寫、不可杜撰。
- 只輸出 JSON，不要任何解釋或多餘文字。
- 輸出格式：{"promise_status":"...","promise_str":"...","evidence_status":"...","evidence_str":"..."}"""

# Gemma turn format (GemmaTokenizer has no chat template; system folds into the
# first user turn). The tokenizer prepends <bos>, so we do not write it literally.
PROMPT_TEMPLATE = "<start_of_turn>user\n{system}\n\n原文：{data}<end_of_turn>\n<start_of_turn>model\n"
COMPLETION_TEMPLATE = "{target}<end_of_turn>\n"
# trl 1.5 applies completion-only loss automatically for the prompt/completion
# dataset format, so prompt tokens are masked and only the JSON target trains.

# ── Target construction (ST1 + ST2) ─────────────────────────────────────────────

def build_target(record: dict) -> dict:
    """Build the JSON target from annotation fields (label only, never input)."""
    promise_status = record.get("promise_status", "No")
    if promise_status == "Yes":
        promise_str = record.get("promise_string", "") or ""
        evidence_status = record.get("evidence_status") or "No"
        # Defensive: only Yes/No are meaningful when there is a promise.
        if evidence_status not in ("Yes", "No"):
            evidence_status = "No"
        evidence_str = (record.get("evidence_string", "") or "") if evidence_status == "Yes" else ""
    else:
        promise_status = "No"
        promise_str = ""
        evidence_status = "N/A"
        evidence_str = ""
    return {
        "promise_status": promise_status,
        "promise_str": promise_str,
        "evidence_status": evidence_status,
        "evidence_str": evidence_str,
    }


def target_to_json(target: dict) -> str:
    # Compact, stable key order, keep CJK characters readable.
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def format_example(record: dict) -> dict:
    """Return prompt/completion/target_json/text for one record (input = data only)."""
    data_text = record.get("data", "") or ""
    prompt = PROMPT_TEMPLATE.format(system=SYSTEM, data=data_text)
    target_json = target_to_json(build_target(record))
    completion = COMPLETION_TEMPLATE.format(target=target_json)
    return {
        "prompt": prompt,
        "completion": completion,
        "target_json": target_json,
        "text": prompt + completion,
    }


# ── Data ────────────────────────────────────────────────────────────────────────

def load_records(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_dataset(records: list):
    from datasets import Dataset
    rows = []
    for r in records:
        ex = format_example(r)
        rows.append({"prompt": ex["prompt"], "completion": ex["completion"]})
    return Dataset.from_list(rows)


# ── Config ──────────────────────────────────────────────────────────────────────

def load_config(path: pathlib.Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str) -> str:
    pp = pathlib.Path(p)
    return str(pp if pp.is_absolute() else (_ROOT / pp))


# ── Dry run (inspect input/target without loading the model) ─────────────────────

def dry_run(records: list, num_show: int) -> None:
    from collections import Counter
    print(f"[dry-run] {len(records)} records from train_path")
    statuses = Counter()
    for r in records:
        t = build_target(r)
        statuses[(t["promise_status"], t["evidence_status"])] += 1
    print(f"[dry-run] target (promise_status, evidence_status) distribution: {dict(statuses)}")
    for i, r in enumerate(records[:num_show]):
        ex = format_example(r)
        print("\n" + "=" * 80)
        print(f"[example {i}] id={r.get('id')}")
        print("-" * 80 + "\n[FULL TEXT FED TO MODEL]\n")
        print(ex["text"])
        print("-" * 80)
        print(f"[TARGET JSON only]\n{ex['target_json']}")


# ── Train ───────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    model_path = cfg["model_path"]
    output_dir = resolve_path(cfg["output_dir"])
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    qc = cfg["quant"]
    compute_dtype = getattr(torch, qc.get("bnb_4bit_compute_dtype", "bfloat16"))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qc.get("load_in_4bit", True),
        bnb_4bit_quant_type=qc.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=qc.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_base_model(model_path, bnb_config, compute_dtype, cfg.get("device", "cuda:0"))
    model.config.use_cache = False

    lcfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lcfg.get("r", 16),
        lora_alpha=lcfg.get("alpha", 32),
        lora_dropout=lcfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lcfg.get("target_modules", "all-linear"),
    )

    records = load_records(resolve_path(cfg["train_path"]))
    train_ds = build_dataset(records)

    # Optional held-out val set for per-epoch evaluation during training.
    # Same completion_only_loss space as train, so eval_loss = val completion loss.
    eval_ds = None
    val_path = cfg.get("val_path")
    if val_path:
        eval_ds = build_dataset(load_records(resolve_path(val_path)))
        print(f"[train] eval-during-training on {val_path} ({len(eval_ds)} rows)")

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        per_device_eval_batch_size=cfg.get("eval_batch_size", cfg.get("batch_size", 1)),
        gradient_accumulation_steps=cfg.get("grad_accum", 16),
        learning_rate=float(cfg.get("lr", 2e-4)),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.0),
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy=cfg.get("save_strategy", "epoch"),
        eval_strategy=("epoch" if eval_ds is not None else "no"),
        max_length=cfg.get("max_len", 1024),
        bf16=cfg.get("bf16", True),
        seed=cfg.get("seed", 42),
        packing=False,
        completion_only_loss=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "base_model": model_path,
        "task": "st1_st2_json_generative",
        "input_field": "data",
        "target_keys": ["promise_status", "promise_str", "evidence_status", "evidence_str"],
        "num_train": len(records),
        "lora": lcfg,
        "quant": qc,
    }
    with open(pathlib.Path(output_dir) / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[train] adapter + tokenizer + meta saved to {output_dir}")


def load_base_model(path, bnb_config, dtype, device):
    """Load the Gemma4Unified causal/conditional-generation model in 4-bit.

    Gemma4Unified is a *ConditionalGeneration multimodal class; AutoModelForCausalLM
    is usually not registered for it, so fall back to the image-text-to-text auto
    class (text-only forward still accepts input_ids/labels).
    """
    common = dict(
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map={"": device},
    )
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(path, **common)
    except (ValueError, KeyError, TypeError) as exc:
        print(f"[train] AutoModelForCausalLM failed ({exc}); trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(path, **common)


# ── Entry ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA SFT for gemma-4-12b ST1+ST2 JSON head")
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print formatted input/target for a few records and exit (no model load).")
    ap.add_argument("--num-show", type=int, default=3, help="Examples to print in --dry-run.")
    args = ap.parse_args()

    cfg = load_config(pathlib.Path(args.config))
    records = load_records(resolve_path(cfg["train_path"]))

    if args.dry_run:
        dry_run(records, args.num_show)
        return

    train(cfg)


if __name__ == "__main__":
    main()
