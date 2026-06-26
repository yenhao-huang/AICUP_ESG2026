#!/usr/bin/env python3
"""Soft-vote ensemble inference for Stage 2 (evidence_status).

Runs each ST2 ensemble member checkpoint over the SAME input, averages the
per-class softmax probabilities across members, then argmax -> final label.
Reuses this stage's ``pred_by_bert.predict_rows`` (identical model / checkpoint
loading / tokenisation). Output CSV uses the ST2 schema, so it drops into
merge_pipeline.py / eval.py unchanged.

Members:
  models/submission/stage2/ensemble_st2_mix_a2_b3_seed<S>/best_st2.pt
  models/ensemble_models/stage2/<dataset>/seed<S>/<loss_tag>/best_st2.pt

Example
-------
  .venv/bin/python core/service/predict/stage2/soft_vote.py \
    --data data/raw_data/vpesg4k_val_1000.json \
    --ckpt-glob 'models/submission/stage2/*/best_st2.pt' \
    --output results/predict/stage2/ensemble/submit/softvote.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from core.service.predict.stage2 import pred_by_bert as st2
from core.service.predict.stage2 import schema as st2_schema

CLASS_ORDER = ["No", "Yes"]          # index 0 = No, 1 = Yes; N/A comes from the cascade
MODEL_ALIASES = {
    "chinese-roberta-wwm-ext-large": "hfl/chinese-roberta-wwm-ext-large",
    "chinese-roberta-wwm-ext": "hfl/chinese-roberta-wwm-ext",
    "roberta-large": "hfl/chinese-roberta-wwm-ext-large",
    "roberta-base": "hfl/chinese-roberta-wwm-ext",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            return [dict(r) for r in csv.DictReader(f)]
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must be a JSON list or CSV")
    return rows


def parse_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_reason(reason: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for chunk in str(reason).split(";"):
        if "=" in chunk:
            k, _, v = chunk.partition("=")
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def resolve_checkpoints(ap, finetune_paths, ckpt_glob) -> list[Path]:
    ckpts = list(finetune_paths)
    if ckpt_glob:
        ckpts += [Path(p) for p in sorted(glob.glob(ckpt_glob))]
    seen, uniq = set(), []
    for c in ckpts:
        if str(c) not in seen:
            seen.add(str(c)); uniq.append(Path(c))
    if not uniq:
        ap.error("no checkpoints: pass --finetune-path (repeatable) and/or --ckpt-glob")
    for c in uniq:
        if not c.is_file():
            ap.error(f"checkpoint not found: {c}")
    if len(uniq) == 1:
        print("[warn] only 1 checkpoint -- soft vote == single model.", file=sys.stderr)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="Input JSON/CSV with id + data text.")
    ap.add_argument("--output", type=Path, required=True, help="Output soft-vote CSV.")
    ap.add_argument("--finetune-path", type=Path, action="append", default=[],
                    help="A member checkpoint; repeat per member.")
    ap.add_argument("--ckpt-glob", default=None, help="Glob expanded to member checkpoints.")
    ap.add_argument("--model", default="hfl/chinese-roberta-wwm-ext-large")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    ap.add_argument("--limit", type=int, default=None, help="Smoke-test row limit.")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--members-out", type=Path, default=None,
                    help="Per-member score CSV (default: <output>.members.csv). Set '' to skip.")
    args = ap.parse_args()

    ckpts = resolve_checkpoints(ap, args.finetune_path, args.ckpt_glob)
    model_name = MODEL_ALIASES.get(args.model, args.model)
    device = parse_device(args.device)
    rows = load_rows(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]
    ids = [str(r.get("id", i + 1)) for i, r in enumerate(rows)]

    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    member_rows: list[dict[str, str]] = []
    for i, ckpt in enumerate(ckpts, 1):
        print(f"### member {i}/{len(ckpts)}: {ckpt}", flush=True)
        member = ckpt.parent.name
        preds = st2.predict_rows(rows, model_name, ckpt, device, args.batch_size,
                                 args.max_len, text_mode="data", local_files_only=True)
        for rid, row in preds.items():
            r = parse_reason(row["postprocess_reason"])
            p = {"No": r["score_no"], "Yes": r["score_yes"]}
            acc = sums.setdefault(rid, {k: 0.0 for k in CLASS_ORDER})
            for k in CLASS_ORDER:
                acc[k] += p[k]
            counts[rid] = counts.get(rid, 0) + 1
            member_rows.append({"id": rid, "member": member,
                                "score_no": f"{p['No']:.8f}", "score_yes": f"{p['Yes']:.8f}",
                                "pred": row["evidence_status"]})

    out_rows = []
    for rid in ids:
        if rid not in sums:
            continue
        n = counts[rid]
        avg = {k: sums[rid][k] / n for k in CLASS_ORDER}
        label = max(CLASS_ORDER, key=lambda k: avg[k])
        reason = ";".join(f"score_{k.lower()}={avg[k]:.8f}" for k in CLASS_ORDER) + f";n_members={n}"
        out_rows.append({
            "id": rid, "evidence_status": label, "evidence_status_raw": str(CLASS_ORDER.index(label)),
            "filter_passed": "yes", "prediction_source": "soft_vote", "postprocess_reason": reason,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(st2_schema.OUTPUT_COLUMNS), extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    members_out = args.members_out
    if members_out is None:
        members_out = args.output.with_name(args.output.stem + ".members.csv")
    if str(members_out):
        with members_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "member", "score_no", "score_yes", "pred"])
            w.writeheader()
            w.writerows(member_rows)

    dist = dict(sorted(Counter(r["evidence_status"] for r in out_rows).items()))
    print(json.dumps({"stage": 2, "output": str(args.output),
                      "members_out": str(members_out) if str(members_out) else None,
                      "rows": len(out_rows), "members": len(ckpts), "label_dist": dist},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
