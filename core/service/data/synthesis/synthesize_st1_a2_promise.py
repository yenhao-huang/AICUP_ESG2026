#!/usr/bin/env python3
"""Generate loop001 ST1 A2 promise_string-derived positive rows."""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from core.service.data.synthesis.common import (
    call_openai_compatible,
    display_path,
    label_counts,
    looks_like_leak,
    normalize_space,
    project,
    read_json,
    resolve,
    sample_with_replacement,
    write_json_atomic,
)


POOL_NAME = "synth_st1_a2_promise_pool.json"
MANIFEST_NAME = "synth_st1_a2_manifest.json"
MIX_TARGETS = {"b1": 500, "b2": 1000, "b3": 2000}
MIX_COLUMNS = ("id", "data", "esg_type", "promise_status")


def fallback_promise(span: str, row: dict[str, Any]) -> str:
    templates = [
        "本公司承諾，{span}",
        "我們將持續落實以下承諾：{span}",
        "為推動永續發展，公司將{span}",
        "公司未來將持續推動：{span}",
    ]
    seed = int(str(row.get("id", "0"))) if str(row.get("id", "")).isdigit() else 0
    return normalize_space(random.Random(seed).choice(templates).format(span=span))


def synthesize(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    span = normalize_space(row.get("promise_string", ""))
    method = "deterministic"
    text = fallback_promise(span, row)
    if not args.no_llm:
        try:
            candidate = call_openai_compatible(
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                temperature=args.temperature,
                system_prompt="你是繁體中文 ESG 承諾句生成器。只輸出一個句子。",
                user_prompt=(
                    "請把 PROMISE_SPAN 改寫成一個自足、自然、明確的企業永續承諾句。"
                    "必須保留原本的數字、目標、時程與承諾意義，但不可逐字複製 span，"
                    "不可加入新事實，不要輸出解釋。\n\n"
                    f"PROMISE_SPAN: {span}"
                ),
            )
            if not looks_like_leak(candidate, span):
                text = candidate
                method = "llm"
        except RuntimeError:
            pass
    return {
        "id": f"syn_a2_{row.get('id')}",
        "data": normalize_space(text),
        "esg_type": row.get("esg_type", ""),
        "promise_status": "Yes",
        "synth_source": "a2_promise_string",
        "synth_method": method,
        "src_id": str(row.get("id", "")),
    }


def build_pool(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if str(row.get("promise_status", "")).strip() == "Yes"
        and len(normalize_space(row.get("promise_string", ""))) >= args.min_promise_chars
    ]
    results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(synthesize, row, args): index for index, row in enumerate(candidates)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [results[index] for index in sorted(results)]


def write_mixes(real_rows: list[dict[str, Any]], pool: list[dict[str, Any]], out_dir: Path, seed: int) -> dict[str, Any]:
    real_projected = [project(row, MIX_COLUMNS) for row in real_rows]
    pool_projected = [project(row, MIX_COLUMNS) for row in pool]
    manifest: dict[str, Any] = {}
    for name, target in MIX_TARGETS.items():
        synthetic = sample_with_replacement(
            pool_projected,
            target,
            seed=seed,
            duplicate_suffix="_i",
        )
        rows = real_projected + synthetic
        path = out_dir / f"synth_st1_a2_promise_mix_{name}.json"
        write_json_atomic(path, rows)
        manifest[name] = {
            "path": display_path(path),
            "rows": len(rows),
            "real_rows": len(real_projected),
            "synthetic_rows": len(synthetic),
            "sample_with_replacement": target > len(pool_projected),
            "label_counts": label_counts(rows, "promise_status"),
        }
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw_data/vpesg_4k_train_1000.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/data_synthesis/synth_st1_a2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-promise-chars", type=int, default=12)
    parser.add_argument("--base-url", default="http://192.168.1.79:3134")
    parser.add_argument("--model", default="/workspace/llm_model")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--rebuild-mixes-from-pool", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    real_rows = read_json(args.input)
    out_dir = resolve(args.out_dir)
    pool_path = out_dir / POOL_NAME

    if args.rebuild_mixes_from_pool:
        pool = read_json(pool_path)
    else:
        pool = build_pool(real_rows, args)
        write_json_atomic(pool_path, pool)

    manifest = {
        "input": display_path(args.input),
        "pool_path": display_path(pool_path),
        "pool_rows": len(pool),
        "seed": args.seed,
        "label_counts": label_counts(pool, "promise_status"),
        "method_counts": label_counts(pool, "synth_method"),
        "mixes": write_mixes(real_rows, pool, out_dir, args.seed),
    }
    write_json_atomic(out_dir / MANIFEST_NAME, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
