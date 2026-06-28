#!/usr/bin/env python3
"""Generate loop001 ST1 A1/A3 synthetic pools.

A1: paraphrase real ``data`` and carry ``promise_status`` as the synthetic
target. A3: combine ``data`` with ``promise_string`` into a positive commitment
sentence. Both are offline-only synthesis sources from the agent-loop notes.
"""

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
    normalize_space,
    project,
    read_json,
    resolve,
    sample_with_replacement,
    write_json_atomic,
)


ST1_COLUMNS = ("id", "data", "promise_status")
MIX_TARGETS = {"b1": 500, "b2": 1000, "b3": 2000}


def fallback_a1(data: str, row: dict[str, Any]) -> str:
    templates = [
        "整體而言，{data}",
        "公司於永續發展脈絡中說明：{data}",
        "{data}",
    ]
    seed = int(str(row.get("id", "0")).split("_")[-1]) if str(row.get("id", "")).isdigit() else 0
    return normalize_space(random.Random(seed).choice(templates).format(data=data))


def fallback_a3(data: str, promise: str, row: dict[str, Any]) -> str:
    templates = [
        "本公司承諾，{promise}相關作法將結合既有營運脈絡持續推動。",
        "為落實永續發展，公司將持續推動以下承諾：{promise}",
        "公司依據既有永續策略，明確承諾{promise}",
    ]
    seed = int(str(row.get("id", "0")).split("_")[-1]) if str(row.get("id", "")).isdigit() else 0
    text = random.Random(seed).choice(templates).format(promise=promise)
    if len(normalize_space(text)) < 20:
        text = f"{text} {data}"
    return normalize_space(text)


def select_a1_rows(rows: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row.get("promise_status", "")), []).append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    total = len(rows)
    labels = sorted(by_label)
    remaining = cap
    for index, label in enumerate(labels):
        bucket = list(by_label[label])
        rng.shuffle(bucket)
        if index == len(labels) - 1:
            take = remaining
        else:
            take = round(cap * len(bucket) / total)
            remaining -= take
        selected.extend(bucket[:take])
    selected.sort(key=lambda row: str(row.get("id", "")))
    return selected[:cap]


def select_a3_rows(rows: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if str(row.get("promise_status", "")).strip() == "Yes"
        and len(normalize_space(row.get("promise_string", ""))) >= 8
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return sorted(candidates[:cap], key=lambda row: str(row.get("id", "")))


def synthesize_a1(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data = normalize_space(row.get("data", ""))
    method = "fallback"
    text = fallback_a1(data, row)
    if not args.no_llm:
        try:
            text = call_openai_compatible(
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                temperature=args.temperature,
                system_prompt="你是繁體中文 ESG 報告句子的改寫器。只輸出改寫後句子。",
                user_prompt=(
                    "請改寫下列 ESG 報告句子，保留原意以及是否包含企業承諾的語意，"
                    "不要加入新事實，不要輸出解釋。\n\n"
                    f"句子：{data}"
                ),
            )
            method = "llm"
        except RuntimeError:
            pass
    return {
        "id": f"syn_a1_{row.get('id')}",
        "data": normalize_space(text),
        "promise_status": str(row.get("promise_status", "")),
        "syn_source": "a1_data_only",
        "syn_method": method,
        "syn_parent_id": str(row.get("id", "")),
    }


def synthesize_a3(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data = normalize_space(row.get("data", ""))
    promise = normalize_space(row.get("promise_string", ""))
    method = "fallback"
    text = fallback_a3(data, promise, row)
    if not args.no_llm:
        try:
            text = call_openai_compatible(
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                temperature=args.temperature,
                system_prompt="你是繁體中文 ESG 承諾句生成器。只輸出一個自足句子。",
                user_prompt=(
                    "請根據 DATA 的上下文與 PROMISE_SPAN，生成一個流暢、完整、"
                    "明確表達同一企業承諾的繁體中文句子。保留數字、目標與時間資訊，"
                    "不要加入新事實，不要輸出解釋。\n\n"
                    f"DATA: {data}\n\nPROMISE_SPAN: {promise}"
                ),
            )
            method = "llm"
        except RuntimeError:
            pass
    return {
        "id": f"syn_a3_{row.get('id')}",
        "data": normalize_space(text),
        "promise_status": "Yes",
        "syn_source": "a3_data_plus_promise",
        "syn_method": method,
        "syn_parent_id": str(row.get("id", "")),
    }


def run_pool(name: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    worker = synthesize_a1 if name == "a1" else synthesize_a3
    results: dict[int, dict[str, Any]] = {}
    total = len(rows)
    print(f"[{name}] start rows={total} workers={max(1, args.workers)}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(worker, row, args): index for index, row in enumerate(rows)}
        done = 0
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            results[index] = result
            done += 1
            print(
                f"[{name}] {done}/{total} parent={result.get('syn_parent_id', '')} "
                f"id={result.get('id', '')} method={result.get('syn_method', '')}",
                flush=True,
            )
    return [results[index] for index in sorted(results)]


def write_mixes(source: str, pool: list[dict[str, Any]], out_dir: Path, seed: int) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for name, target in MIX_TARGETS.items():
        rows = sample_with_replacement(
            [project(row, ST1_COLUMNS) for row in pool],
            target,
            seed=seed,
            duplicate_suffix="_k",
        )
        path = out_dir / f"mix_{source}_{name}.json"
        write_json_atomic(path, rows)
        manifest[name] = {
            "path": display_path(path),
            "rows": len(rows),
            "label_counts": label_counts(rows, "promise_status"),
            "sample_with_replacement": target > len(pool),
        }
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw_data/vpesg_4k_train_1000.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/data_synthesis/stage1"))
    parser.add_argument("--pool-cap", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None, help="Limit selected A1/A3 source rows for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-url", default="http://192.168.1.78:3132")
    parser.add_argument("--model", default="/workspace/llm_model")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-llm", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_json(args.input)
    out_dir = resolve(args.out_dir)

    a1_rows = select_a1_rows(rows, args.pool_cap, args.seed)
    a3_rows = select_a3_rows(rows, args.pool_cap, args.seed)
    if args.limit is not None:
        a1_rows = a1_rows[: args.limit]
        a3_rows = a3_rows[: args.limit]
    pool_a1 = run_pool("a1", a1_rows, args)
    pool_a3 = run_pool("a3", a3_rows, args)

    write_json_atomic(out_dir / "pool_a1_data_only.json", pool_a1)
    write_json_atomic(out_dir / "pool_a3_data_plus_promise.json", pool_a3)
    manifest = {
        "input": display_path(args.input),
        "seed": args.seed,
        "pool_cap": args.pool_cap,
        "sources": {
            "a1": {
                "pool_path": display_path(out_dir / "pool_a1_data_only.json"),
                "pool_rows": len(pool_a1),
                "label_counts": label_counts(pool_a1, "promise_status"),
                "method_counts": label_counts(pool_a1, "syn_method"),
                "mixes": write_mixes("a1", pool_a1, out_dir, args.seed),
            },
            "a3": {
                "pool_path": display_path(out_dir / "pool_a3_data_plus_promise.json"),
                "pool_rows": len(pool_a3),
                "label_counts": label_counts(pool_a3, "promise_status"),
                "method_counts": label_counts(pool_a3, "syn_method"),
                "mixes": write_mixes("a3", pool_a3, out_dir, args.seed),
            },
        },
    }
    write_json_atomic(out_dir / "sampling_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
