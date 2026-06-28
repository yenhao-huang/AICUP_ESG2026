#!/usr/bin/env python3
"""Generate Stage 2 synthetic evidence-status rows.

The method follows the documented Stage 2 synthesis recipe:
- Yes: promise + evidence -> supported promise sentence.
- No/remove-evidence: supported row -> generic unsupported promise.
- No/rewrite: original no-evidence row -> light rewrite preserving unsupported nature.
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
    write_json_atomic,
)


ST2_COLUMNS = ("id", "data", "esg_type", "promise_status", "evidence_status")


def topic_from(row: dict[str, Any]) -> str:
    text = normalize_space(row.get("promise_string") or row.get("data", ""))
    topics = [
        "減碳",
        "生物多樣性",
        "公司治理",
        "資安",
        "人才培育",
        "供應鏈",
        "職場安全",
        "能源管理",
        "水資源",
        "循環經濟",
        "永續發展",
    ]
    for topic in topics:
        if topic in text:
            return topic
    esg = str(row.get("esg_type", "")).strip().upper()
    return {"E": "環境永續", "S": "社會責任", "G": "公司治理"}.get(esg, "永續發展")


def fallback_yes(row: dict[str, Any]) -> str:
    promise = normalize_space(row.get("promise_string", ""))
    evidence = normalize_space(row.get("evidence_string", ""))
    if promise and evidence:
        return normalize_space(f"為落實{promise}，公司已規劃並執行以下措施：{evidence}")
    return normalize_space(row.get("data", ""))


def fallback_no_remove(row: dict[str, Any]) -> str:
    topic = topic_from(row)
    return f"公司高度關注{topic}議題，期望逐步改善並回應利害關係人期待。"


def fallback_no_rewrite(row: dict[str, Any]) -> str:
    data = normalize_space(row.get("data", ""))
    return normalize_space(f"整體而言，{data}")


def call_or_fallback(
    row: dict[str, Any],
    args: argparse.Namespace,
    *,
    source: str,
) -> tuple[str, str]:
    if source == "yes":
        fallback = fallback_yes(row)
        system = "你是繁體中文 ESG 訓練資料生成器。只輸出一個句子。"
        user = (
            "請根據 PROMISE 與 EVIDENCE 生成一個自足句子，明確包含承諾及支持證據。"
            "保留數字、目標、措施與時間資訊，不要加入新事實，不要輸出解釋。\n\n"
            f"PROMISE: {normalize_space(row.get('promise_string', ''))}\n"
            f"EVIDENCE: {normalize_space(row.get('evidence_string', ''))}"
        )
    elif source == "no_remove":
        fallback = fallback_no_remove(row)
        system = "你是繁體中文 ESG 訓練資料生成器。只輸出一個句子。"
        user = (
            "請根據 PROMISE 生成一個泛化的 ESG 承諾句，但不要包含任何具體支持證據、"
            "已執行措施、數據或可驗證行動。不要輸出解釋。\n\n"
            f"PROMISE: {normalize_space(row.get('promise_string', ''))}"
        )
    else:
        fallback = fallback_no_rewrite(row)
        system = "你是繁體中文 ESG 報告句子改寫器。只輸出一個句子。"
        user = (
            "請輕度改寫下列原本沒有具體支持證據的 ESG 承諾句，保留其未支持性質，"
            "不要新增措施、數據或證據，不要輸出解釋。\n\n"
            f"DATA: {normalize_space(row.get('data', ''))}"
        )

    if args.no_llm:
        return fallback, "fallback"
    try:
        text = call_openai_compatible(
            base_url=args.base_url,
            model=args.model,
            system_prompt=system,
            user_prompt=user,
            timeout=args.timeout,
            temperature=args.temperature,
        )
        return text or fallback, "llm"
    except RuntimeError:
        return fallback, "fallback"


def synthesize_one(row: dict[str, Any], args: argparse.Namespace, *, source: str) -> dict[str, Any]:
    text, method = call_or_fallback(row, args, source=source)
    if source == "yes":
        prefix = "syn_st2_a2yes"
        label = "Yes"
    elif source == "no_remove":
        prefix = "syn_st2_a2no"
        label = "No"
    else:
        prefix = "syn_st2_a2no2"
        label = "No"
    return {
        "id": f"{prefix}_{row.get('id')}",
        "data": normalize_space(text),
        "esg_type": row.get("esg_type", ""),
        "promise_status": "Yes",
        "evidence_status": label,
        "syn_source": source,
        "syn_method": method,
        "syn_parent_id": str(row.get("id", "")),
    }


def sample_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected = list(rows)
    rng.shuffle(selected)
    return sorted(selected[:count], key=lambda row: str(row.get("id", "")))


def run_pool(rows: list[dict[str, Any]], args: argparse.Namespace, *, source: str) -> list[dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    total = len(rows)
    print(f"[{source}] start rows={total} workers={max(1, args.workers)}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(synthesize_one, row, args, source=source): index
            for index, row in enumerate(rows)
        }
        done = 0
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            results[index] = result
            done += 1
            print(
                f"[{source}] {done}/{total} parent={result.get('syn_parent_id', '')} "
                f"id={result.get('id', '')} method={result.get('syn_method', '')} "
                f"label={result.get('evidence_status', '')}",
                flush=True,
            )
    return [results[index] for index in sorted(results)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw_data/vpesg_4k_train_1000.json"))
    parser.add_argument("--val", type=Path, default=Path("data/raw_data/vpesg4k_val_1000.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/data_synthesis/stage2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--yes-count", type=int, default=274)
    parser.add_argument("--no-remove-count", type=int, default=233)
    parser.add_argument("--no-rewrite-count", type=int, default=41)
    parser.add_argument("--limit", type=int, default=None, help="Limit each selected synthetic source for smoke tests.")
    parser.add_argument("--base-url", default="http://192.168.1.78:3132")
    parser.add_argument("--model", default="/workspace/llm_model")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-llm", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_rows = read_json(args.train)
    val_rows = read_json(args.val)
    out_dir = resolve(args.out_dir)

    supported = [
        row
        for row in train_rows
        if str(row.get("promise_status", "")).strip() == "Yes"
        and str(row.get("evidence_status", "")).strip() == "Yes"
        and normalize_space(row.get("evidence_string", ""))
    ]
    unsupported = [
        row
        for row in train_rows
        if str(row.get("promise_status", "")).strip() == "Yes"
        and str(row.get("evidence_status", "")).strip() == "No"
    ]

    yes_sources = sample_rows(supported, args.yes_count, args.seed)
    no_remove_sources = sample_rows(supported, args.no_remove_count, args.seed + 1)
    no_rewrite_sources = sample_rows(unsupported, args.no_rewrite_count, args.seed + 2)
    if args.limit is not None:
        yes_sources = yes_sources[: args.limit]
        no_remove_sources = no_remove_sources[: args.limit]
        no_rewrite_sources = no_rewrite_sources[: args.limit]

    pool_yes = run_pool(yes_sources, args, source="yes")
    pool_no_remove = run_pool(no_remove_sources, args, source="no_remove")
    pool_no_rewrite = run_pool(no_rewrite_sources, args, source="no_rewrite")
    synthetic = pool_yes + pool_no_remove + pool_no_rewrite

    write_json_atomic(out_dir / "pool_st2_yes.json", pool_yes)
    write_json_atomic(out_dir / "pool_st2_no_remove_evidence.json", pool_no_remove)
    write_json_atomic(out_dir / "pool_st2_no_rewrite.json", pool_no_rewrite)
    write_json_atomic(out_dir / "synthetic_st2_balanced.json", synthetic)

    train_projected = [project(row, ST2_COLUMNS) for row in train_rows]
    val_projected = [project(row, ST2_COLUMNS) for row in val_rows]
    synthetic_projected = [project(row, ST2_COLUMNS) for row in synthetic]
    mix = train_projected + synthetic_projected
    mix_add_val = mix + val_projected
    write_json_atomic(out_dir / "mix_a2_b3.json", mix)
    write_json_atomic(out_dir / "mix_a2_b3_add_val.json", mix_add_val)

    manifest = {
        "train": display_path(args.train),
        "val": display_path(args.val),
        "out_dir": display_path(out_dir),
        "seed": args.seed,
        "base_url": args.base_url,
        "model": args.model,
        "limit": args.limit,
        "sources": {
            "yes": {"rows": len(pool_yes), "method_counts": label_counts(pool_yes, "syn_method")},
            "no_remove": {"rows": len(pool_no_remove), "method_counts": label_counts(pool_no_remove, "syn_method")},
            "no_rewrite": {"rows": len(pool_no_rewrite), "method_counts": label_counts(pool_no_rewrite, "syn_method")},
        },
        "outputs": {
            "synthetic": {
                "path": display_path(out_dir / "synthetic_st2_balanced.json"),
                "rows": len(synthetic),
                "evidence_status": label_counts(synthetic, "evidence_status"),
            },
            "mix": {
                "path": display_path(out_dir / "mix_a2_b3.json"),
                "rows": len(mix),
                "evidence_status": label_counts(mix, "evidence_status"),
            },
            "mix_add_val": {
                "path": display_path(out_dir / "mix_a2_b3_add_val.json"),
                "rows": len(mix_add_val),
                "evidence_status": label_counts(mix_add_val, "evidence_status"),
            },
        },
    }
    write_json_atomic(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
