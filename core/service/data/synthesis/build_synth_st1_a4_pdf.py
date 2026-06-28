#!/usr/bin/env python3
"""Build loop001 ST1 A4 PDF/data-derived hard-negative rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from core.service.data.synthesis.common import (
    display_path,
    label_counts,
    normalize_space,
    read_json,
    resolve,
    sample_with_replacement,
    write_json_atomic,
)


POOL_COLUMNS = ("id", "data", "promise_status")
TARGETS_DEFAULT = {"b1": 500, "b2": 1000, "b3": 2000}
TARGETS_LOOP_RECORDED = {"b1": 400, "b2": 800, "b3": 1600}


def build_pool(source_rows: list[dict[str, Any]], real_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    real_texts = {normalize_space(row.get("data", "")) for row in real_rows}
    seen: set[str] = set()
    pool: list[dict[str, Any]] = []
    for row in source_rows:
        data = normalize_space(row.get("data", ""))
        if not data or data in seen or data in real_texts:
            continue
        label = str(row.get("promise_status", "")).strip()
        if label != "No":
            continue
        seen.add(data)
        pool.append(
            {
                "id": f"syn_a4_{row.get('id', len(pool) + 1)}",
                "data": data,
                "promise_status": "No",
            }
        )
    return pool


def write_mixes(pool: list[dict[str, Any]], out_dir: Path, seed: int, targets: dict[str, int]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for name, target in targets.items():
        rows = sample_with_replacement(pool, target, seed=seed, duplicate_suffix="_k")
        path = out_dir / f"synth_st1_a4_pdf_{name}.json"
        write_json_atomic(path, rows)
        manifest[name] = {
            "path": display_path(path),
            "rows": len(rows),
            "sample_with_replacement": target > len(pool),
            "unique_ids": len({row["id"] for row in rows}),
            "label_counts": label_counts(rows, "promise_status"),
        }
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-pool",
        type=Path,
        default=Path("data/generated/generation_from_data/merged_strict_st1_train.json"),
        help="Pre-materialized PDF/data-derived pool from agent-loop input.",
    )
    parser.add_argument(
        "--real-train",
        type=Path,
        default=Path("data/raw_data/vpesg_4k_train_1000.json"),
        help="Real train rows used for overlap removal.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/data_synthesis/synth_st1_a4"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--loop-recorded-800-base",
        action="store_true",
        help="Use the original loop-recorded A4 400/800/1600 target counts instead of normalized 500/1000/2000.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_rows = read_json(args.source_pool)
    real_rows = read_json(args.real_train)
    out_dir = resolve(args.out_dir)
    targets = TARGETS_LOOP_RECORDED if args.loop_recorded_800_base else TARGETS_DEFAULT

    pool = build_pool(source_rows, real_rows)
    pool_path = out_dir / "synth_st1_a4_pdf_pool.json"
    write_json_atomic(pool_path, pool)
    manifest = {
        "source_pool": display_path(args.source_pool),
        "real_train": display_path(args.real_train),
        "pool_path": display_path(pool_path),
        "pool_rows": len(pool),
        "seed": args.seed,
        "targets": targets,
        "label_counts": label_counts(pool, "promise_status"),
        "mixes": write_mixes(pool, out_dir, args.seed, targets),
    }
    write_json_atomic(out_dir / "synth_st1_a4_pdf_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
