#!/usr/bin/env python3
"""Materialize training inputs used by the reproduction wrappers.

Synthesis scripts write intermediate outputs under ``results/data_synthesis``.
By default this script writes materialized inputs under ``results/reproduce_inputs``
so generated data is not accidentally staged as repository data. Use
``--output-root data`` only when you intentionally need ``data/synthesis_data``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

RAW_TRAIN = ROOT / "data/raw_data/vpesg_4k_train_1000.json"
RAW_VAL = ROOT / "data/raw_data/vpesg4k_val_1000.json"

STAGE1_SYNTH = ROOT / "results/data_synthesis/stage1/mix_a3_b1.json"
STAGE2_SYNTH = ROOT / "results/data_synthesis/stage2/mix_a2_b3_add_val.json"

STAGE1_COLUMNS = ("id", "data", "promise_status")

OUTPUT_ROOTS = {
    "results": ROOT / "results/reproduce_inputs",
    "data": ROOT / "data/synthesis_data",
}

STAGE_OUTPUT_FILES = {
    "stage1": Path("stage1/a3_b1_add_val.json"),
    "stage2": Path("stage2/mix_a2_b3_add_val.json"),
    "stage3": Path("stage3/vpesg_4k_train_1000_add_val.json"),
    "stage12": Path("stage12/vpesg4k_train_val_mix_2000.json"),
}


def read_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing input: {path.relative_to(ROOT)}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list: {path.relative_to(ROOT)}")
    return data


def project(row: dict[str, Any], columns: tuple[str, ...]) -> dict[str, Any]:
    return {column: row.get(column, "") for column in columns}


def write_json(path: Path, rows: list[dict[str, Any]], *, force: bool, dry_run: bool) -> None:
    display = path.relative_to(ROOT)
    if path.exists() and not force:
        print(f"[skip] exists: {display} ({len(read_json(path))} rows); use --force to overwrite")
        return
    print(f"[write] {display} ({len(rows)} rows)")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")
        tmp = Path(f.name)
    tmp.replace(path)


def build_stage1() -> list[dict[str, Any]]:
    train = [project(row, STAGE1_COLUMNS) for row in read_json(RAW_TRAIN)]
    synth = [project(row, STAGE1_COLUMNS) for row in read_json(STAGE1_SYNTH)]
    val = read_json(RAW_VAL)
    return train + synth + val


def build_stage2() -> list[dict[str, Any]]:
    return read_json(STAGE2_SYNTH)


def build_stage3() -> list[dict[str, Any]]:
    return read_json(RAW_TRAIN) + read_json(RAW_VAL)


def build_stage12() -> list[dict[str, Any]]:
    train = [{**row, "split": "train"} for row in read_json(RAW_TRAIN)]
    val = [{**row, "split": "val"} for row in read_json(RAW_VAL)]
    return train + val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "stage1", "stage2", "stage3", "stage12"),
        default="all",
        help="Which frozen input to materialize.",
    )
    parser.add_argument(
        "--output-root",
        choices=tuple(OUTPUT_ROOTS),
        default="results",
        help="Write under results/reproduce_inputs by default; use data for data/synthesis_data.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing materialized files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned writes without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builders = {
        "stage1": build_stage1,
        "stage2": build_stage2,
        "stage3": build_stage3,
        "stage12": build_stage12,
    }
    selected = builders.keys() if args.stage == "all" else (args.stage,)
    for stage in selected:
        path = OUTPUT_ROOTS[args.output_root] / STAGE_OUTPUT_FILES[stage]
        write_json(path, builders[stage](), force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
