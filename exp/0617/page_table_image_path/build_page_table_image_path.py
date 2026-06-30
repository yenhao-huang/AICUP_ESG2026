#!/usr/bin/env python3
"""Rewrite raw_page_table.jsonl image fields to doc_id-based naming.

For every page row:
    image_path  ->  "<doc_id>_p<page_no>.png"      (doc_id-based logical name)
    image_file  ->  the original render-hash relative path that actually exists
                    on disk (e.g. data/raw_reports/images/<hash>_p<page_no>.png)

The real PNG files keep their render-hash names, so the on-disk correspondence
is preserved under `image_file`; `image_path` becomes the canonical doc_id name.
`image_file` is inserted right after `image_path`. Idempotent: re-running detects
an already-render-hash `image_file` and rebuilds from it.

Writes in place after backing up to raw_page_table.jsonl.bak_imgpath.

Usage:
    python build_page_table_image_path.py            # in-place + backup
    python build_page_table_image_path.py --dry-run
    python build_page_table_image_path.py --output /tmp/pt.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in HERE.parents if (p / "core" / "human" / "predict").is_dir()), HERE)
sys.path.insert(0, str(REPO_ROOT))
import core.human.predict.stage4.build_page_context as bpc  # noqa: E402

_HASH_RE = re.compile(r"([0-9a-f]+)_p\d+\.png$")


def reorder_with_image_file(row: dict, image_path_val: str, image_file_val: str) -> dict:
    """Set image_path, and put image_file right after it (keep other order)."""
    out = {}
    seen_ip = False
    for k, v in row.items():
        if k == "image_file":
            continue  # re-inserted in the right spot
        if k == "image_path":
            out["image_path"] = image_path_val
            out["image_file"] = image_file_val
            seen_ip = True
        else:
            out[k] = v
    if not seen_ip:
        out["image_path"] = image_path_val
        out["image_file"] = image_file_val
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page-table", type=Path, default=Path(bpc.DEFAULT_PAGE_TABLE))
    ap.add_argument("--output", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = bpc.read_jsonl(args.page_table)
    ok = no_hash = no_doc = 0
    out_rows = []
    examples = []
    for r in rows:
        doc_id = str(r.get("doc_id") or "").strip()
        page_no = r.get("page_no")
        # Source of the real file: existing image_file (idempotent re-run) else image_path.
        real = str(r.get("image_file") or r.get("image_path") or r.get("image_url") or "").strip()
        if not doc_id:
            no_doc += 1
            out_rows.append(r)
            continue
        if not _HASH_RE.search(real):
            no_hash += 1
        new_image_path = f"{doc_id}_p{page_no}.png"
        out_rows.append(reorder_with_image_file(r, new_image_path, real))
        ok += 1
        if len(examples) < 4:
            examples.append((new_image_path, real))

    print(f"page rows: {len(rows)}")
    print(f"  rewritten            : {ok}")
    print(f"  skipped (no doc_id)  : {no_doc}")
    print(f"  image_file w/o render-hash pattern : {no_hash}")
    print("  examples (image_path -> image_file):")
    for ip, rf in examples:
        print(f"    {ip}  ->  {rf}")

    if args.dry_run:
        print("[dry-run] nothing written")
        return

    if args.output:
        out = args.output
    else:
        out = args.page_table
        bak = args.page_table.with_suffix(args.page_table.suffix + ".bak_imgpath")
        if not bak.exists():
            shutil.copy2(args.page_table, bak)
            print(f"backup written: {bak}")
        else:
            print(f"backup already exists, not overwriting: {bak}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}  rows={len(out_rows)}")


if __name__ == "__main__":
    main()
