#!/usr/bin/env python3
"""Add a `doc_id` column to each row of offsets.jsonl.

Resolves the report doc id the same way the predictors do: by `pdf_url` via
raw_doc_table.jsonl (url -> doc_id), falling back to the row `id` (id -> doc_id).
Idempotent (re-running overwrites doc_id). In-place writes back up to
offsets.jsonl.bak_docid first.

Usage:
    python add_doc_id_to_offsets.py                 # in-place + backup
    python add_doc_id_to_offsets.py --dry-run        # stats only
    python add_doc_id_to_offsets.py --output /tmp/o.jsonl
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # 0617/offset_add_docid/
# 0617/offset_add_docid -> 0617 -> .../integrated_stage_predictions -> ... ; the
# offsets live under the sibling test_add_context experiment dir.
DEFAULT_OFFSETS = HERE.parent / "test_add_context" / "data" / "offsets.jsonl"

REPO_ROOT = next((p for p in HERE.parents if (p / "core" / "human" / "predict").is_dir()), HERE)
sys.path.insert(0, str(REPO_ROOT))
import core.human.predict.stage4.build_page_context as bpc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offsets", type=Path, default=DEFAULT_OFFSETS)
    ap.add_argument("--output", type=Path, help="Write here instead of in-place (no backup).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    doc_rows = bpc.read_jsonl(bpc.DEFAULT_DOC_TABLE)
    url2doc = bpc.build_url2doc(doc_rows)
    id2doc = bpc.build_id2doc(doc_rows)

    rows = bpc.read_jsonl(args.offsets)
    via_url = via_id = missing = 0
    for r in rows:
        doc_id = url2doc.get(r.get("pdf_url"))
        if doc_id:
            via_url += 1
        else:
            doc_id = id2doc.get(str(r.get("id")))
            if doc_id:
                via_id += 1
            else:
                missing += 1
        r["doc_id"] = doc_id or ""

    def reorder(r: dict) -> dict:
        """Place doc_id right after id, keep the rest of the order."""
        out = {}
        for k, v in r.items():
            if k == "doc_id":
                continue
            out[k] = v
            if k == "id":
                out["doc_id"] = r.get("doc_id", "")
        if "doc_id" not in out:  # no id field -> append at end
            out["doc_id"] = r.get("doc_id", "")
        return out

    rows = [reorder(r) for r in rows]

    total = len(rows)
    print(f"offsets rows: {total}")
    print(f"  doc_id via pdf_url : {via_url}")
    print(f"  doc_id via id      : {via_id}")
    print(f"  unresolved         : {missing}")

    if args.dry_run:
        print("[dry-run] nothing written")
        return

    if args.output:
        out = args.output
    else:
        out = args.offsets
        bak = args.offsets.with_suffix(args.offsets.suffix + ".bak_docid")
        if not bak.exists():
            shutil.copy2(args.offsets, bak)
            print(f"backup written: {bak}")
        else:
            print(f"backup already exists, not overwriting: {bak}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}  rows={total}")


if __name__ == "__main__":
    main()
