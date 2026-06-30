#!/usr/bin/env python3
"""Add a resolved `image_path` column to each row of offsets.jsonl.

For every offsets row (one per annotation id), pick the page image the same way
core/build_prompt/add_image.py does at runtime:

    1. offsets `matched_page_no`            (exact OCR-located page)
    2. offsets `weakly_matched_page_no[i]`  (weaker OCR matches, in order)
    -- the raw annotation `page_number` is intentionally NOT used as a fallback.

The chosen page is joined to raw_page_table.jsonl via the doc id (url -> doc),
and the first candidate whose page image file actually exists on disk wins.

Adds three columns (existing ones are overwritten so the script is idempotent):
    image_path        repo-relative PNG path (or "" if nothing resolved)
    image_page_no     the page number the image came from (or null)
    image_source      "matched_page_no" | "weakly_matched_page_no" | ""

Writes in place after backing up to offsets.jsonl.bak (unless --output given).

Usage:
    python build_offsets_image_path.py            # in-place + .bak
    python build_offsets_image_path.py --output /tmp/offsets_with_image.jsonl
    python build_offsets_image_path.py --dry-run  # just print stats
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent              # stage4/
EXP_ROOT = HERE.parent                              # test_add_context/
DEFAULT_OFFSETS = EXP_ROOT / "data" / "offsets.jsonl"

REPO_ROOT = next((p for p in HERE.parents if (p / "core" / "human" / "predict").is_dir()), HERE)
sys.path.insert(0, str(REPO_ROOT))
import core.human.predict.stage4.build_page_context as bpc  # noqa: E402


def _img_field(page_row: dict[str, Any]) -> str:
    return str(page_row.get("image_path") or page_row.get("image_url") or "").strip()


def build_indexes() -> tuple[dict, dict, dict]:
    doc_rows = bpc.read_jsonl(bpc.DEFAULT_DOC_TABLE)
    url2doc = bpc.build_url2doc(doc_rows)
    id2doc = bpc.build_id2doc(doc_rows)
    by_page_image: dict[tuple[str, int], str] = {}
    for r in bpc.read_jsonl(bpc.DEFAULT_PAGE_TABLE):
        doc_id = r.get("doc_id")
        img = _img_field(r)
        try:
            page_no = int(r.get("page_no"))
        except (TypeError, ValueError):
            continue
        if doc_id and img:
            by_page_image[(doc_id, page_no)] = img
    return url2doc, id2doc, by_page_image


def candidates(row: dict[str, Any]) -> list[int]:
    cands: list[int] = []
    try:
        cands.append(int(row.get("matched_page_no")))
    except (TypeError, ValueError):
        pass
    weak = row.get("weakly_matched_page_no")
    if isinstance(weak, (list, tuple)):
        for w in weak:
            try:
                cands.append(int(w))
            except (TypeError, ValueError):
                continue
    elif weak is not None:
        try:
            cands.append(int(weak))
        except (TypeError, ValueError):
            pass
    seen: set[int] = set()
    return [c for c in cands if not (c in seen or seen.add(c))]


def resolve(row: dict[str, Any], url2doc, id2doc, by_page_image) -> tuple[str, int | None, str]:
    doc_id = url2doc.get(row.get("pdf_url")) or id2doc.get(str(row.get("id")))
    if not doc_id:
        return "", None, "no_doc_id"
    try:
        matched = int(row.get("matched_page_no"))
    except (TypeError, ValueError):
        matched = None
    for c in candidates(row):
        rel = by_page_image.get((doc_id, c))
        if not rel:
            continue
        path = Path(rel)
        abs_path = path if path.is_absolute() else REPO_ROOT / path
        if abs_path.exists():
            source = "matched_page_no" if (matched is not None and c == matched) else "weakly_matched_page_no"
            return rel, c, source
    return "", None, "no_image_for_candidates"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offsets", type=Path, default=DEFAULT_OFFSETS)
    ap.add_argument("--output", type=Path, help="Write here instead of in-place (no .bak made).")
    ap.add_argument("--dry-run", action="store_true", help="Print stats only, write nothing.")
    args = ap.parse_args()

    url2doc, id2doc, by_page_image = build_indexes()
    rows = bpc.read_jsonl(args.offsets)

    stats = {"matched_page_no": 0, "weakly_matched_page_no": 0, "no_doc_id": 0,
             "no_image_for_candidates": 0}
    fallbacks = []
    for r in rows:
        rel, page_no, source = resolve(r, url2doc, id2doc, by_page_image)
        r["image_path"] = rel
        r["image_page_no"] = page_no
        r["image_source"] = source if source in ("matched_page_no", "weakly_matched_page_no") else ""
        stats[source] = stats.get(source, 0) + 1
        if source == "weakly_matched_page_no" and len(fallbacks) < 8:
            fallbacks.append((r.get("id"), r.get("matched_page_no"), r.get("weakly_matched_page_no"), page_no))

    total = len(rows)
    print(f"offsets rows: {total}")
    print(f"  resolved via matched_page_no : {stats['matched_page_no']}")
    print(f"  resolved via weakly fallback : {stats['weakly_matched_page_no']}")
    print(f"  unresolved (no doc id)       : {stats['no_doc_id']}")
    print(f"  unresolved (no image file)   : {stats['no_image_for_candidates']}")
    if fallbacks:
        print("  weakly-fallback examples (id, matched, weak, used):")
        for f in fallbacks:
            print("   ", f)

    if args.dry_run:
        print("[dry-run] nothing written")
        return

    if args.output:
        out = args.output
    else:
        out = args.offsets
        bak = args.offsets.with_suffix(args.offsets.suffix + ".bak")
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
