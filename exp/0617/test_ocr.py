"""
Scan every page's OCR text in raw_page_table.jsonl and flag garbage characters.

"Garbage" = characters outside the set a Traditional-Chinese ESG report can
legitimately contain (CJK, Bopomofo, ASCII, fullwidth forms, common punctuation
and symbols). Visual OCR tends to emit look-alike code points from unrelated
scripts (Ethiopic / Myanmar / Oriya / Tibetan / Runic ...) or PDF embedded-font
private-use glyphs when it fails on figure / chart / vertical art text.

Outputs (written next to this script, in today's loop dir):
  - ocr_garbage_per_page.jsonl : one row per page that contains >=1 garbage char
  - ocr_garbage_summary.json   : aggregate stats + worst reports / pages

Usage:
    python exp/integrated_stage_predictions/0617/test_ocr.py
    python exp/integrated_stage_predictions/0617/test_ocr.py \
        --page-table data/generated/raw_page_table.jsonl \
        --doc-table  data/generated/raw_doc_table.jsonl \
        --min-ratio 0.02
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_PAGE_TABLE = REPO_ROOT / "data/generated/raw_page_table.jsonl"
DEFAULT_DOC_TABLE = REPO_ROOT / "data/generated/raw_doc_table.jsonl"


def is_allowed(ch: str) -> bool:
    """True if ch is a character a zh-Hant ESG report can legitimately contain."""
    if ch in "\n\r\t\f\v ":
        return True
    o = ord(ch)
    return (
        0x20 <= o <= 0x7E            # ASCII printable
        or 0x00A0 <= o <= 0x00FF     # Latin-1 (°, ×, ², µ, ...)
        or 0x2000 <= o <= 0x206F     # general punctuation (— " " … etc)
        or 0x2070 <= o <= 0x209F     # super/subscripts
        or 0x20A0 <= o <= 0x20CF     # currency symbols
        or 0x2100 <= o <= 0x214F     # letterlike (™, ℃, №)
        or 0x2150 <= o <= 0x218F     # number forms (Roman numerals)
        or 0x2190 <= o <= 0x21FF     # arrows
        or 0x2200 <= o <= 0x22FF     # math operators
        or 0x2460 <= o <= 0x24FF     # enclosed alphanumerics ①②
        or 0x2500 <= o <= 0x257F     # box drawing
        or 0x25A0 <= o <= 0x25FF     # geometric shapes ■●▲
        or 0x2600 <= o <= 0x26FF     # misc symbols
        or 0x3000 <= o <= 0x303F     # CJK symbols & punctuation
        or 0x3100 <= o <= 0x312F     # Bopomofo 注音
        or 0x3200 <= o <= 0x33FF     # enclosed CJK / CJK compat
        or 0x3400 <= o <= 0x4DBF     # CJK ext A
        or 0x4E00 <= o <= 0x9FFF     # CJK unified
        or 0xF900 <= o <= 0xFAFF     # CJK compat ideographs
        or 0xFE30 <= o <= 0xFE4F     # CJK compat forms (vertical punctuation)
        or 0xFF00 <= o <= 0xFFEF     # fullwidth & halfwidth forms
        or 0x20000 <= o <= 0x2FFFF   # CJK ext B+ (rare hanzi)
    )


def char_block(ch: str) -> str:
    """Coarse label for a garbage char: PUA, or the script name prefix."""
    o = ord(ch)
    if 0xE000 <= o <= 0xF8FF or 0xF0000 <= o <= 0xFFFFD or 0x100000 <= o <= 0x10FFFD:
        return "PRIVATE-USE"
    try:
        return unicodedata.name(ch).split(" ")[0]
    except ValueError:
        return "<unnamed>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-table", default=str(DEFAULT_PAGE_TABLE))
    parser.add_argument("--doc-table", default=str(DEFAULT_DOC_TABLE))
    parser.add_argument("--min-ratio", type=float, default=0.0,
                        help="only list pages whose garbage ratio >= this in per-page output")
    parser.add_argument("--out-dir", default=str(HERE))
    args = parser.parse_args()

    page_table = Path(args.page_table)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # doc_id -> company for nicer summaries.
    doc_company: dict[str, str] = {}
    doc_table = Path(args.doc_table)
    if doc_table.exists():
        for line in doc_table.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                doc_company[str(d.get("doc_id", ""))] = d.get("company", "")

    per_page_path = out_dir / "ocr_garbage_per_page.jsonl"
    summary_path = out_dir / "ocr_garbage_summary.json"

    total_pages = 0
    pages_with_garbage = 0
    total_chars = 0
    total_garbage = 0
    block_counter: Counter[str] = Counter()
    char_counter: Counter[str] = Counter()
    per_doc_pages: Counter[str] = Counter()        # garbage pages per doc
    per_doc_total: Counter[str] = Counter()        # total pages per doc
    worst_pages: list[tuple[float, int, str, str, str]] = []

    with per_page_path.open("w", encoding="utf-8") as out:
        for line in page_table.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "") or ""
            doc_id = str(row.get("doc_id", ""))
            total_pages += 1
            per_doc_total[doc_id] += 1

            n = len(text)
            total_chars += n
            bad_chars = [c for c in text if not is_allowed(c)]
            nb = len(bad_chars)
            if nb == 0:
                continue

            total_garbage += nb
            pages_with_garbage += 1
            per_doc_pages[doc_id] += 1
            ratio = nb / n if n else 0.0

            local_blocks = Counter(char_block(c) for c in bad_chars)
            local_chars = Counter(bad_chars)
            block_counter.update(local_blocks)
            char_counter.update(local_chars)
            worst_pages.append((ratio, nb, doc_id, str(row.get("page_no", "")),
                                row.get("image_path", "")))

            if ratio >= args.min_ratio:
                out.write(json.dumps({
                    "doc_id": doc_id,
                    "company": doc_company.get(doc_id, ""),
                    "page_no": row.get("page_no", ""),
                    "image_path": row.get("image_path", ""),
                    "n_chars": n,
                    "n_garbage": nb,
                    "garbage_ratio": round(ratio, 4),
                    "garbage_blocks": dict(local_blocks.most_common()),
                    "garbage_chars": "".join(sorted(local_chars,
                                                    key=lambda c: -local_chars[c])),
                }, ensure_ascii=False) + "\n")

    worst_pages.sort(reverse=True)
    summary = {
        "page_table": str(page_table),
        "total_pages": total_pages,
        "pages_with_garbage": pages_with_garbage,
        "pages_with_garbage_pct": round(100 * pages_with_garbage / total_pages, 2) if total_pages else 0,
        "total_chars": total_chars,
        "total_garbage_chars": total_garbage,
        "garbage_char_pct": round(100 * total_garbage / total_chars, 4) if total_chars else 0,
        "top_garbage_blocks": dict(block_counter.most_common(25)),
        "top_garbage_chars": [
            {"char": c, "U+": f"{ord(c):04X}", "block": char_block(c), "count": n}
            for c, n in char_counter.most_common(30)
        ],
        "worst_reports_by_garbage_page_share": [
            {
                "doc_id": did,
                "company": doc_company.get(did, ""),
                "garbage_pages": per_doc_pages[did],
                "total_pages": per_doc_total[did],
                "share_pct": round(100 * per_doc_pages[did] / per_doc_total[did], 1),
            }
            for did, _ in per_doc_pages.most_common(15)
        ],
        "worst_pages_by_ratio": [
            {"garbage_ratio": round(r, 4), "n_garbage": nb, "doc_id": did,
             "page_no": pno, "image_path": img}
            for r, nb, did, pno, img in worst_pages[:20]
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"pages: {total_pages}  | with garbage: {pages_with_garbage} "
          f"({summary['pages_with_garbage_pct']}%)")
    print(f"chars: {total_chars}  | garbage: {total_garbage} "
          f"({summary['garbage_char_pct']}%)")
    print("\ntop garbage blocks (script of look-alike code points):")
    for blk, c in block_counter.most_common(12):
        print(f"  {c:7d}  {blk}")
    print("\nworst reports (garbage-page share):")
    for r in summary["worst_reports_by_garbage_page_share"][:10]:
        print(f"  {r['share_pct']:5.1f}%  {r['garbage_pages']:3d}/{r['total_pages']:<3d}  "
              f"{r['company'] or r['doc_id']}")
    print(f"\nper-page detail -> {per_page_path}")
    print(f"summary          -> {summary_path}")


if __name__ == "__main__":
    main()
