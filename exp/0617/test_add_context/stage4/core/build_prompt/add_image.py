#!/usr/bin/env python3
"""Page-image enrichment for the modular Stage 3 VLM predictor.

Resolves a row to the rendered page image of its SAME report page, so a
vision-language model can read the page directly. The image is taken from
raw_page_table.jsonl. The plan calls the field `image_url`; the live table
exposes the page render as `image_path` (a repo-relative PNG path), so both
keys are accepted, `image_path` first.

The page is chosen from offsets.jsonl (matched_page_no) when available, else the
row's annotated page_number, joined to a doc_id via raw_doc_table.jsonl.

Data-use note: a page image embeds raw report content beyond `data`; it is OFF
by default and only used for the explicit add_image experiment.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "core" / "human" / "predict").is_dir() and (p / "data" / "generated").is_dir():
            return p
    return start.parents[-1]


_ROOT = _find_repo_root(Path(__file__).resolve())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import core.human.predict.stage4.build_page_context as bpc  # noqa: E402

DEFAULT_DOC_TABLE = bpc.DEFAULT_DOC_TABLE
DEFAULT_PAGE_TABLE = bpc.DEFAULT_PAGE_TABLE
DEFAULT_OFFSETS = Path(__file__).resolve().parents[3] / "data" / "offsets.jsonl"

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def _page_image_field(page_row: dict[str, Any]) -> str:
    # `image_path` is now the doc_id-based logical name (<doc_id>_p<n>.png) and
    # does NOT exist on disk; the real render-hash PNG is in `image_file`. Prefer
    # it, falling back to image_path/image_url for older tables.
    return str(page_row.get("image_file") or page_row.get("image_path")
               or page_row.get("image_url") or "").strip()


class ImageBuilder:
    def __init__(
        self,
        *,
        doc_table: Path = DEFAULT_DOC_TABLE,
        page_table: Path = DEFAULT_PAGE_TABLE,
        offsets_path: Path = DEFAULT_OFFSETS,
    ) -> None:
        doc_rows = bpc.read_jsonl(doc_table)
        self.url2doc = bpc.build_url2doc(doc_rows)
        self.id2doc = bpc.build_id2doc(doc_rows)
        self.by_page_image: dict[tuple[str, int], str] = {}
        for row in bpc.read_jsonl(page_table):
            doc_id = row.get("doc_id")
            img = _page_image_field(row)
            try:
                page_no = int(row.get("page_no"))
            except (TypeError, ValueError):
                continue
            if doc_id and img:
                self.by_page_image[(doc_id, page_no)] = img
        self.offsets: dict[str, dict[str, Any]] = {}
        if Path(offsets_path).exists():
            for r in bpc.read_jsonl(offsets_path):
                rid = str(r.get("id", "")).strip()
                if rid:
                    self.offsets[rid] = r

    def _page_candidates(self, row: dict[str, Any]) -> list[int]:
        """Ordered candidate page numbers for the image, all OCR-aligned.

        Priority: offsets `matched_page_no` (exact OCR-located page), then the
        offsets `weakly_matched_page_no` candidates (weaker OCR matches), in
        their listed order. The raw annotation `page_number` is intentionally
        NOT used as a fallback: ~46% of rows are off by one or two against the
        rendered page index, so it would attach the wrong page image.
        """
        off = self.offsets.get(str(row.get("id", "")).strip())
        if off is None:
            return []
        cands: list[int] = []
        try:
            cands.append(int(off.get("matched_page_no")))
        except (TypeError, ValueError):
            pass
        weak = off.get("weakly_matched_page_no")
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
        # de-dup, keep order
        seen: set[int] = set()
        return [c for c in cands if not (c in seen or seen.add(c))]

    def image_path_for_row(self, row: dict[str, Any]) -> Path | None:
        doc_id = self.url2doc.get(row.get("pdf_url")) or self.id2doc.get(str(row.get("id")))
        if not doc_id:
            return None
        for page_no in self._page_candidates(row):
            rel = self.by_page_image.get((doc_id, page_no))
            if not rel:
                continue
            path = Path(rel)
            if not path.is_absolute():
                path = _ROOT / path
            if path.exists():
                return path
        return None

    def data_url_for_row(self, row: dict[str, Any]) -> str | None:
        """Return a base64 data: URL for the page image, or None if unavailable."""
        path = self.image_path_for_row(row)
        if path is None:
            return None
        mime = _MIME.get(path.suffix.lower(), "image/png")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"
