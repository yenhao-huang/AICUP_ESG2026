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
from pathlib import Path
from typing import Any

from core.service.predict.stage3.vlm import page_context as bpc

ROOT = bpc.ROOT
DEFAULT_DOC_TABLE = bpc.DEFAULT_DOC_TABLE
DEFAULT_PAGE_TABLE = bpc.DEFAULT_PAGE_TABLE
DEFAULT_OFFSETS = bpc.DEFAULT_OFFSETS

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def _page_image_field(page_row: dict[str, Any]) -> str:
    return str(page_row.get("image_path") or page_row.get("image_url") or "").strip()


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

    def _page_no(self, row: dict[str, Any]) -> int | None:
        off = self.offsets.get(str(row.get("id", "")).strip())
        if off is not None:
            try:
                return int(off.get("matched_page_no"))
            except (TypeError, ValueError):
                pass
        try:
            return int(row.get("page_number"))
        except (TypeError, ValueError):
            return None

    def image_path_for_row(self, row: dict[str, Any]) -> Path | None:
        doc_id = self.url2doc.get(row.get("pdf_url")) or self.id2doc.get(str(row.get("id")))
        page_no = self._page_no(row)
        if not doc_id or page_no is None:
            return None
        rel = self.by_page_image.get((doc_id, page_no))
        if not rel:
            return None
        path = Path(rel)
        if not path.is_absolute():
            path = ROOT / path
        return path if path.exists() else None

    def data_url_for_row(self, row: dict[str, Any]) -> str | None:
        """Return a base64 data: URL for the page image, or None if unavailable."""
        path = self.image_path_for_row(row)
        if path is None:
            return None
        mime = _MIME.get(path.suffix.lower(), "image/png")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"
