#!/usr/bin/env python3
"""Context enrichment for the modular Stage 3 VLM predictor.

Builds the augmented `data` string with optional blocks:

1. same-page-content: the SAME report page's OCR text, located via the precomputed
   `offsets.jsonl` (id -> hit_kind, matched_page_no) joined against
   raw_doc_table.jsonl (url->doc_id) and raw_page_table.jsonl (doc_id,page_no->text).
   Modes select which offset hit_kinds are injected:
     - "all" (default): inject for any offset hit (and live-locate as a fallback
       when the id has no offsets row).
     - "hit_exact_window_norm_window": inject only when the offsets hit_kind is
       hit_exact_window or hit_norm_window; otherwise data-only.
2. evidence_string: the row's annotated佐證句 (optional).
3. promise_string: the row's annotated承諾句 (optional).

same-page-content is the WHOLE matched page's OCR text (no before/after windowing);
an optional max_chars cap truncates very long pages. Doc/page joins reuse
core/human/predict/stage4/build_page_context.py.

`build_blocks()` returns the raw block texts; the final <tag> template is assembled
by build_prompt/template.py.

Data-use note: same-page-content injects raw OCR text beyond the `data` field, and
evidence_string / promise_string are annotation fields. All three exceed the
CLAUDE.md `data`-only default and are OFF unless explicitly enabled — they exist for
the test_add_context probe and require user sign-off before promotion.
"""
from __future__ import annotations

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
# stage3/core/build_prompt/add_context.py -> parents[3] == test_add_context.
DEFAULT_OFFSETS = Path(__file__).resolve().parents[3] / "data" / "offsets.jsonl"

# mode -> set of accepted offsets hit_kinds (None == accept all hits).
MODE_HIT_KINDS: dict[str, set[str] | None] = {
    "all": None,
    "hit_exact_window_norm_window": {"hit_exact_window", "hit_norm_window"},
}


def load_offsets(path: Path) -> dict[str, dict[str, Any]]:
    """id -> {hit_kind, matched_page_no} from offsets.jsonl."""
    out: dict[str, dict[str, Any]] = {}
    for row in bpc.read_jsonl(path):
        rid = str(row.get("id", "")).strip()
        if rid:
            out[rid] = row
    return out


class ContextBuilder:
    def __init__(
        self,
        *,
        mode: str = "all",
        add_same_page: bool = True,
        add_evidence_string: bool = False,
        add_promise_string: bool = False,
        max_chars: int = 0,
        doc_table: Path = DEFAULT_DOC_TABLE,
        page_table: Path = DEFAULT_PAGE_TABLE,
        offsets_path: Path = DEFAULT_OFFSETS,
    ) -> None:
        if mode not in MODE_HIT_KINDS:
            raise ValueError(f"unknown context mode {mode!r}; choices: {sorted(MODE_HIT_KINDS)}")
        self.mode = mode
        self.accepted_kinds = MODE_HIT_KINDS[mode]
        self.add_same_page = add_same_page
        self.add_evidence_string = add_evidence_string
        self.add_promise_string = add_promise_string
        # max chars of same-page text to inject; <= 0 means the whole page.
        self.max_chars = max_chars

        needs_page = add_same_page
        if needs_page:
            doc_rows = bpc.read_jsonl(doc_table)
            page_rows = bpc.read_jsonl(page_table)
            self.url2doc = bpc.build_url2doc(doc_rows)
            self.id2doc = bpc.build_id2doc(doc_rows)
            self.by_page = bpc.build_page_index(page_rows)
            self.norm_index = bpc.build_norm_page_index(self.by_page)
            self.doc_pages = bpc.build_doc_pages(self.by_page)
            self.offsets = load_offsets(offsets_path)
        else:
            self.url2doc = self.id2doc = self.by_page = {}
            self.norm_index = self.doc_pages = self.offsets = {}

    # ── doc / page resolution ────────────────────────────────────────────────
    def _doc_id(self, row: dict[str, Any]) -> str | None:
        return self.url2doc.get(row.get("pdf_url")) or self.id2doc.get(str(row.get("id")))

    def _cap(self, text: str) -> str:
        return text[: self.max_chars] if self.max_chars and self.max_chars > 0 else text

    def _page_text(self, doc_id: str, page_no: int) -> str:
        return self._cap(self.by_page.get((doc_id, page_no), ""))

    def _same_page_context(self, row: dict[str, Any], promise: str) -> tuple[str, str]:
        """Return (whole_matched_page_text, hit_kind). Empty on any miss."""
        rid = str(row.get("id", "")).strip()
        doc_id = self._doc_id(row)
        if not doc_id:
            return "", "miss_no_doc"

        off = self.offsets.get(rid)
        if off is not None:
            hit_kind = off.get("hit_kind", "")
            if self.accepted_kinds is not None and hit_kind not in self.accepted_kinds:
                return "", f"skip_mode:{hit_kind}"
            try:
                page_no = int(off.get("matched_page_no"))
            except (TypeError, ValueError):
                return "", "miss_no_matched_page"
            text = self._page_text(doc_id, page_no)
            return (text, f"offset_{hit_kind}") if text else ("", "miss_empty_page")

        # No offsets row. Only the permissive "all" mode falls back to live location.
        if self.accepted_kinds is not None:
            return "", "skip_mode:no_offset"
        try:
            page_number = int(row.get("page_number"))
        except (TypeError, ValueError):
            page_number = None
        prefix = bpc.collapse(promise)[: bpc.DEFAULT_PREFIX_CHARS]
        prefix_norm = bpc.normalize_text(promise)[: bpc.DEFAULT_PREFIX_CHARS]
        anchor = bpc.locate_anchor_fallback(
            doc_id, page_number, prefix, prefix_norm, self.by_page,
            bpc.DEFAULT_PAGE_OFFSETS, self.norm_index, self.doc_pages,
        )
        if anchor is None:
            return "", "miss_no_match"
        page_no, _match_idx, method = anchor
        text = self._page_text(doc_id, page_no)
        return (text, f"live_{method}") if text else ("", "miss_empty_page")

    # ── public API ───────────────────────────────────────────────────────────
    def build_blocks(self, row: dict[str, Any], data: str) -> tuple[dict[str, str], str]:
        """Return (blocks, hit_kind). blocks has keys same_page_context /
        promise_string / evidence_string (empty string when not injected)."""
        blocks = {"same_page_context": "", "promise_string": "", "evidence_string": ""}
        hit_kind = "data_only"
        if self.add_promise_string:
            blocks["promise_string"] = str(row.get("promise_string", "")).strip()
        if self.add_evidence_string:
            blocks["evidence_string"] = str(row.get("evidence_string", "")).strip()
        if self.add_same_page:
            blocks["same_page_context"], hit_kind = self._same_page_context(row, data)
        return blocks, hit_kind
