#!/usr/bin/env python3
"""Doc/page lookup helpers for the Stage 3 VLM predictor."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DOC_TABLE = ROOT / "data" / "generated" / "raw_doc_table.jsonl"
DEFAULT_PAGE_TABLE = ROOT / "data" / "generated" / "raw_page_table.jsonl"
DEFAULT_OFFSETS = ROOT / "data" / "generated" / "stage3_offsets.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"JSONL file not found: {source}")
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(dict(json.loads(line)))
    return rows


def build_url2doc(rows: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        url, doc_id = row.get("url"), row.get("doc_id")
        if url and doc_id:
            out[str(url)] = str(doc_id)
    return out


def build_id2doc(rows: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id:
            continue
        for sid in row.get("source_ids") or []:
            out[str(sid)] = doc_id
    return out


def collapse(text: str) -> str:
    return " ".join(str(text or "").split())


_NORM_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize_text(text: str) -> str:
    return _NORM_STRIP_RE.sub("", unicodedata.normalize("NFKC", str(text or "")))


def build_page_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    for row in rows:
        doc_id = str(row.get("doc_id") or "").strip()
        try:
            page_no = int(row.get("page_no") or row.get("page_number"))
        except (TypeError, ValueError):
            continue
        text = collapse(str(row.get("text_clean") or row.get("text") or row.get("content") or ""))
        if doc_id and text:
            out[(doc_id, page_no)] = text
    return out


def build_norm_page_index(
    by_page: dict[tuple[str, int], str],
) -> dict[tuple[str, int], tuple[str, list[int]]]:
    out: dict[tuple[str, int], tuple[str, list[int]]] = {}
    for key, text in by_page.items():
        norm_chars: list[str] = []
        pos_map: list[int] = []
        for i, ch in enumerate(text):
            for c in _NORM_STRIP_RE.sub("", unicodedata.normalize("NFKC", ch)):
                norm_chars.append(c)
                pos_map.append(i)
        out[key] = ("".join(norm_chars), pos_map)
    return out


def build_doc_pages(by_page: dict[tuple[str, int], str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for doc_id, page_no in by_page:
        out.setdefault(doc_id, []).append(page_no)
    for pages in out.values():
        pages.sort()
    return out
