#!/usr/bin/env python3
"""Output parsing for the modular Stage 3 VLM predictor.

Normalizes a raw model reply into one Stage 3 label, tolerating:
- markdown fences (```...```), inline backticks, and bold/heading markers;
- JSON / quoted-string wrappers ({"evidence_quality": "Clear"}, '"Clear"');
- chain-of-thought replies that end with 「輸出：<label>」 / "label: ...";
- English / 中文 aliases and full/half-width punctuation.

Returns (label, reason). label is one of schemas.ALLOWED_LABELS; on failure it is
"N/A" with a diagnostic reason (the caller maps that to an error sentinel).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # stage3/core
import schemas  # noqa: E402

LABELS = {"Clear", "Not Clear", "Misleading"}

# Longest-label-first so "Clear" does not match inside "Not Clear" / "不明確".
_LABEL_PATTERNS = [
    ("Misleading", ("misleading", "greenwashing", "誤導", "漂綠")),
    ("Not Clear", ("not clear", "notclear", "unclear", "不清楚", "不明確", "模糊")),
    ("Clear", ("clear", "明確", "清楚", "可驗證")),
]
_ALIASES = {
    "Clear": {"clear", "明確", "清楚", "可驗證"},
    "Not Clear": {"not clear", "unclear", "notclear", "不清楚", "不明確", "模糊"},
    "Misleading": {"misleading", "greenwashing", "誤導", "漂綠"},
}


def strip_markdown(text: str) -> str:
    """Drop fenced code blocks' fences and inline markdown decoration."""
    t = text.strip()
    # ```json\n...\n``` -> inner content
    fenced = re.match(r"^```[a-zA-Z0-9]*\s*\n?(.*?)\n?```$", t, re.DOTALL)
    if fenced:
        t = fenced.group(1).strip()
    t = t.replace("`", "")
    t = re.sub(r"[*#>]+", "", t)
    return t.strip()


def _unwrap_json(text: str):
    """If the reply is JSON / a quoted string, pull out the inner label value."""
    s = text.strip()
    if not (s.startswith("{") or s.startswith("[") or s.startswith('"')):
        return text
    try:
        loaded = json.loads(s)
    except json.JSONDecodeError:
        return text
    if isinstance(loaded, str):
        return loaded
    if isinstance(loaded, dict):
        for key in (schemas.TARGET_FIELD, "label", "prediction", "answer", "output"):
            if key in loaded:
                return str(loaded[key])
    return text


def _extract_label_from_text(text: str) -> str | None:
    """Pull a label from a free-text / CoT reply, preferring the last output marker."""
    seg = text
    markers = list(re.finditer(r"(?:輸出|output|答案|label)\s*[:：]\s*", text, re.IGNORECASE))
    if markers:
        seg = text[markers[-1].end():]
    norm = re.sub(r"[\s_-]+", " ", seg.lower())
    for label, pats in _LABEL_PATTERNS:
        if any(p in norm for p in pats):
            return label
    return None


def parse_label(raw: object) -> tuple[str, str]:
    """Normalize a raw reply into (label, reason)."""
    text = strip_markdown(str(_unwrap_json(str(raw or ""))))
    if not text:
        return "N/A", "empty_output"
    lowered = re.sub(r"[\s_-]+", " ", text.lower()).strip()
    hits = [label for label, names in _ALIASES.items() if lowered in names or text in names]
    if len(hits) == 1:
        return hits[0], "normalized_label"
    if text in LABELS:
        return text, "normalized_label"
    extracted = _extract_label_from_text(text)
    if extracted:
        return extracted, "normalized_label_cot"
    return "N/A", "invalid_label"
