#!/usr/bin/env python3
"""Output parsing for the modular Stage 4 VLM predictor.

Normalizes a raw model reply into one Stage 4 verification_timeline label,
tolerating:
- markdown fences (```...```), inline backticks, and bold/heading markers;
- JSON / quoted-string wrappers ({"verification_timeline": "already"}, '"already"');
- chain-of-thought replies that end with 「輸出：<label>」 / "label: ...";
- snake_case / spaced / 中文 aliases and full/half-width punctuation.

Returns (label, reason). label is one of schemas.ALLOWED_LABELS (the four live
timeline classes); on failure it is "N/A" with a diagnostic reason (the caller
maps that to an error sentinel).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # stage4/core
import schemas  # noqa: E402

LABELS = {"already", "within_2_years", "between_2_and_5_years", "more_than_5_years"}

# Longest / most-specific first so "more_than_5_years" is not shadowed by a
# generic "year" match, and "between_2_and_5_years" wins over "within".
_LABEL_PATTERNS = [
    ("more_than_5_years", ("more_than_5_years", "more than 5 years", "morethan5",
                           "超過5年", "超過五年", "5年以上", "五年以上", "長期")),
    ("between_2_and_5_years", ("between_2_and_5_years", "between 2 and 5 years",
                               "2to5", "2-5", "2~5", "2至5年", "二到五年", "中期")),
    ("within_2_years", ("within_2_years", "within 2 years", "within2",
                        "2年內", "兩年內", "2年以內", "短期")),
    ("already", ("already", "done", "completed", "已完成", "已達成", "已實施",
                 "現在", "當期")),
]
_ALIASES = {
    "already": {"already", "已完成", "已達成"},
    "within_2_years": {"within_2_years", "within 2 years", "2年內", "兩年內"},
    "between_2_and_5_years": {"between_2_and_5_years", "between 2 and 5 years", "2至5年"},
    "more_than_5_years": {"more_than_5_years", "more than 5 years", "5年以上", "五年以上"},
}


def strip_markdown(text: str) -> str:
    """Drop fenced code blocks' fences and inline markdown decoration."""
    t = text.strip()
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


def _norm(s: str) -> str:
    """Lowercase and collapse separators so 'Within 2 Years' == 'within_2_years'."""
    return re.sub(r"[\s_\-~]+", " ", s.lower()).strip()


def _extract_label_from_text(text: str) -> str | None:
    """Pull a label from a free-text / CoT reply, preferring the last output marker."""
    seg = text
    markers = list(re.finditer(r"(?:輸出|output|答案|label)\s*[:：]\s*", text, re.IGNORECASE))
    if markers:
        seg = text[markers[-1].end():]
    norm = _norm(seg)
    for label, pats in _LABEL_PATTERNS:
        if any(_norm(p) in norm for p in pats):
            return label
    return None


def parse_label(raw: object) -> tuple[str, str]:
    """Normalize a raw reply into (label, reason)."""
    text = strip_markdown(str(_unwrap_json(str(raw or ""))))
    if not text:
        return "N/A", "empty_output"
    normed = _norm(text)
    # Exact / alias match on the whole (short) reply first.
    if normed.replace(" ", "_") in LABELS:
        return normed.replace(" ", "_"), "normalized_label"
    hits = [label for label, names in _ALIASES.items() if normed in {_norm(n) for n in names}]
    if len(hits) == 1:
        return hits[0], "normalized_label"
    extracted = _extract_label_from_text(text)
    if extracted:
        return extracted, "normalized_label_cot"
    return "N/A", "invalid_label"
