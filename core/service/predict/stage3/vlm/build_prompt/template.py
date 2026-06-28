#!/usr/bin/env python3
"""Prompt template assembly for the modular Stage 3 VLM predictor.

Renders the request as a single tagged block, in this fixed order:

    <system-prompt> ... </system-prompt>             (classifier instructions)
    <page-abstract> ... </page-abstract>             (optional)
    <same-page-context> ... </same-page-context>   (optional)
    <promise-string> ... </promise-string>          (optional)
    <evidence-string> ... </evidence-string>         (optional)
    <data-prompt> ... </data-prompt>                 (the sentence to classify)

There is no before/after windowing — same-page-context is the whole matched page.
Optional blocks are omitted when empty. The system prompt is part of this single
template (it is not sent as a separate chat `system` role).
"""
from __future__ import annotations

TAG_ORDER = ("system-prompt", "page-abstract", "same-page-context", "promise-string", "evidence-string", "data-prompt")


def _wrap(tag: str, text: str) -> str:
    return f"<{tag}>\n{text.strip()}\n</{tag}>"


def render(
    *,
    system_prompt: str,
    data: str,
    page_abstract: str = "",
    same_page_context: str = "",
    promise_string: str = "",
    evidence_string: str = "",
) -> str:
    values = {
        "page-abstract": page_abstract,
        "same-page-context": same_page_context,
        "promise-string": promise_string,
        "evidence-string": evidence_string,
        "data-prompt": data,
        "system-prompt": system_prompt,
    }
    # data-prompt / system-prompt are always present; the rest only when non-empty.
    always = {"data-prompt", "system-prompt"}
    parts = [_wrap(tag, values[tag]) for tag in TAG_ORDER if tag in always or values[tag].strip()]
    return "\n\n".join(parts)
