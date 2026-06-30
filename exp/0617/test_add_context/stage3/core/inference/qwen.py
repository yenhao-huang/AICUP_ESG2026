#!/usr/bin/env python3
"""Qwen (llama-server / OpenAI-compatible) inference backend.

Sends a system + user chat request to a /v1/chat/completions endpoint and returns
the raw reply text plus optional per-output-token confidences (from the endpoint
`logprobs`). Supports a list of image data: URLs for vision-language prediction.

Thinking is disabled by default (chat_template_kwargs.enable_thinking = false) so
the reply is a bare label.
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # stage3/core
from schemas import TokenConfidence  # noqa: E402

DEFAULT_ENDPOINT = "http://192.168.1.78:3132/v1/chat/completions"
DEFAULT_MODEL = "local-qwen"


def _content(text: str, image_urls: list[str] | None) -> Any:
    if not image_urls:
        return text
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for url in image_urls:
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _parse_token_confidence(choice: dict[str, Any]) -> list[TokenConfidence] | None:
    lp = choice.get("logprobs")
    if not isinstance(lp, dict):
        return None
    content = lp.get("content")
    if not isinstance(content, list):
        return None
    out: list[TokenConfidence] = []
    for item in content:
        if not isinstance(item, dict) or "logprob" not in item:
            continue
        logprob = float(item["logprob"])
        out.append(TokenConfidence(token=str(item.get("token", "")), logprob=logprob, prob=math.exp(logprob)))
    return out or None


def classify(
    prompt: str,
    *,
    role: str = "user",
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    image_urls: list[str] | None = None,
    max_tokens: int = 16,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    logprobs: bool = False,
    timeout: int = 120,
    retries: int = 4,
) -> tuple[str, list[TokenConfidence] | None]:
    """Send the full tagged template as one message. Return (raw_text, token_confidence|None)."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": role, "content": _content(prompt, image_urls)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 1
    payload = json.dumps(body).encode()
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            choice = r["choices"][0]
            raw = choice["message"]["content"]
            conf = _parse_token_confidence(choice) if logprobs else None
            return raw, conf
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            time.sleep(min(20, 2 * (attempt + 1)))
    raise RuntimeError(f"qwen request failed after {retries + 1} tries: {last}")
