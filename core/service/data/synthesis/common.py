"""Shared helpers for offline synthesis data generation."""

from __future__ import annotations

import json
import random
import re
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    resolved = resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [dict(row) for row in payload]


def write_json_atomic(path: Path, rows: Any) -> None:
    output = resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")
        tmp = Path(f.name)
    tmp.replace(output)


def label_counts(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    return dict(Counter(str(row.get(column, "")) for row in rows))


def project(row: dict[str, Any], columns: tuple[str, ...]) -> dict[str, Any]:
    return {column: row.get(column, "") for column in columns}


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sample_with_replacement(
    pool: list[dict[str, Any]],
    n: int,
    *,
    seed: int,
    id_key: str = "id",
    duplicate_suffix: str = "_k",
) -> list[dict[str, Any]]:
    if not pool:
        raise ValueError("Cannot sample from an empty pool")
    rng = random.Random(seed)
    if n <= len(pool):
        indexes = rng.sample(range(len(pool)), n)
        return [dict(pool[index]) for index in indexes]

    output: list[dict[str, Any]] = []
    indexes = list(range(len(pool)))
    repeat = 0
    while len(output) < n:
        rng.shuffle(indexes)
        for index in indexes:
            row = dict(pool[index])
            if repeat > 0:
                row[id_key] = f"{row[id_key]}{duplicate_suffix}{repeat}"
            output.append(row)
            if len(output) >= n:
                break
        repeat += 1
    return output


def call_openai_compatible(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    temperature: float = 0.2,
) -> str:
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM response has no choices: {data}")
    content = choices[0].get("message", {}).get("content", "")
    return normalize_space(content)


def charset_overlap(a: str, b: str) -> float:
    left = set(normalize_space(a))
    right = set(normalize_space(b))
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def looks_like_leak(candidate: str, source: str, *, overlap_threshold: float = 0.92) -> bool:
    candidate = normalize_space(candidate)
    source = normalize_space(source)
    if not candidate or not source:
        return True
    return candidate == source or source in candidate or candidate in source or charset_overlap(candidate, source) >= overlap_threshold
