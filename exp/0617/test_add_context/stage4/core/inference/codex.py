#!/usr/bin/env python3
"""Codex CLI inference backend.

Runs `codex exec --json` as a subprocess and returns its last message. Codex does
not expose output-token logprobs, so token confidence is always None (per the
schema, the Codex path leaves token_confidence empty).

Images are supported via `codex exec --image <FILE>`: each entry in `image_urls`
is either a local image path or a `data:<mime>;base64,...` URL (decoded to a temp
file). Because `--image <FILE>...` is variadic and would otherwise swallow the
positional prompt, the prompt is sent on stdin whenever images are attached.
"""
from __future__ import annotations

import base64
import binascii
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

CODEX_RUN_DIR = Path("/tmp")
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_CODEX_BIN = "codex"

_DATA_URL_SUFFIX = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/webp": ".webp", "image/gif": ".gif",
}


def _resolve_image_paths(image_urls: list[str] | None) -> tuple[list[str], list[Path]]:
    """Return (paths to pass to --image, temp files to clean up).

    Accepts local filesystem paths (used as-is) and `data:` base64 URLs (decoded
    to a temp file). Unreadable / malformed entries are skipped.
    """
    if not image_urls:
        return [], []
    paths: list[str] = []
    tmp_files: list[Path] = []
    for url in image_urls:
        if not url:
            continue
        if url.startswith("data:"):
            try:
                header, b64 = url.split(",", 1)
                mime = header[len("data:"):].split(";", 1)[0].strip().lower()
                suffix = _DATA_URL_SUFFIX.get(mime, ".png")
                with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
                    tmp.write(base64.b64decode(b64))
                    tmp_path = Path(tmp.name)
            except (ValueError, binascii.Error):
                continue
            paths.append(str(tmp_path))
            tmp_files.append(tmp_path)
        elif Path(url).exists():
            paths.append(url)
    return paths, tmp_files


def classify(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    codex_bin: str = DEFAULT_CODEX_BIN,
    image_urls: list[str] | None = None,
    timeout: int = 300,
    **_ignored: Any,
) -> tuple[str, None]:
    """Send the full tagged template as the Codex prompt. Return (raw_text, None).

    Raises RuntimeError on a non-zero Codex exit.
    """

    image_paths, tmp_files = _resolve_image_paths(image_urls)

    with tempfile.NamedTemporaryFile("r", encoding="utf-8", delete=False) as tmp:
        last_message_path = Path(tmp.name)
    codex_command = [
        codex_bin, "exec", "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--model", model,
        "--skip-git-repo-check",
        "--output-last-message", str(last_message_path),
    ]
    for path in image_paths:
        codex_command += ["--image", path]
    # `--image <FILE>...` is variadic and would consume a positional prompt, so
    # when images are attached send the prompt on stdin instead; otherwise keep
    # the prompt positional (unchanged text-only behaviour).
    stdin_text: str | None = None
    if image_paths:
        stdin_text = prompt
    else:
        codex_command.append(prompt)
    shell_command = (
        f"cd {shlex.quote(str(CODEX_RUN_DIR))} && "
        + " ".join(shlex.quote(part) for part in codex_command)
    )
    try:
        result = subprocess.run(
            shell_command, shell=True, text=True, input=stdin_text,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
        )
        last_message = last_message_path.read_text(encoding="utf-8").strip()
    finally:
        last_message_path.unlink(missing_ok=True)
        for path in tmp_files:
            path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Codex CLI failed (exit={result.returncode}): {(result.stderr or result.stdout).strip()}"
        )
    return last_message or result.stdout.strip(), None
