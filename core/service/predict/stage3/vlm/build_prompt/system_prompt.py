#!/usr/bin/env python3
"""System-prompt loading for the modular Stage 3 VLM predictor.

Sets the active prompt file (the classifier instructions sent as the chat
`system` message). The default points at the Stage 3 VLM prompt under configs.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
DEFAULT_PROMPT_PATH = ROOT / "configs" / "prompts" / "stage3" / "add-context.txt"


def load_system_prompt(prompt_path: Path | str | None = None) -> str:
    """Read the system prompt text. Falls back to DEFAULT_PROMPT_PATH when None."""
    path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
