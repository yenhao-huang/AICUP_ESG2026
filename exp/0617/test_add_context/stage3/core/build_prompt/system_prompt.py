#!/usr/bin/env python3
"""System-prompt loading for the modular Stage 3 VLM predictor.

Sets the active prompt file (the classifier instructions sent as the chat
`system` message). The default points at the same-page-context scoped prompt
used by the test_add_context experiment; pass any other path (for example a
stage4 prompt like configs/prompt/stage4/codex/*.txt) to swap instructions.
"""
from __future__ import annotations

from pathlib import Path

# stage3/core/build_prompt/system_prompt.py -> parents[2] == stage3,
# parents[3] == test_add_context (which holds prompts/).
_EXP_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_PATH = _EXP_ROOT / "prompts" / "clear_notclear_with_context_scoped.txt"


def load_system_prompt(prompt_path: Path | str | None = None) -> str:
    """Read the system prompt text. Falls back to DEFAULT_PROMPT_PATH when None."""
    path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
