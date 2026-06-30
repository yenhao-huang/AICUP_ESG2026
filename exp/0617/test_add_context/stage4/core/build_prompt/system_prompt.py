#!/usr/bin/env python3
"""System-prompt loading for the modular Stage 4 VLM predictor.

Sets the active prompt file (the classifier instructions sent as the chat
`system` message). The default points at the Stage 4 vanilla timeline prompt
(a copy of configs/prompt/stage4/codex/boundary_rules_v4.txt); pass any other
path to swap instructions (add-context / add-image / add-promise / all).
"""
from __future__ import annotations

from pathlib import Path

# stage4/core/build_prompt/system_prompt.py -> parents[2] == stage4
# (which holds prompts/codex/).
_STAGE4_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = _STAGE4_ROOT / "prompts" / "codex" / "vanilla.txt"


def load_system_prompt(prompt_path: Path | str | None = None) -> str:
    """Read the system prompt text. Falls back to DEFAULT_PROMPT_PATH when None."""
    path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
