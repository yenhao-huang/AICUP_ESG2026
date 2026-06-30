#!/usr/bin/env python3
"""Output schema for the modular Stage 4 VLM predictor (vlm_pred.py).

Defines the prediction CSV columns and an in-memory result object. The Qwen
backend can attach per-output-token confidence (from the endpoint logprobs);
the Codex CLI backend cannot, so `token_confidence` is left empty for Codex.

Stage 4 target is `verification_timeline`. The four live classes use the same
snake_case spelling as the benchmark ground truth, so prediction labels are
directly comparable to the GT field with no remapping. `N/A` stays valid for
the upstream gate (non-promise rows), but the model itself only emits the four
timeline classes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

STAGE = "stage4"
TARGET_FIELD = "verification_timeline"

# Stage 4 verification_timeline label space (snake_case, matches benchmark GT
# spelling). N/A is reserved for upstream gate filtering of non-promise rows.
ALLOWED_LABELS = (
    "already",
    "within_2_years",
    "between_2_and_5_years",
    "more_than_5_years",
    "N/A",
)

# CSV columns. `context_hit` records the same-page-content join outcome and
# `token_confidence` is a JSON string of [{token, prob}] (Qwen) or "" (Codex).
OUTPUT_COLUMNS = (
    "id",
    TARGET_FIELD,
    f"{TARGET_FIELD}_raw",
    f"{TARGET_FIELD}_source",
    f"{TARGET_FIELD}_reason",
    "context_hit",
    "token_confidence",
)


@dataclass
class TokenConfidence:
    token: str
    logprob: float
    prob: float

    def to_dict(self) -> dict[str, object]:
        return {"token": self.token, "logprob": round(self.logprob, 5), "prob": round(self.prob, 5)}


@dataclass
class PredictionResult:
    id: str
    label: str
    raw: str = ""
    source: str = ""
    reason: str = ""
    context_hit: str = "data_only"
    # Qwen-only: per-output-token confidences. None for Codex (no logprobs).
    token_confidence: Optional[list[TokenConfidence]] = field(default=None)

    def to_row(self) -> dict[str, str]:
        if self.token_confidence:
            tc = json.dumps([t.to_dict() for t in self.token_confidence], ensure_ascii=False)
        else:
            tc = ""
        return {
            "id": self.id,
            TARGET_FIELD: self.label,
            f"{TARGET_FIELD}_raw": self.raw,
            f"{TARGET_FIELD}_source": self.source,
            f"{TARGET_FIELD}_reason": self.reason,
            "context_hit": self.context_hit,
            "token_confidence": tc,
        }
