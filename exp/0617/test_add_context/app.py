#!/usr/bin/env python3
"""Gradio UI for inspecting online Stage 3 Qwen predictions with page context.

Run:
  /workspace/esg_contest/.venv/bin/python app.py --port 7862
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import gradio as gr

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.human.predict.stage3.pred_by_codex import load_system_prompt, read_data_rows  # noqa: E402
from core.human.predict.stage3.pred_by_qwen import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROMPT_PATH,
    augment_data,
    build_context_index,
    normalize_evidence_quality,
    qwen_classify,
)
import core.human.predict.stage4.build_page_context as bpc  # noqa: E402

DEFAULT_DATA = HERE / "data" / "val_ctxhit.json"
DEFAULT_UI_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"


def short_text(text: str, limit: int = 58) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + "..."


def split_context(augmented: str, original: str) -> tuple[str, str]:
    if augmented == original:
        return original, ""
    marker = bpc.CTX_HEADER + "\n"
    if marker not in augmented:
        return augmented, ""
    before, context = augmented.split(marker, 1)
    prefix = bpc.PROMISE_HEADER
    promise = before.strip()
    if promise.startswith(prefix):
        promise = promise[len(prefix):].strip()
    return promise, context.strip()


class AppState:
    def __init__(self, data_path: Path, prompt_path: Path) -> None:
        self.data_path = data_path
        self.prompt_path = prompt_path
        self.rows = read_data_rows(data_path)
        self.system_prompt = load_system_prompt(prompt_path)
        self.context_index: dict[str, Any] | None = None

    def get_context_index(self) -> dict[str, Any]:
        if self.context_index is None:
            self.context_index = build_context_index(bpc.DEFAULT_DOC_TABLE, bpc.DEFAULT_PAGE_TABLE)
        return self.context_index


STATE: AppState


def choices() -> list[str]:
    out = []
    for idx, row in enumerate(STATE.rows):
        row_id = str(row.get("id", idx + 1))
        out.append(f"{idx + 1:04d} | id {row_id} | {short_text(row.get('data', ''))}")
    return out


def parse_choice(choice: str | None) -> int:
    if not choice:
        return 0
    try:
        return max(0, int(choice.split("|", 1)[0].strip()) - 1)
    except ValueError:
        return 0


def build_prompt_view(
    idx: int,
    add_context: bool,
    context_budget: int,
    context_window: str,
    cross_page: bool,
    prefix_chars: int,
) -> tuple[str, str, str, str, str, str]:
    idx = max(0, min(idx, len(STATE.rows) - 1))
    row = STATE.rows[idx]
    original = str(row.get("data", ""))
    hit = "data_only"
    augmented = original
    if add_context:
        augmented, hit = augment_data(
            row,
            original,
            STATE.get_context_index(),
            context_budget,
            context_window,
            cross_page,
            prefix_chars,
        )
    promise, context = split_context(augmented, original)
    metadata = json.dumps(
        {
            "index": idx + 1,
            "id": row.get("id"),
            "company": row.get("company"),
            "page_number": row.get("page_number"),
            "pdf_url": row.get("pdf_url"),
            "context_hit": hit,
            "gold_evidence_status": row.get("evidence_status"),
            "gold_evidence_quality": row.get("evidence_quality"),
        },
        ensure_ascii=False,
        indent=2,
    )
    user_prompt = f"DATA：{augmented}"
    return metadata, original, context, STATE.system_prompt, user_prompt, hit


def ground_truth_view(idx: int) -> str:
    idx = max(0, min(idx, len(STATE.rows) - 1))
    row = STATE.rows[idx]
    labels = {
        "promise_status": row.get("promise_status", ""),
        "evidence_status": row.get("evidence_status", ""),
        "evidence_quality": row.get("evidence_quality", ""),
        "verification_timeline": row.get("verification_timeline", ""),
    }
    return "\n".join(f"{key}: {value}" for key, value in labels.items())


def render(choice: str | None, add_context: bool, context_budget: int, context_window: str,
           cross_page: bool, prefix_chars: int) -> tuple[int, str, str, str, str, str, str, str, str, str]:
    idx = parse_choice(choice)
    metadata, original, context, system_prompt, user_prompt, hit = build_prompt_view(
        idx, add_context, context_budget, context_window, cross_page, prefix_chars
    )
    return idx, metadata, ground_truth_view(idx), original, context, system_prompt, user_prompt, "", "", f"context_hit={hit}"


def go(idx: int, delta: int, add_context: bool, context_budget: int, context_window: str,
       cross_page: bool, prefix_chars: int) -> tuple[Any, ...]:
    idx = max(0, min(idx + delta, len(STATE.rows) - 1))
    choice = choices()[idx]
    rendered = render(choice, add_context, context_budget, context_window, cross_page, prefix_chars)
    return (idx, gr.update(value=choice), *rendered[1:])


def predict(
    idx: int,
    add_context: bool,
    context_budget: int,
    context_window: str,
    cross_page: bool,
    prefix_chars: int,
    endpoint: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
) -> tuple[str, str, str, str]:
    metadata, _original, _context, _system_prompt, user_prompt, hit = build_prompt_view(
        idx, add_context, context_budget, context_window, cross_page, prefix_chars
    )
    augmented = user_prompt.removeprefix("DATA：")
    try:
        raw = qwen_classify(
            endpoint,
            model,
            STATE.system_prompt,
            augmented,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=False,
            timeout=timeout,
            retries=retries,
        )
        label, reason = normalize_evidence_quality(raw)
        if reason == "empty_codex_label":
            label, reason = "Excpetion:empty word", "model_output_error:empty_codex_label"
        elif label == "N/A":
            label, reason = "except", f"model_output_error:{reason}"
        status = f"label={label}\nreason={reason}\ncontext_hit={hit}"
        return raw, label, status, ground_truth_view(idx)
    except Exception as exc:
        return "", "except", f"qwen_error={type(exc).__name__}: {exc}\ncontext_hit={hit}\nmetadata={metadata}", ground_truth_view(idx)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Stage3 Qwen Context Inspector") as demo:
        gr.Markdown("# Stage3 Qwen Context Inspector")
        idx_state = gr.State(0)

        with gr.Row():
            item = gr.Dropdown(choices=choices(), value=choices()[0] if STATE.rows else None,
                               label="題目", scale=5)
            prev_btn = gr.Button("Prev", scale=1)
            next_btn = gr.Button("Next", scale=1)

        with gr.Row():
            add_context = gr.Checkbox(value=True, label="add same-page context")
            cross_page = gr.Checkbox(value=True, label="cross page")
            context_window = gr.Radio(["after_biased", "symmetric"], value="after_biased",
                                      label="context window")
            context_budget = gr.Number(value=800, precision=0, label="context budget")
            prefix_chars = gr.Number(value=bpc.DEFAULT_PREFIX_CHARS, precision=0, label="prefix chars")

        with gr.Row():
            endpoint = gr.Textbox(value=DEFAULT_UI_ENDPOINT, label="endpoint")
            model = gr.Textbox(value=DEFAULT_MODEL, label="model")
            max_tokens = gr.Number(value=16, precision=0, label="max tokens")
            temperature = gr.Number(value=0.0, label="temperature")
            timeout = gr.Number(value=120, precision=0, label="timeout")
            retries = gr.Number(value=4, precision=0, label="retries")

        with gr.Row():
            metadata = gr.Code(label="metadata / gold for reference", language="json")
            ground_truth = gr.Textbox(label="ground truth", lines=4, interactive=False)
            status = gr.Textbox(label="prediction status", lines=7, interactive=False)

        original = gr.Textbox(label="original data", lines=5, max_lines=10, interactive=False)
        context = gr.Textbox(label="same-page context", lines=8, max_lines=18, interactive=False)

        with gr.Row():
            system_prompt = gr.Textbox(label="system prompt", lines=15, max_lines=24, interactive=False)
            user_prompt = gr.Textbox(label="user prompt sent to Qwen", lines=15, max_lines=24,
                                     interactive=False)

        with gr.Row():
            predict_btn = gr.Button("Online Predict", variant="primary")
            raw = gr.Textbox(label="raw Qwen output", lines=4, interactive=False)
            label = gr.Textbox(label="normalized label", lines=1, interactive=False)

        render_inputs = [item, add_context, context_budget, context_window, cross_page, prefix_chars]
        render_outputs = [
            idx_state,
            metadata,
            ground_truth,
            original,
            context,
            system_prompt,
            user_prompt,
            raw,
            label,
            status,
        ]
        item.change(render, render_inputs, render_outputs)
        for ctl in [add_context, context_budget, context_window, cross_page, prefix_chars]:
            ctl.change(render, render_inputs, render_outputs)

        nav_inputs = [idx_state, add_context, context_budget, context_window, cross_page, prefix_chars]
        nav_outputs = [
            idx_state,
            item,
            metadata,
            ground_truth,
            original,
            context,
            system_prompt,
            user_prompt,
            raw,
            label,
            status,
        ]
        prev_btn.click(lambda *args: go(args[0], -1, *args[1:]), nav_inputs, nav_outputs)
        next_btn.click(lambda *args: go(args[0], 1, *args[1:]), nav_inputs, nav_outputs)

        predict_btn.click(
            predict,
            [
                idx_state,
                add_context,
                context_budget,
                context_window,
                cross_page,
                prefix_chars,
                endpoint,
                model,
                max_tokens,
                temperature,
                timeout,
                retries,
            ],
            [raw, label, status, ground_truth],
        )
        demo.load(render, render_inputs, render_outputs)
    return demo


def main() -> None:
    global STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7862)))
    args = parser.parse_args()

    STATE = AppState(args.data, args.prompt_path)
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
