#!/usr/bin/env python3
"""Gradio UI to verify the same-page context resolved for a Stage 3 row.

Left  = the input example (the row: id, GT label, pdf_url, page_number, data,
        page_abstract).
Right = exactly what would be sent as context — resolved via the real
        ContextBuilder path (pdf_url -> doc_id -> offsets matched_page_no /
        weakly_matched_page_no), plus a check of whether the page text actually
        contains the `data` sentence.

Run:
    /workspace/esg_contest/.venv/bin/python \
      exp/integrated_stage_predictions/0617/test_add_context/stage3/verify_context_app.py --port 7864
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent           # stage3/
EXP = HERE.parent                                 # test_add_context/
sys.path.insert(0, str(HERE / "core"))
sys.path.insert(0, str(HERE / "core" / "build_prompt"))

import gradio as gr  # noqa: E402
from add_context import ContextBuilder, load_offsets, DEFAULT_OFFSETS  # noqa: E402

DATA_DIR = EXP / "data"
MODES = ["all", "hit_exact_window_norm_window"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s))


def list_datasets() -> list[str]:
    return sorted(p.name for p in DATA_DIR.glob("*.json"))


def load_rows(name: str) -> list[dict]:
    payload = json.loads((DATA_DIR / name).read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


# Caches keyed by simple args so we don't rebuild page indexes on every click.
_ROWS: dict[str, list[dict]] = {}
_BUILDER: dict[str, ContextBuilder] = {}
_OFFSETS = load_offsets(DEFAULT_OFFSETS)


def rows_for(name: str) -> list[dict]:
    if name not in _ROWS:
        _ROWS[name] = load_rows(name)
    return _ROWS[name]


def builder_for(mode: str) -> ContextBuilder:
    if mode not in _BUILDER:
        _BUILDER[mode] = ContextBuilder(add_same_page=True, mode=mode)
    return _BUILDER[mode]


def ids_for(name: str) -> list[str]:
    return [str(r.get("id")) for r in rows_for(name)]


def inspect(dataset: str, row_id: str, mode: str):
    rows = rows_for(dataset)
    row = next((r for r in rows if str(r.get("id")) == str(row_id)), None)
    if row is None:
        return "row not found", "row not found"

    data = str(row.get("data", ""))
    off = _OFFSETS.get(str(row_id), {})

    # LEFT: input example
    left = [
        f"### Input example — id `{row_id}`",
        f"- **GT evidence_quality**: `{row.get('evidence_quality','')}`",
        f"- **evidence_status (gate)**: `{row.get('evidence_status','')}`",
        f"- **company**: {row.get('company','')}",
        f"- **pdf_url**: {row.get('pdf_url','')}",
        f"- **page_number (row)**: `{row.get('page_number','')}`",
        "",
        "**data (要判斷的句子):**",
        f"> {data}",
    ]
    if str(row.get("page_abstract", "")).strip():
        left += ["", "**page_abstract:**", f"> {row.get('page_abstract')}"]

    # RIGHT: resolved context via the real ContextBuilder
    cb = builder_for(mode)
    doc_id = cb._doc_id(row)
    try:
        blocks, src = cb.build_blocks(row, data)
        ctx = blocks["same_page_context"]
        err = None
    except Exception as exc:  # strict mode can raise on empty matched_page_no
        ctx, src, err = "", "RAISED", f"{type(exc).__name__}: {exc}"

    prefix = _norm(data)[:25]
    contains = bool(prefix) and prefix in _norm(ctx)
    flag = "✅ context CONTAINS the data sentence" if ctx and contains else (
        "⚠️ context does NOT contain the data sentence" if ctx else "— no context injected")

    right = [
        f"### Context to send — mode `{mode}`",
        f"- **resolved doc_id**: `{doc_id}`",
        f"- **offsets.hit_kind**: `{off.get('hit_kind')}`",
        f"- **offsets.matched_page_no**: `{off.get('matched_page_no')}`",
        f"- **offsets.weakly_matched_page_no**: `{off.get('weakly_matched_page_no')}`",
        f"- **source (chosen)**: `{src}`",
        f"- **verify**: {flag}",
    ]
    if err:
        right += ["", f"**ERROR**: `{err}`"]
    right += ["", "**<same-page-context> (實際送出的內容):**",
              (f"```\n{ctx}\n```" if ctx else "_（此 mode 下不放 context）_")]

    return "\n".join(left), "\n".join(right)


def build_ui() -> gr.Blocks:
    datasets = list_datasets()
    default_ds = "val_nonNA.json" if "val_nonNA.json" in datasets else (datasets[0] if datasets else "")
    init_ids = ids_for(default_ds) if default_ds else []
    with gr.Blocks(title="Stage 3 context verifier") as demo:
        gr.Markdown("# Stage 3 同頁 context 驗證\n左：輸入範例　|　右：實際要送出的 context（含解析出的頁碼來源與包含性檢查）")
        with gr.Row():
            dataset = gr.Dropdown(datasets, value=default_ds, label="dataset (data/*.json)")
            row_id = gr.Dropdown(init_ids, value=(init_ids[0] if init_ids else None), label="row id")
            mode = gr.Radio(MODES, value="all", label="context-mode")
        with gr.Row():
            left = gr.Markdown(label="input")
            right = gr.Markdown(label="context")

        def on_dataset(ds):
            ids = ids_for(ds)
            return gr.update(choices=ids, value=(ids[0] if ids else None))

        dataset.change(on_dataset, dataset, row_id)
        for comp in (row_id, mode):
            comp.change(inspect, [dataset, row_id, mode], [left, right])
        dataset.change(inspect, [dataset, row_id, mode], [left, right])
        if init_ids:
            demo.load(inspect, [dataset, row_id, mode], [left, right])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7864)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)
