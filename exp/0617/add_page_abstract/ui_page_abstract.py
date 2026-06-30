#!/usr/bin/env python3
"""Gradio inspector for page abstracts.

Browse, per unique matched page: the page image, the payload (the page OCR text
that was sent to the model = text_clean), the generated abstract, and the val/test
rows whose sentence falls on that page.

Datasets = subdirs holding page_abstracts.jsonl + *_with_page_abstract.jsonl:
  val  -> add_page_abstract/            (val_ctxhit.json, 551 pages)
  test -> add_page_abstract/test2000/   (vpesg4k_test_2000.json, 1153 pages)

Usage:
  .venv/bin/python exp/integrated_stage_predictions/0617/add_page_abstract/ui_page_abstract.py
  ... --host 127.0.0.1 --port 7864
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from core.human.predict.stage4.build_page_context import (  # noqa: E402
    DEFAULT_PAGE_TABLE,
    read_jsonl,
)

DATASETS = {
    "val (551 pages)": HERE,
    "test2000 (1598 pages)": HERE / "test2000",
}


def _read_jsonl_local(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def build_page_text(page_table: Path) -> dict[tuple[str, int], str]:
    """(doc_id, page_no) -> full text_clean payload."""
    out: dict[tuple[str, int], str] = {}
    for r in read_jsonl(page_table):
        doc = r.get("doc_id")
        try:
            pno = int(r.get("page_no"))
        except (TypeError, ValueError):
            continue
        if not doc:
            continue
        out[(doc, pno)] = r.get("text_clean") or r.get("text") or ""
    return out


def build_page_images(page_table: Path) -> dict[tuple[str, int], str]:
    """(doc_id, page_no) -> page image_file (relative path from raw_page_table)."""
    out: dict[tuple[str, int], str] = {}
    for r in read_jsonl(page_table):
        doc = r.get("doc_id")
        try:
            pno = int(r.get("page_no"))
        except (TypeError, ValueError):
            continue
        if not doc:
            continue
        out[(doc, pno)] = r.get("image_file") or r.get("image_path") or ""
    return out


def load_dataset(ds_dir: Path) -> tuple[list[dict], dict[tuple, list[dict]]]:
    pages = _read_jsonl_local(ds_dir / "page_abstracts.jsonl")
    rows_file = next(ds_dir.glob("*_with_page_abstract.jsonl"))
    rows = _read_jsonl_local(rows_file)
    rows_by_page: dict[tuple, list[dict]] = {}
    for r in rows:
        k = (r.get("matched_doc_id"), r.get("matched_page_no"))
        if k[0] is None:
            continue
        rows_by_page.setdefault(k, []).append(r)
    return pages, rows_by_page


def build_app(page_text: dict[tuple[str, int], str],
              page_img: dict[tuple[str, int], str]) -> gr.Blocks:
    cache: dict[str, tuple[list[dict], dict]] = {}

    def get(ds_label: str):
        if ds_label not in cache:
            cache[ds_label] = load_dataset(DATASETS[ds_label])
        return cache[ds_label]

    def companies(ds_label: str) -> list[str]:
        pages, _ = get(ds_label)
        return ["全部"] + sorted({p.get("company") or "" for p in pages})

    def filtered(ds_label: str, company: str, only_ab: bool) -> list[int]:
        pages, _ = get(ds_label)
        idxs = []
        for i, p in enumerate(pages):
            if company and company != "全部" and (p.get("company") or "") != company:
                continue
            if only_ab and not (p.get("abstract") or "").strip():
                continue
            idxs.append(i)
        return idxs

    def render(pos: int, ds_label: str, company: str, only_ab: bool):
        pages, rows_by_page = get(ds_label)
        idxs = filtered(ds_label, company, only_ab)
        if not idxs:
            return None, "（此篩選沒有資料）", "", "", "", gr.update(maximum=0, value=0)
        pos = max(0, min(int(pos), len(idxs) - 1))
        p = pages[idxs[pos]]

        key = (p.get("doc_id"), p.get("page_no"))
        img_rel = page_img.get(key, "") or ""   # image_file from raw_page_table
        img = str(REPO_ROOT / img_rel) if img_rel else None
        if img and not Path(img).exists():
            img = None

        payload = page_text.get(key, "") or "（raw_page_table 找不到此頁文字）"
        rows = rows_by_page.get(key, [])

        meta = "\n".join([
            f"篩選內: {pos + 1} / {len(idxs)}（全部 {len(pages)} 頁）",
            f"company: {p.get('company','')}   ticker: {p.get('ticker','')}",
            f"doc_id: {p.get('doc_id','')}",
            f"page_no: {p.get('page_no','')}   page_id: {p.get('page_id','')}",
            f"payload 字數 (n_chars): {p.get('n_chars','')}",
            f"對應 row 數: {len(rows)}",
            f"image_file: {img_rel or '(空)'}",
        ])
        abstract = p.get("abstract") or "（空 abstract）"
        rows_txt = "\n\n".join(
            f"#{r.get('id')}  (page_number={r.get('page_number')})\n{r.get('data','')}"
            for r in rows
        ) or "（沒有 row 對到此頁）"
        return (img, meta, payload, abstract, rows_txt,
                gr.update(maximum=len(idxs) - 1, value=pos))

    first_ds = next(iter(DATASETS))
    with gr.Blocks(title="Page Abstract Inspector") as demo:
        gr.Markdown(
            "## 頁面摘要檢視器\n"
            "逐頁檢視：**payload**（送進模型的頁面 OCR `text_clean`）、**頁面圖片**、"
            "對應的 **abstract**，以及落在這頁的 val/test 句子。"
        )
        with gr.Row():
            ds_dd = gr.Dropdown(choices=list(DATASETS), value=first_ds, label="資料集")
            co_dd = gr.Dropdown(choices=companies(first_ds), value="全部", label="篩選 company")
            only_cb = gr.Checkbox(value=False, label="只看有 abstract 的頁")
        with gr.Row():
            prev_btn = gr.Button("← 上一頁")
            slider = gr.Slider(0, 1, value=0, step=1, label="頁索引")
            next_btn = gr.Button("下一頁 →")
        with gr.Row():
            image = gr.Image(label="頁面圖片", height=760, type="filepath")
            with gr.Column():
                meta = gr.Textbox(label="中繼資料", lines=8)
                abstract = gr.Textbox(label="abstract（模型輸出）", lines=6)
                payload = gr.Textbox(label="payload（頁面 OCR text_clean，模型輸入）",
                                     lines=14, max_lines=30)
                rows_box = gr.Textbox(label="落在此頁的 row（id + data）", lines=8, max_lines=20)

        outs = [image, meta, payload, abstract, rows_box, slider]

        def _go(pos, ds, co, ob):
            return render(pos, ds, co, ob)

        def _step(pos, ds, co, ob, d):
            return render(int(pos) + d, ds, co, ob)

        def _ds_change(ds):
            return gr.update(choices=companies(ds), value="全部")

        slider.change(_go, [slider, ds_dd, co_dd, only_cb], outs)
        co_dd.change(lambda ds, co, ob: render(0, ds, co, ob), [ds_dd, co_dd, only_cb], outs)
        only_cb.change(lambda ds, co, ob: render(0, ds, co, ob), [ds_dd, co_dd, only_cb], outs)
        ds_dd.change(_ds_change, ds_dd, co_dd).then(
            lambda ds, ob: render(0, ds, "全部", ob), [ds_dd, only_cb], outs)
        prev_btn.click(lambda pos, ds, co, ob: _step(pos, ds, co, ob, -1),
                       [slider, ds_dd, co_dd, only_cb], outs)
        next_btn.click(lambda pos, ds, co, ob: _step(pos, ds, co, ob, +1),
                       [slider, ds_dd, co_dd, only_cb], outs)
        demo.load(_go, [slider, ds_dd, co_dd, only_cb], outs)

    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-table", type=Path, default=DEFAULT_PAGE_TABLE)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7864)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    page_text = build_page_text(args.page_table)
    page_img = build_page_images(args.page_table)
    demo = build_app(page_text, page_img)
    img_dir = REPO_ROOT / "data/raw_reports/images"
    allowed = [str(img_dir), str(img_dir.resolve())]
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                allowed_paths=allowed)


if __name__ == "__main__":
    main()
