#!/usr/bin/env python3
"""Gradio UI：左邊看 example（offsets 列的 `data` 文字＋中繼資料），
右邊看該 example 的 colnomic 檢索 top@3 頁面圖片（weakly_img_path）。

資料來源：
  - offsets（含 weakly_img_path）：0618/weakly_img_path/offsets.jsonl
  - example 文字：offsets.id --join--> 原始 train/val/test json 的 `data` 欄
  - 圖片：weakly_img_path 指向 /data/.../weakly_img/*.jpg（colnomic_esg_contest 檢索結果）

圖片以 PIL 物件回傳，避開 Gradio 6 的 allowed_paths 限制。

用法：
    P=/workspace/esg_contest/.venv/bin/python
    $P exp/integrated_stage_predictions/0618/weakly_img_path/view_colnomic_examples.py
    $P .../view_colnomic_examples.py --port 7870 --host 0.0.0.0
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import gradio as gr
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OFFSETS = (
    REPO_ROOT / "exp/integrated_stage_predictions/0618/weakly_img_path/offsets.jsonl"
)
_IMG_RE = re.compile(r"^(?P<stem>.+)_p(?P<page>\d+)__[0-9a-fA-F-]+\.(?:jpg|jpeg|png)$")


def load_rows(offsets: Path) -> list[dict]:
    with offsets.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_id_to_data(rows: list[dict]) -> dict[str, str]:
    """從 offsets 用到的所有 source_file 載入 id -> data 文字。"""
    id2data: dict[str, str] = {}
    for src in sorted({r.get("source_file", "") for r in rows if r.get("source_file")}):
        p = Path(src)
        if not p.exists():
            p = REPO_ROOT / Path(src).name
        if not p.exists():
            continue
        for rec in json.load(p.open(encoding="utf-8")):
            id2data[str(rec["id"])] = rec.get("data", "") or ""
    return id2data


def split_of(row_id: str) -> str:
    try:
        n = int(row_id)
    except (TypeError, ValueError):
        return "?"
    if 10001 <= n <= 11000:
        return "train"
    if 11001 <= n <= 12000:
        return "val"
    if 12001 <= n <= 14000:
        return "test"
    return "?"


def caption_for(path_str: str, rank: int) -> str:
    name = Path(path_str).name
    m = _IMG_RE.match(name)
    if m:
        return f"rank{rank}  {m.group('stem')}  p{m.group('page')}"
    return f"rank{rank}  {name}"


def build_app(offsets: Path) -> gr.Blocks:
    rows = load_rows(offsets)
    id2data = build_id_to_data(rows)
    n = len(rows)

    split_choices = [("全部", "all"), ("train", "train"), ("val", "val"), ("test", "test")]

    def filtered_indices(split: str) -> list[int]:
        if split == "all":
            return list(range(n))
        return [i for i, r in enumerate(rows) if split_of(str(r.get("id", ""))) == split]

    def render(idx_in_filter: int, split: str):
        idxs = filtered_indices(split)
        if not idxs:
            return "（此篩選沒有資料）", [], gr.update(maximum=0, value=0)
        idx_in_filter = max(0, min(int(idx_in_filter), len(idxs) - 1))
        row = rows[idxs[idx_in_filter]]
        rid = str(row.get("id", ""))

        # 右邊：colnomic top@3 圖片（PIL 物件 + caption）
        gallery = []
        for rank, p in enumerate(row.get("weakly_img_path", []) or [], start=1):
            try:
                gallery.append((Image.open(p), caption_for(p, rank)))
            except Exception as exc:  # noqa: BLE001
                gallery.append((Image.new("RGB", (400, 300), "gray"),
                                f"rank{rank}  (讀檔失敗: {exc})"))

        # 左邊：example 文字 + 中繼資料
        data_text = id2data.get(rid, "（找不到 data 文字）")
        info = "\n".join([
            f"篩選內: {idx_in_filter + 1} / {len(idxs)}    全表索引: {idxs[idx_in_filter]} / {n - 1}",
            f"id: {rid}    split: {split_of(rid)}",
            f"doc_id: {row.get('doc_id', '')}",
            f"page_number(標註): {row.get('page_number', '')}    "
            f"matched_page_no: {row.get('matched_page_no', '')}    "
            f"weakly_matched_page_no: {row.get('weakly_matched_page_no', '')}",
            f"pdf_url: {row.get('pdf_url', '')}",
            "",
            "── example (data) ──",
            data_text,
        ])
        return info, gallery, gr.update(maximum=len(idxs) - 1, value=idx_in_filter)

    with gr.Blocks(title="Colnomic Example / Image Viewer") as demo:
        gr.Markdown(
            "## Colnomic 檢索瀏覽\n"
            "左：offsets 例子的 `data` 文字與中繼資料　|　"
            "右：該例子的 colnomic（colnomic_esg_contest）檢索 top@3 頁面圖片"
            "（即 `weakly_img_path`）。"
        )
        with gr.Row():
            split_dd = gr.Dropdown(choices=split_choices, value="all", label="篩選 split")
            id_box = gr.Textbox(label="跳到 id（例如 12345）", scale=2)
            jump_btn = gr.Button("跳轉")
        with gr.Row():
            prev_btn = gr.Button("← 上一筆")
            slider = gr.Slider(0, max(0, n - 1), value=0, step=1, label="索引")
            next_btn = gr.Button("下一筆 →")
        with gr.Row():
            info = gr.Textbox(label="example（左）", lines=22, max_lines=30)
            gallery = gr.Gallery(label="colnomic top@3（右）", columns=3, height=620,
                                 object_fit="contain", preview=True)

        outputs = [info, gallery, slider]

        def _go(idx, split):
            return render(idx, split)

        def _step(idx, split, delta):
            return render(int(idx) + delta, split)

        def _jump(id_text, split):
            idxs = filtered_indices(split)
            target = next((k for k, i in enumerate(idxs)
                           if str(rows[i].get("id", "")) == id_text.strip()), None)
            if target is None:
                return ("（找不到 id="+id_text+"，或不在此 split）", [],
                        gr.update())
            return render(target, split)

        slider.change(_go, [slider, split_dd], outputs)
        split_dd.change(lambda s: render(0, s), [split_dd], outputs)
        prev_btn.click(lambda i, s: _step(i, s, -1), [slider, split_dd], outputs)
        next_btn.click(lambda i, s: _step(i, s, +1), [slider, split_dd], outputs)
        jump_btn.click(_jump, [id_box, split_dd], outputs)
        id_box.submit(_jump, [id_box, split_dd], outputs)
        demo.load(_go, [slider, split_dd], outputs)

    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", type=Path, default=DEFAULT_OFFSETS)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build_app(args.offsets).launch(server_name=args.host, server_port=args.port,
                                   share=args.share)


if __name__ == "__main__":
    main()
