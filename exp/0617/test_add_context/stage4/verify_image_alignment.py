#!/usr/bin/env python3
"""Gradio UI to verify the Stage 4 page image attached to each row is correct.

Left panel  : the input example (the promise sentence + page-number metadata).
Right panel : the exact image that would be sent to the model, plus which OCR
              page candidate (matched_page_no / weakly_matched_page_no[i]) it
              resolved from and the file path.

This mirrors core/build_prompt/add_image.py exactly, so what you see on the
right is what `--add-image` attaches. The raw annotation `page_number` is shown
for comparison only; it is NOT used to pick the image.

Run:
  /workspace/esg_contest/.venv/bin/python \
    exp/integrated_stage_predictions/0617/test_add_context/stage4/verify_image_alignment.py --port 7864
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

HERE = Path(__file__).resolve().parent          # stage4/
sys.path.insert(0, str(HERE / "core"))

from build_prompt.add_image import ImageBuilder  # noqa: E402
from build_prompt.add_context import ContextBuilder, load_offsets, DEFAULT_OFFSETS  # noqa: E402

DATASETS = {
    "val_yes.100 (99 screen)": HERE / "data" / "val_yes.100.json",
    "val_yes (491 full)": HERE / "data" / "val_yes.json",
}


def short(text: str, limit: int = 60) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


class State:
    def __init__(self) -> None:
        self.img = ImageBuilder()
        self.offsets = load_offsets(DEFAULT_OFFSETS)
        self.rows: list[dict[str, Any]] = []
        self.path: Path | None = None

    def load(self, name: str) -> list[str]:
        self.path = DATASETS[name]
        self.rows = json.loads(self.path.read_text(encoding="utf-8"))
        return [f"[{i}] {r.get('id')} | {short(r.get('data'), 50)}" for i, r in enumerate(self.rows)]


STATE = State()


def _resolved_candidate(row: dict[str, Any]) -> tuple[int | None, str]:
    """Return (page_no_used, how) by replaying ImageBuilder candidate order."""
    doc_id = STATE.img.url2doc.get(row.get("pdf_url")) or STATE.img.id2doc.get(str(row.get("id")))
    if not doc_id:
        return None, "no_doc_id"
    cands = STATE.img._page_candidates(row)
    off = STATE.offsets.get(str(row.get("id", "")).strip()) or {}
    matched = off.get("matched_page_no")
    for c in cands:
        rel = STATE.img.by_page_image.get((doc_id, c))
        if rel:
            try:
                how = "matched_page_no" if int(matched) == c else "weakly_matched_page_no"
            except (TypeError, ValueError):
                how = "weakly_matched_page_no"
            return c, how
    return None, "no_image_for_candidates"


def inspect(choice: str):
    if not STATE.rows:
        return "（先載入資料）", None, ""
    idx = int(choice.split("]")[0].lstrip("["))
    row = STATE.rows[idx]
    rid = str(row.get("id"))
    off = STATE.offsets.get(rid, {})

    used_page, how = _resolved_candidate(row)
    img_path = STATE.img.image_path_for_row(row)

    left = f"""## 輸入範例  (idx {idx})

- **id**: {rid}
- **company**: {row.get('company')} ({row.get('ticker')})
- **GT verification_timeline**: `{row.get('verification_timeline')}`
- **pdf_url**: {row.get('pdf_url')}

### 承諾句 (data — 送模型的主體)
{row.get('data')}

### promise_string
{row.get('promise_string') or '（無）'}

---
### 頁碼資訊（右圖怎麼挑的）
| 欄位 | 值 |
| --- | --- |
| offsets hit_kind | `{off.get('hit_kind', '（無 offsets）')}` |
| **matched_page_no** | `{off.get('matched_page_no')}` |
| weakly_matched_page_no | `{off.get('weakly_matched_page_no')}` |
| 標註 page_number（不使用） | `{row.get('page_number')}` |
"""

    if used_page is None:
        caption = f"⚠ 找不到圖（{how}）。候選頁: {STATE.img._page_candidates(row)}"
    else:
        flag = "✅ 用 matched 頁" if how == "matched_page_no" else "↩ 退回 weakly 頁"
        same_as_raw = (str(used_page) == str(row.get("page_number")))
        caption = (f"{flag}｜實際送出頁 = **{used_page}**（來源: {how}）"
                   f"｜與標註 page_number {'相同' if same_as_raw else '不同（差頁，證明不能用 page_number）'}"
                   f"\n路徑: {img_path}")
    # Return a PIL image (not the external path) so Gradio caches it itself and
    # we sidestep the allowed_paths check for files outside the app dir.
    pil = None
    if img_path:
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            caption += f"\n⚠ 讀圖失敗: {exc}"
    return left, pil, caption


def on_load(name: str):
    labels = STATE.load(name)
    first = labels[0] if labels else None
    left, img, cap = inspect(first) if first else ("（空資料）", None, "")
    return gr.update(choices=labels, value=first), left, img, cap


def step(choice: str, delta: int):
    if not STATE.rows or not choice:
        return choice, "（先載入資料）", None, ""
    idx = max(0, min(len(STATE.rows) - 1, int(choice.split("]")[0].lstrip("[")) + delta))
    new = f"[{idx}] {STATE.rows[idx].get('id')} | {short(STATE.rows[idx].get('data'), 50)}"
    return new, *inspect(new)


def build() -> gr.Blocks:
    with gr.Blocks(title="Stage 4 image alignment 驗證") as demo:
        gr.Markdown("# Stage 4 page-image 對齊驗證\n左：input example｜右：實際會 `--add-image` 送出的圖")
        with gr.Row():
            ds = gr.Dropdown(choices=list(DATASETS), value=list(DATASETS)[0], label="資料集", scale=2)
            picker = gr.Dropdown(choices=[], label="選一列 (idx | id | 承諾句)", scale=5)
            prev = gr.Button("◀ 上一列", scale=1)
            nxt = gr.Button("下一列 ▶", scale=1)
        with gr.Row():
            with gr.Column(scale=1):
                left_md = gr.Markdown()
            with gr.Column(scale=1):
                caption = gr.Markdown()
                image = gr.Image(label="要傳送的 page image", type="pil", height=900)

        ds.change(on_load, inputs=ds, outputs=[picker, left_md, image, caption])
        picker.change(inspect, inputs=picker, outputs=[left_md, image, caption])
        prev.click(lambda c: step(c, -1), inputs=picker, outputs=[picker, left_md, image, caption])
        nxt.click(lambda c: step(c, +1), inputs=picker, outputs=[picker, left_md, image, caption])
        demo.load(on_load, inputs=ds, outputs=[picker, left_md, image, caption])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7864)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    # The page PNGs live under the repo's data/raw_reports/images, outside this
    # app dir; Gradio 6 blocks reads outside allowed paths, so whitelist them.
    repo_root = next((p for p in HERE.parents if (p / "data" / "raw_reports").is_dir()), HERE)
    allowed = [str(repo_root / "data" / "raw_reports"), str(repo_root)]
    build().launch(server_name=args.host, server_port=args.port, share=args.share,
                   allowed_paths=allowed)


if __name__ == "__main__":
    main()
