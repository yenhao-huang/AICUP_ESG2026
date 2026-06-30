#!/usr/bin/env python3
"""fix_correct_image_path — 輸入文字，透過 RAG Search API 取回相關頁面圖片的 path。

服務：192.168.1.78:8766（FastAPI "RAG Search API"，collection=amd_pdf）
    POST /api/search  body = {query, top_k, config_path}
    每個 hit 回傳：rank / score / page_id / document_id / page_number /
                   image_base64 / extra{pdf_path, filename, text, img_text}

「相關圖片的 path」在這裡有兩種具體意義，本工具都會給：
  - source : ``<pdf_path>#page=<page_number>`` —— 該頁圖片在原始 PDF 內的位置
             （collection 是 per-page 影像，pdf_path + page_number 即可定位）。
  - saved  : 加 ``--save-images DIR`` 時，把回傳的 image_base64 落地成 PNG/JPG，
             回報實際磁碟路徑，方便直接開檔肉眼確認。

用法：
    P=/workspace/esg_contest/.venv/bin/python
    $P exp/integrated_stage_predictions/0618/fix_correct_image_path.py "碳中和 減碳目標"
    $P .../fix_correct_image_path.py "水資源管理" --top-k 5 \
        --save-images exp/integrated_stage_predictions/0618/fix_correct_image_path_out
    echo "供應商節電輔導" | $P .../fix_correct_image_path.py -      # 從 stdin 讀 query
    $P .../fix_correct_image_path.py "碳中和" --json                # 原始 JSON（去掉 base64）

備註：本腳本是使用者授權的 image-path 探針，超出 CLAUDE.md data-only 規則，
      不可當作 stage 的 data-only runtime 路徑升級。大量落地圖片請存到 /data。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://192.168.1.78:8766"
DEFAULT_CONFIG = "configs/retriever/search.yml"


def search(url: str, query: str, top_k: int, config_path: str, timeout: float) -> dict:
    """呼叫 POST /api/search 並回傳解析後的 dict。"""
    payload = json.dumps(
        {"query": query, "top_k": top_k, "config_path": config_path}
    ).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/api/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    if "detail" in data and "results" not in data:
        raise RuntimeError(f"search API error: {data['detail']}")
    return data


def _img_ext(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:2] == b"\xff\xd8":
        return "jpg"
    return "bin"


def save_image(b64: str, out_dir: Path, stem: str) -> Path:
    """把 base64 影像寫到 out_dir/<stem>.<ext>，回傳路徑。"""
    raw = base64.b64decode(b64)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.{_img_ext(raw[:16])}"
    path.write_bytes(raw)
    return path


def source_image_path(result: dict) -> str:
    """把一個 hit 表示成可定位的圖片 path：<pdf_path>#page=<n>。"""
    extra = result.get("extra") or {}
    pdf_path = extra.get("pdf_path") or extra.get("filename") or "(unknown)"
    page = result.get("page_number")
    return f"{pdf_path}#page={page}" if page is not None else pdf_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("query", help="查詢文字；用 '-' 表示從 stdin 讀整段文字")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="search API 的 config_path")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument(
        "--save-images",
        metavar="DIR",
        help="把每個 hit 的 image_base64 落地成圖檔到 DIR，並回報磁碟路徑",
    )
    ap.add_argument(
        "--snippet",
        type=int,
        default=140,
        help="每筆顯示的 extra.text 字數（0 = 不顯示）",
    )
    ap.add_argument("--json", action="store_true", help="輸出原始 JSON（移除 base64）")
    args = ap.parse_args()

    query = sys.stdin.read().strip() if args.query == "-" else args.query
    if not query:
        ap.error("query 為空")

    try:
        data = search(args.url, query, args.top_k, args.config, args.timeout)
    except Exception as exc:  # noqa: BLE001 - CLI 直接回報
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    results = data.get("results", [])
    out_dir = Path(args.save_images) if args.save_images else None

    # 落地圖片並把路徑掛回每筆 result。
    paths: list[str] = []
    for r in results:
        saved = None
        if out_dir is not None and r.get("image_base64"):
            extra = r.get("extra") or {}
            stem_base = Path(extra.get("filename") or "page").stem
            stem = f"rank{r.get('rank', 0):02d}_{stem_base}_p{r.get('page_number')}"
            saved = str(save_image(r["image_base64"], out_dir, stem))
        r["_saved_image_path"] = saved
        paths.append(saved or source_image_path(r))

    if args.json:
        for r in results:
            r.pop("image_base64", None)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    # 人類可讀輸出。
    print(f"query     : {query}")
    print(f"collection: {data.get('collection')}   hits: {len(results)}")
    print("-" * 72)
    for r in results:
        print(f"[rank {r.get('rank')}] score={r.get('score'):.4f}")
        print(f"  image_path : {source_image_path(r)}")
        print(f"  page_id    : {r.get('page_id')}")
        print(f"  document_id: {r.get('document_id')}")
        if r.get("_saved_image_path"):
            print(f"  saved      : {r['_saved_image_path']}")
        if args.snippet:
            text = ((r.get("extra") or {}).get("text") or "").replace("\n", " ").strip()
            if text:
                print(f"  text       : {text[:args.snippet]}")
        print()

    # 最後印出純路徑清單，方便複製。
    print("== image paths ==")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
