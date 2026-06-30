#!/usr/bin/env python3
"""build_weakly_img_path — 為 offsets.jsonl 每列用 `data` 文字做 RAG search，
取 top@K 頁面圖片、落地存檔，並加上 `weakly_img_path` 欄（top@K 圖片路徑清單）。

流程（每列）：
    offsets.id --join--> 原始 json 的 `data` 文字（唯一允許的 raw 輸入）
        --POST /api/search (top_k=K)--> K 個 hit（含 image_base64 / page_id / pdf）
        --存圖(以 page_id 去重)--> weakly_img_path = [path_rank1, ..., path_rankK]

「weakly」= 用語意檢索（非 OCR 精準頁碼）找到的弱對齊頁圖，沿用 0617 image-path 探針脈絡。

資料使用：查詢只用 `data` 欄，符合 CLAUDE.md data-only；落地圖片是使用者授權的
enrichment 探針，非可升級的 data-only runtime 路徑。圖片一律存 /data（/workspace 近滿）。

特性：
  - 去重：相同 page_id 的圖只存一次（多列共用）。
  - 可續跑：已填好且檔案都在的列會跳過（--resume，預設開）。
  - 併發：--concurrency 個 thread 同時打 search。
  - checkpoint：每 --checkpoint 列把 offsets 寫一次，長跑可中斷續跑。
  - --limit N 只跑前 N 列（小批量實測用）。

用法：
    P=/workspace/esg_contest/.venv/bin/python
    # 小批量實測（不動原檔，輸出到別處）
    $P exp/integrated_stage_predictions/0618/build_weakly_img_path.py \
        --limit 20 --out /tmp/off_test.jsonl
    # 全量、就地加欄（自動備份 .bak_weakly）
    $P exp/integrated_stage_predictions/0618/build_weakly_img_path.py --concurrency 6
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OFFSETS = REPO_ROOT / "exp/integrated_stage_predictions/0617/test_add_context/data/offsets.jsonl"
DEFAULT_IMG_DIR = Path("/data/integrated_stage_predictions_0618/weakly_img")
DEFAULT_URL = "http://192.168.1.78:8766"
DEFAULT_CONFIG = "configs/retriever/search.yml"


# ---------------------------------------------------------------- I/O helpers
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def build_id_to_data(offsets: list[dict]) -> dict[str, str]:
    """從 offsets 用到的所有 source_file 載入 id -> data 文字。"""
    id2data: dict[str, str] = {}
    for src in sorted({r.get("source_file", "") for r in offsets if r.get("source_file")}):
        p = Path(src)
        if not p.exists():  # source_file 是絕對路徑；保險再試 repo 相對
            p = REPO_ROOT / Path(src).name
        if not p.exists():
            print(f"[warn] source_file 找不到: {src}", file=sys.stderr)
            continue
        for rec in json.load(p.open(encoding="utf-8")):
            id2data[str(rec["id"])] = rec.get("data", "") or ""
    return id2data


# ---------------------------------------------------------------- search + save
def search(url: str, query: str, top_k: int, config_path: str, timeout: float) -> list[dict]:
    payload = json.dumps({"query": query, "top_k": top_k, "config_path": config_path}).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/api/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if "results" not in data:
        raise RuntimeError(f"search API error: {data.get('detail')}")
    return data["results"]


def _ext_of(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:2] == b"\xff\xd8":
        return "jpg"
    return "bin"


class ImageSaver:
    """以 page_id 去重，把 image_base64 落地到 img_dir。thread-safe。"""

    def __init__(self, img_dir: Path, to_png: bool):
        self.img_dir = img_dir
        self.to_png = to_png
        self.lock = threading.Lock()
        self.cache: dict[str, str] = {}
        img_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: dict) -> str | None:
        page_id = str(result.get("page_id") or "").strip()
        b64 = result.get("image_base64")
        if not page_id or not b64:
            return None
        with self.lock:
            if page_id in self.cache:
                return self.cache[page_id]
        extra = result.get("extra") or {}
        stem = Path(extra.get("filename") or "page").stem
        page = result.get("page_number")
        raw = base64.b64decode(b64)
        ext = "png" if self.to_png else _ext_of(raw[:16])
        fname = f"{stem}_p{page}__{page_id}.{ext}"
        path = self.img_dir / fname
        if not path.exists():
            if self.to_png and _ext_of(raw[:16]) != "png":
                from io import BytesIO

                from PIL import Image

                Image.open(BytesIO(raw)).convert("RGB").save(path, format="PNG")
            else:
                path.write_bytes(raw)
        sp = str(path)
        with self.lock:
            self.cache[page_id] = sp
        return sp


# ---------------------------------------------------------------- per-row work
def needs_work(row: dict, k: int) -> bool:
    paths = row.get("weakly_img_path")
    if not isinstance(paths, list) or not paths:
        return True
    return not all(isinstance(p, str) and Path(p).exists() for p in paths)


def process_row(row, id2data, saver, args):
    """回傳 (weakly_img_path:list, error:str|None)。會就地寫進 row。"""
    query = (id2data.get(str(row.get("id"))) or "").strip()
    if not query:
        return [], "no data text"
    if args.max_query_chars and len(query) > args.max_query_chars:
        query = query[: args.max_query_chars]
    last_err = None
    for attempt in range(args.retries + 1):
        try:
            results = search(args.url, query, args.top_k, args.config, args.timeout)
            paths = []
            for r in results:
                sp = saver.save(r)
                if sp:
                    paths.append(sp)
            row["weakly_img_path"] = paths
            return paths, None
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(min(2 ** attempt, 8))
    return [], last_err


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offsets", type=Path, default=DEFAULT_OFFSETS)
    ap.add_argument("--out", type=Path, help="輸出路徑；省略=就地寫回 --offsets（先備份 .bak_weakly）")
    ap.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 列（0=全部）")
    ap.add_argument("--checkpoint", type=int, default=100, help="每 N 列寫一次輸出")
    ap.add_argument("--max-query-chars", type=int, default=2000)
    ap.add_argument("--png", action="store_true", help="轉存 PNG（預設存原生 JPEG，省空間）")
    ap.add_argument("--no-resume", action="store_true", help="不跳過已完成的列")
    args = ap.parse_args()

    rows = read_jsonl(args.offsets)
    id2data = build_id_to_data(rows)
    print(f"offsets rows: {len(rows)}   id2data: {len(id2data)}")

    out_path = args.out or args.offsets
    if args.out is None:
        bak = args.offsets.with_suffix(args.offsets.suffix + ".bak_weakly")
        if not bak.exists():
            shutil.copy2(args.offsets, bak)
            print(f"backup written: {bak}")

    saver = ImageSaver(args.img_dir, to_png=args.png)

    targets = list(enumerate(rows))
    if args.limit:
        targets = targets[: args.limit]
    todo = [(i, r) for i, r in targets if args.no_resume or needs_work(r, args.top_k)]
    print(f"to process: {len(todo)} / {len(targets)}  (img-dir={args.img_dir}, png={args.png})")

    done = err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(process_row, r, id2data, saver, args): i for i, r in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            paths, e = fut.result()
            if e:
                err += 1
                if err <= 10:
                    print(f"[err] row idx={futs[fut]} id={rows[futs[fut]].get('id')}: {e}", file=sys.stderr)
            else:
                done += 1
            if n % args.checkpoint == 0:
                write_jsonl(out_path, rows)
                rate = n / max(1e-6, time.time() - t0)
                print(f"  {n}/{len(todo)}  ok={done} err={err}  uniq_img={len(saver.cache)}  "
                      f"{rate:.1f} rows/s  ckpt-> {out_path.name}")

    write_jsonl(out_path, rows)
    dt = time.time() - t0
    print(f"DONE  processed={len(todo)} ok={done} err={err}  uniq_img={len(saver.cache)}  "
          f"{dt:.0f}s  -> {out_path}")
    # 落地容量摘要
    total = sum(Path(p).stat().st_size for p in saver.cache.values() if Path(p).exists())
    print(f"saved {len(saver.cache)} unique images, {total/1e6:.1f} MB in {args.img_dir}")


if __name__ == "__main__":
    main()
