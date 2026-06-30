#!/usr/bin/env python3
"""Generate a per-page abstract for every row's matched report page (Codex CLI).

Codex-backed variant of `build_page_abstract.py`. Page alignment is identical:
each row (id) is joined to `offsets.jsonl`, preferring `matched_page_no` and
falling back to `weakly_matched_page_no[0]`; the matched OCR page text comes
from the page table (`text_clean`, falling back to raw `text`). The only
difference is the summarizer: instead of the local Qwen HTTP server, each unique
page is summarized by calling the Codex CLI (`codex exec --json ...`). Pages are
deduplicated, so each unique (doc_id, page_no) is summarized once.

Outputs (under --out-dir):
  page_abstracts.jsonl        one record per unique matched page + abstract
  val_with_page_abstract.jsonl each input row + matched page_no + abstract

Usage:
  EXP=exp/integrated_stage_predictions/0617
  .venv/bin/python $EXP/add_page_abstract/build_page_abstract_codex.py \
      --data data/raw_data/vpesg4k_test_2000.json \
      --out-dir $EXP/add_page_abstract/test2000_codex \
      --concurrency 8
  # smoke first:
  ... --data $EXP/test_add_context/data/val.100.jsonl --limit 2
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from core.human.predict.stage4.build_page_context import (  # noqa: E402
    DEFAULT_DOC_TABLE,
    DEFAULT_PAGE_TABLE,
    DEFAULT_PREFIX_CHARS,
    read_json_list,
    read_jsonl,
)

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_PROMPT = Path(__file__).resolve().parent / "prompts" / "page_abstract.txt"
CODEX_RUN_DIR = Path("/tmp")


def load_rows(path: Path) -> list[dict]:
    path = path if path.is_absolute() else ROOT / path
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    return read_json_list(path)


def load_offsets(path: Path) -> dict[str, dict]:
    """id -> precomputed alignment record from offsets.jsonl."""
    path = path if path.is_absolute() else ROOT / path
    out: dict[str, dict] = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[str(rec.get("id"))] = rec
    return out


def load_cache(path: Path) -> dict[tuple[str, int], str]:
    """Load per-page abstracts already computed in a prior run (resume)."""
    out: dict[tuple[str, int], str] = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        doc, pno = r.get("doc_id"), r.get("page_no")
        if doc is None or pno is None:
            continue
        try:
            out[(doc, int(pno))] = r.get("abstract", "")
        except (TypeError, ValueError):
            continue
    return out


def pick_page_no(rec: dict) -> int | None:
    """Prefer matched_page_no; else fall back to weakly_matched_page_no[0]."""
    pn = rec.get("matched_page_no")
    if pn is None:
        weak = rec.get("weakly_matched_page_no") or []
        pn = weak[0] if weak else None
    try:
        return int(pn) if pn is not None else None
    except (TypeError, ValueError):
        return None


def build_page_meta(page_rows: list[dict]) -> dict[tuple[str, int], dict]:
    """(doc_id, page_no) -> {page_id, image_path, text_clean|text}."""
    meta: dict[tuple[str, int], dict] = {}
    for r in page_rows:
        doc = r.get("doc_id")
        try:
            pno = int(r.get("page_no"))
        except (TypeError, ValueError):
            continue
        if not doc:
            continue
        text = r.get("text_clean") or r.get("text") or ""
        meta[(doc, pno)] = {
            "page_id": r.get("page_id"),
            "image_path": r.get("image_path"),
            "text": text,
        }
    return meta


def codex_once(prompt: str, *, codex_bin: str, model: str, timeout: int) -> str:
    """Run one `codex exec` call and return its last message text."""
    with tempfile.NamedTemporaryFile("r", encoding="utf-8", delete=False) as tmp:
        last_message_path = Path(tmp.name)
    codex_command = [
        codex_bin,
        "exec",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--model",
        model,
        "--skip-git-repo-check",
        "--output-last-message",
        str(last_message_path),
        prompt,
    ]
    shell_command = (
        f"cd {shlex.quote(str(CODEX_RUN_DIR))} && "
        + " ".join(shlex.quote(part) for part in codex_command)
    )
    result = subprocess.run(
        shell_command,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    try:
        last_message = last_message_path.read_text(encoding="utf-8").strip()
    finally:
        last_message_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(
            "codex exec failed "
            f"(exit={result.returncode}): {(result.stderr or result.stdout).strip()}"
        )
    return last_message or result.stdout.strip()


def summarize(text: str, system: str, *, codex_bin: str, model: str,
              timeout: int, retries: int) -> str:
    prompt = f"{system}\n\n頁面 OCR 全文：\n{text}"
    last = None
    for attempt in range(retries + 1):
        try:
            return codex_once(prompt, codex_bin=codex_bin, model=model, timeout=timeout)
        except (subprocess.TimeoutExpired, RuntimeError, OSError) as e:
            last = e
            time.sleep(min(20, 2 * (attempt + 1)))
    raise RuntimeError(f"summarize failed after {retries + 1} tries: {last}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--doc-table", type=Path, default=DEFAULT_DOC_TABLE)
    ap.add_argument("--page-table", type=Path, default=DEFAULT_PAGE_TABLE)
    ap.add_argument("--offsets", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "test_add_context" / "data" / "offsets.jsonl",
                    help="Precomputed id->page alignment "
                         "(matched_page_no / weakly_matched_page_no).")
    ap.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT)
    ap.add_argument("--prefix-chars", type=int, default=DEFAULT_PREFIX_CHARS)
    ap.add_argument("--max-page-chars", type=int, default=6000,
                    help="Cap page OCR chars sent to the model (0 = no cap).")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--cache", type=Path, default=None,
                    help="Resume cache JSONL (default: <out-dir>/page_abstracts.cache.jsonl). "
                         "Completed pages are appended live; reruns skip cached pages.")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N rows (smoke).")
    args = ap.parse_args()

    system = (args.prompt_path if args.prompt_path.is_absolute()
              else ROOT / args.prompt_path).read_text(encoding="utf-8").strip()
    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]

    page_rows = read_jsonl(args.page_table)
    page_meta = build_page_meta(page_rows)         # (doc, page_no) -> text_clean + ids
    offsets = load_offsets(args.offsets)           # id -> precomputed alignment record

    # Resolve each row -> matched (doc_id, page_no) from offsets.jsonl:
    # prefer matched_page_no, else weakly_matched_page_no[0]. Page text comes
    # from page_meta, which uses text_clean (falling back to raw text).
    row_page: dict[str, tuple[str, int] | None] = {}
    unique: dict[tuple[str, int], dict] = {}
    miss = 0
    for r in rows:
        rid = str(r.get("id"))
        rec = offsets.get(rid)
        key = None
        if rec:
            doc = rec.get("doc_id")
            pn = pick_page_no(rec)
            if doc and pn is not None:
                key = (doc, pn)
        row_page[rid] = key
        if key is None:
            miss += 1
        elif key not in unique:
            m = page_meta.get(key, {})
            unique[key] = {
                "doc_id": key[0], "page_no": key[1],
                "page_id": m.get("page_id"), "image_path": m.get("image_path"),
                "company": r.get("company"), "ticker": r.get("ticker"),
                "text": m.get("text", ""),
            }

    print(f"rows={len(rows)}  matched={len(rows) - miss}  miss={miss}  "
          f"unique_pages={len(unique)}  -> {len(unique)} Codex calls", flush=True)

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume: pull already-computed pages from the cache, summarize only the rest.
    cache_path = (args.cache if args.cache is not None
                  else out_dir / "page_abstracts.cache.jsonl")
    cache_path = cache_path if cache_path.is_absolute() else ROOT / cache_path
    cached = load_cache(cache_path)
    abstracts: dict[tuple[str, int], str] = {k: cached[k] for k in unique if k in cached}
    todo = [k for k in unique if k not in abstracts]
    print(f"resume: {len(abstracts)} cached, {len(todo)} to summarize  "
          f"(cache={cache_path})", flush=True)

    # Summarize each unique page concurrently (each call is one Codex process).
    def work(key: tuple[str, int]) -> tuple[tuple[str, int], str]:
        text = unique[key]["text"] or ""
        if args.max_page_chars and len(text) > args.max_page_chars:
            text = text[: args.max_page_chars]
        if not text.strip():
            return key, ""
        ab = summarize(text, system, codex_bin=args.codex_bin, model=args.model,
                       timeout=args.timeout, retries=args.retries)
        return key, ab

    cache_lock = threading.Lock()
    done = len(abstracts)
    t0 = time.time()
    if todo:
        with cache_path.open("a", encoding="utf-8") as cache_fh, \
                ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(work, k): k for k in todo}
            for fut in as_completed(futs):
                key, ab = fut.result()
                abstracts[key] = ab
                with cache_lock:  # append each finished page so a kill never wastes work
                    cache_fh.write(json.dumps(
                        {"doc_id": key[0], "page_no": key[1], "abstract": ab},
                        ensure_ascii=False) + "\n")
                    cache_fh.flush()
                done += 1
                if done % 25 == 0 or done == len(unique):
                    print(f"  [{done}/{len(unique)}] {time.time() - t0:.0f}s", flush=True)

    pages_path = out_dir / "page_abstracts.jsonl"
    with pages_path.open("w", encoding="utf-8") as f:
        for key, info in unique.items():
            f.write(json.dumps({
                "doc_id": info["doc_id"], "page_no": info["page_no"],
                "page_id": info["page_id"], "image_path": info["image_path"],
                "company": info["company"], "ticker": info["ticker"],
                "n_chars": len(info["text"] or ""),
                "abstract": abstracts.get(key, ""),
            }, ensure_ascii=False) + "\n")

    rows_path = out_dir / "val_with_page_abstract.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for r in rows:
            rid = str(r.get("id"))
            key = row_page.get(rid)
            f.write(json.dumps({
                "id": rid,
                "data": r.get("data"),
                "company": r.get("company"),
                "page_number": r.get("page_number"),
                "matched_doc_id": key[0] if key else None,
                "matched_page_no": key[1] if key else None,
                "page_abstract": abstracts.get(key, "") if key else "",
            }, ensure_ascii=False) + "\n")

    empties = sum(1 for v in abstracts.values() if not v)
    print(f"unique pages: {len(unique)}  empty abstracts: {empties}")
    print(f"page abstracts -> {pages_path}")
    print(f"rows joined    -> {rows_path}")


if __name__ == "__main__":
    main()
