#!/usr/bin/env python3
"""Modular Stage 4 verification_timeline predictor (Qwen / Codex), with same-page
content and optional page-image enrichment.

Stage 4 variant of the test_add_context modular harness. Composable modules:

    build_prompt/system_prompt.py  active prompt (system message)
    build_prompt/add_context.py    same-page-content + evidence/promise string blocks
    build_prompt/add_image.py      page image (vision-language)
    inference/qwen.py              llama-server endpoint (+ token confidence)
    inference/codex.py             Codex CLI
    inference/parser.py            output -> label (markdown / JSON / CoT tolerant)
    schemas.py                     output CSV columns + result object

Defaults to data/val_ctxhit.json and predicts EVERY input row (no gate). Pass
--gate-csv to filter by an upstream Stage 2 evidence_status column.

Same-page-content modes (--context-mode): all (default) | hit_exact_window_norm_window.

Settings come from a YAML --config (see ../configs/), built-in defaults, or CLI flags.
Precedence: explicit CLI flag > config value > built-in default.

Usage:
  python core/vlm_pred.py --config ../configs/qwen_ctx.yaml
  python core/vlm_pred.py --config ../configs/qwen_ctx.yaml --limit 20   # CLI override
  python core/vlm_pred.py --output preds/st3_qwen_ctx.csv                # pure CLI, no config
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent  # stage3/core
sys.path.insert(0, str(_HERE))

import schemas  # noqa: E402
from build_prompt.system_prompt import DEFAULT_PROMPT_PATH, load_system_prompt  # noqa: E402
from build_prompt.add_context import ContextBuilder, MODE_HIT_KINDS  # noqa: E402
from build_prompt.add_image import ImageBuilder  # noqa: E402
from build_prompt import template  # noqa: E402
from inference import codex as codex_backend  # noqa: E402
from inference import qwen as qwen_backend  # noqa: E402
from inference.parser import parse_label  # noqa: E402

_EXP_ROOT = _HERE.parents[1]  # test_add_context
_STAGE4_ROOT = _HERE.parent  # stage4
DEFAULT_DATA = _STAGE4_ROOT / "data" / "val_yes.json"


def read_data_rows(path: Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        return [dict(json.loads(line)) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list in {source}")
        return [dict(row) for row in payload]
    with source.open(newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_gate(path: Path, gate_col: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_data_rows(path):
        rid = str(row.get("id", "")).strip()
        if rid:
            out[rid] = str(row.get(gate_col, row.get("pred", row.get("label", "")))).strip()
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(schemas.OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


# Full setting space with built-in defaults. A YAML --config overrides these, and
# an explicitly-passed CLI flag overrides the config (precedence: CLI > config > default).
DEFAULTS: dict[str, Any] = {
    "data": DEFAULT_DATA,
    "output": None,
    "prompt_path": DEFAULT_PROMPT_PATH,
    "data_col": "data",
    "limit": None,
    "backend": "qwen",
    "gate_csv": None,
    "gate_col": "promise_status",
    "add_context": True,
    "context_mode": "all",
    "context_max_chars": 0,
    "add_evidence_string": False,
    "add_promise_string": False,
    "add_image": False,
    "prompt_role": "user",
    "endpoint": qwen_backend.DEFAULT_ENDPOINT,
    "model": None,
    "max_tokens": 16,
    "temperature": 0.0,
    "enable_thinking": False,
    "logprobs": False,
    "timeout": 120,
    "retries": 4,
    "concurrency": 4,
    "codex_bin": codex_backend.DEFAULT_CODEX_BIN,
}
# Keys whose values are filesystem paths (relative config paths resolve against the
# config file's directory; relative CLI paths resolve against the current dir).
PATH_KEYS = {"data", "output", "prompt_path", "gate_csv"}


def build_parser() -> argparse.ArgumentParser:
    # argument_default=SUPPRESS + no per-arg defaults => only explicitly-passed flags
    # land in the namespace, so they cleanly override the config/defaults.
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    p.add_argument("--config", type=Path, help="YAML config file with any of the settings below.")
    p.add_argument("--data", type=Path, help="Input JSON/CSV rows (default val_ctxhit.json).")
    p.add_argument("--output", type=Path, help="Output Stage 3 CSV path (config or CLI).")
    p.add_argument("--prompt-path", type=Path)
    p.add_argument("--data-col")
    p.add_argument("--limit", type=int)
    p.add_argument("--backend", choices=["qwen", "codex"])
    p.add_argument("--gate-csv", type=Path, help="Optional Stage 2 CSV/JSON gate (id + evidence_status).")
    p.add_argument("--gate-col")
    ctx = p.add_mutually_exclusive_group()
    ctx.add_argument("--add-context", dest="add_context", action="store_true", help="Inject same-page content.")
    ctx.add_argument("--no-add-context", dest="add_context", action="store_false")
    p.add_argument("--context-mode", choices=sorted(MODE_HIT_KINDS))
    p.add_argument("--context-max-chars", type=int, help="Cap same-page text chars (0 = whole page).")
    p.add_argument("--add-evidence-string", action="store_true", help="(data-use) inject evidence_string block.")
    p.add_argument("--add-promise-string", action="store_true", help="(data-use) inject promise_string block.")
    p.add_argument("--add-image", action="store_true", help="Attach same-page image (qwen VLM only).")
    p.add_argument("--prompt-role", choices=["user", "system"], help="Chat role for the single tagged message.")
    p.add_argument("--endpoint")
    p.add_argument("--model", help="Backend model id (defaults per backend).")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--logprobs", action="store_true", help="Record per-token confidence (qwen only).")
    p.add_argument("--timeout", type=int)
    p.add_argument("--retries", type=int)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--codex-bin")
    return p


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"config {path} must be a YAML mapping, got {type(raw).__name__}")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f"unknown config keys {sorted(unknown)}; valid keys: {sorted(DEFAULTS)}")
    cfg_dir = Path(path).resolve().parent
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key in PATH_KEYS and value is not None:
            value = Path(value)
            if not value.is_absolute():
                value = (cfg_dir / value).resolve()
        out[key] = value
    return out


def resolve_settings(argv: list[str] | None = None) -> SimpleNamespace:
    """Merge built-in DEFAULTS < YAML config < explicit CLI flags."""
    cli = vars(build_parser().parse_args(argv))
    config = load_config(cli.pop("config")) if cli.get("config") else {}
    merged = {**DEFAULTS, **config, **cli}
    if not merged.get("output"):
        raise SystemExit("no output path set (provide `output:` in config or --output)")
    return SimpleNamespace(**merged)


def main() -> None:
    args = resolve_settings()
    rows = read_data_rows(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]

    gate = load_gate(args.gate_csv, args.gate_col) if args.gate_csv else None
    system_prompt = load_system_prompt(args.prompt_path)

    ctx_builder = None
    if args.add_context or args.add_evidence_string or args.add_promise_string:
        ctx_builder = ContextBuilder(
            mode=args.context_mode,
            add_same_page=args.add_context,
            add_evidence_string=args.add_evidence_string,
            add_promise_string=args.add_promise_string,
            max_chars=args.context_max_chars,
        )
    empty_blocks = {"same_page_context": "", "promise_string": "", "evidence_string": ""}
    img_builder = ImageBuilder() if args.add_image else None

    model = args.model or (qwen_backend.DEFAULT_MODEL if args.backend == "qwen" else codex_backend.DEFAULT_MODEL)
    total = len(rows)
    results: list[dict[str, str] | None] = [None] * total
    counters = {"pred": 0, "err": 0, "ctx": 0, "img": 0, "done": 0}
    lock = threading.Lock()

    def process(i: int, row: dict[str, Any]) -> None:
        rid = str(row.get("id", i + 1)).strip()
        if gate is not None and gate.get(rid) != "Yes":
            res = schemas.PredictionResult(id=rid, label="N/A", source="gate_filter",
                                           reason="evidence_status_not_yes", context_hit="gated")
            results[i] = res.to_row()
            with lock:
                counters["done"] += 1
            return

        data = str(row.get(args.data_col, ""))
        hit = "data_only"
        blocks = empty_blocks
        if ctx_builder is not None:
            blocks, hit = ctx_builder.build_blocks(row, data)
        prompt = template.render(
            system_prompt=system_prompt,
            data=data,
            same_page_context=blocks["same_page_context"],
            promise_string=blocks["promise_string"],
            evidence_string=blocks["evidence_string"],
        )
        image_urls = None
        has_img = False
        if img_builder is not None:
            if args.backend == "qwen":
                ref = img_builder.data_url_for_row(row)          # base64 data: URL
            else:
                path = img_builder.image_path_for_row(row)       # local file for codex --image
                ref = str(path) if path else None
            if ref:
                image_urls = [ref]
                has_img = True

        try:
            if args.backend == "qwen":
                raw, conf = qwen_backend.classify(
                    prompt, role=args.prompt_role, endpoint=args.endpoint, model=model,
                    image_urls=image_urls, max_tokens=args.max_tokens, temperature=args.temperature,
                    enable_thinking=args.enable_thinking, logprobs=args.logprobs,
                    timeout=args.timeout, retries=args.retries,
                )
            else:
                raw, conf = codex_backend.classify(
                    prompt, model=model, codex_bin=args.codex_bin, timeout=args.timeout,
                    image_urls=image_urls,
                )
            label, reason = parse_label(raw)
            if reason == "empty_output":
                label, reason = "Excpetion:empty word", "model_output_error:empty_output"
            elif label == "N/A":
                label, reason = "except", f"model_output_error:{reason}"
        except Exception as exc:  # noqa: BLE001
            raw, conf, label, reason = "", None, "except", f"{args.backend}_error:{type(exc).__name__}"

        source = f"{args.backend}_ctx" if hit.startswith(("offset", "live")) else args.backend
        if has_img:
            source += "_img"
        res = schemas.PredictionResult(
            id=rid, label=label, raw=str(raw), source=source,
            reason=f"{reason}|ctx={hit}", context_hit=hit, token_confidence=conf,
        )
        results[i] = res.to_row()
        with lock:
            counters["pred"] += label not in ("N/A", "except") and not label.startswith("Excpetion")
            counters["err"] += label in ("N/A", "except") or label.startswith("Excpetion")
            counters["ctx"] += hit.startswith(("offset", "live"))
            counters["img"] += has_img
            counters["done"] += 1
            d = counters["done"]
        print(f"[{d}/{total}] {rid} -> {label} (ctx={hit}{' +img' if has_img else ''})", flush=True)

    # Both backends honour --concurrency. qwen fans out HTTP requests to the
    # endpoint; codex fans out independent `codex exec` subprocesses (each uses its
    # own temp file, so concurrent calls are safe).
    workers = max(1, args.concurrency)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda a: process(*a), enumerate(rows)))

    out_rows = [r for r in results if r is not None]
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    write_csv(output, out_rows)
    print(f"\nwrote {output}  rows={len(out_rows)} predicted={counters['pred']} err={counters['err']} "
          f"context_hits={counters['ctx']} image_hits={counters['img']}", flush=True)


if __name__ == "__main__":
    main()
