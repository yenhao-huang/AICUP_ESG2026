#!/usr/bin/env python3
"""Run Stage 4 verification_timeline prediction through Codex CLI.

Pipeline:
1. Read input rows containing `id` and `data`.
2. If `--stage1-csv` is provided, send only passed rows to Codex and emit
   `N/A` for filtered rows.
3. If `--stage1-csv` is omitted, send every row to Codex.
4. Write a Stage 4 CSV aligned to all input rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from core.service.predict.stage4 import schema


DEFAULT_PROMPT_PATH = ROOT / "configs" / "prompts" / "stage4" / "boundary_rules_v4.txt"
DEFAULT_RAW_OUTPUT_DIR = ROOT / "results" / "predict" / "stage4" / "codex" / "raw"
DEFAULT_TOKEN_USAGE_OUTPUT = ROOT / "results" / "predict" / "stage4" / "codex" / "token_usage.jsonl"
CODEX_RUN_DIR = Path("/tmp")
FLOW_NAME = "filter_pred_by_codex"
ALLOWED_TIMELINES = {
    "already",
    "within_2_years",
    "between_2_and_5_years",
    "more_than_5_years",
}
OUTPUT_COLUMNS = list(schema.OUTPUT_COLUMNS)


def read_json_or_csv(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list in {path}")
        return [dict(row) for row in payload]
    with source.open(newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def passes_stage1(value: str) -> bool:
    return value.strip().lower() == "yes"


def extract_prediction_value(row: dict[str, Any]) -> Any:
    for key in ("verification_timeline", "label", "prediction", "answer"):
        if key in row:
            return row[key]
    if "response" in row:
        response = row["response"]
        if isinstance(response, str):
            try:
                return extract_prediction_value(json.loads(response))
            except json.JSONDecodeError:
                return response
        if isinstance(response, dict):
            return extract_prediction_value(response)
    return ""


def normalize_timeline(value: Any) -> tuple[str, str]:
    if isinstance(value, dict):
        value = extract_prediction_value(value)
    text = str(value or "").strip()
    if not text:
        return "N/A", "missing_codex_prediction"
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict):
        return normalize_timeline(loaded)
    lowered = text.lower()
    aliases = [
        ("between_2_and_5_years", ("between_2_and_5_years", "between 2 and 5", "2-5", "三年", "四年", "五年", "中期")),
        ("more_than_5_years", ("more_than_5_years", "more_than_5", "more than 5", "more than five", "5年以上", "長期", "2030", "2035", "2040", "2050")),
        ("within_2_years", ("within_2_years", "within 2", "兩年", "二年", "短期", "近期")),
        ("already", ("already", "已完成", "已實施", "每年", "定期", "現行", "持續執行")),
    ]
    hits = [label for label, keys in aliases if any(key.lower() in lowered for key in keys)]
    if len(set(hits)) == 1:
        return hits[0], ""
    if text in ALLOWED_TIMELINES:
        return text, ""
    return "N/A", "invalid_codex_label"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_stage1_gate(path: Path, gate_col: str) -> dict[str, str]:
    output = {}
    for row in read_json_or_csv(str(resolve_path(path))):
        rid = str(row.get("id", "")).strip()
        if not rid:
            continue
        if gate_col in row:
            output[rid] = str(row.get(gate_col, "")).strip()
        elif gate_col == "promise_str":
            output[rid] = str(row.get("promise_status", "")).strip()
        elif gate_col == "promise_status":
            output[rid] = str(row.get("promise_str", "")).strip()
        else:
            output[rid] = ""
    return output


def safe_file_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized[:80] or "row"


def parse_jsonl_events(raw_stdout: str) -> list[dict[str, Any]]:
    events = []
    for line in raw_stdout.splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"type": "raw_line", "raw": line})
    return events


def extract_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for event in events:
        for key in ("usage", "token_usage", "tokens"):
            value = event.get(key)
            if isinstance(value, dict):
                usage.update(value)
        if event.get("type") in {"token_usage", "usage"}:
            usage.update({k: v for k, v in event.items() if k != "type"})
    input_tokens = usage.get("input_tokens") or usage.get("inputTokens")
    output_tokens = usage.get("output_tokens") or usage.get("outputTokens")
    total_tokens = usage.get("total_tokens") or usage.get("totalTokens")
    if total_tokens is None and isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": usage.get("total_cost_usd") or usage.get("costUSD"),
        "raw_usage": usage,
    }


def build_prompt(system_prompt: str, data: str) -> str:
    return (
        f"{system_prompt}\n\n"
        "只能根據下方 DATA 判斷 verification_timeline；不要使用 evidence_string、"
        "promise_string、標註欄位或任何外部資料。\n\n"
        f"DATA:\n{data}\n\n"
        "只輸出其中一個 label：already、within_2_years、"
        "between_2_and_5_years、more_than_5_years。不要解釋，不要使用 markdown。"
    )


def predict_one(codex_bin: str, model: str, prompt: str, timeout: int) -> tuple[str, dict[str, Any]]:
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
    metadata = {
        "command": shell_command,
        "codex_command": codex_command,
        "returncode": result.returncode,
        "raw_stdout": result.stdout,
        "raw_stderr": result.stderr,
        "events": parse_jsonl_events(result.stdout),
    }
    if result.returncode != 0:
        raise RuntimeError(
            "Codex CLI failed "
            f"(exit={result.returncode}): {(result.stderr or result.stdout).strip()}"
        )
    return last_message or result.stdout.strip(), metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=None, help="Single question text (skips --data and --stage1-csv).")
    parser.add_argument("--data", type=Path, default=None, help="Input JSON/CSV rows containing id and data.")
    parser.add_argument("--stage1-csv", type=Path, default=None, help="Stage 1 CSV/JSON with id and promise_str.")
    parser.add_argument("--output", type=Path, default=None, help="Output Stage 4 CSV path.")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--stage1-gate-col", default="promise_str")
    parser.add_argument("--data-col", default="data")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent Codex predictions.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--raw-output-dir", type=Path, default=DEFAULT_RAW_OUTPUT_DIR)
    parser.add_argument("--token-usage-output", type=Path, default=DEFAULT_TOKEN_USAGE_OUTPUT)
    parser.add_argument("--start-from", type=int, default=1, metavar="N",
                        help="Resume from this 1-based row index; earlier rows are recovered from existing raw JSONs.")
    return parser


def run_single_question(args: argparse.Namespace) -> None:
    prompt_path = resolve_path(args.prompt_path)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    prompt = build_prompt(system_prompt, args.question)
    raw, metadata = predict_one(args.codex_bin, args.model, prompt, args.timeout)
    label, error = normalize_timeline(raw)
    print(label)
    if error:
        print(f"[warning] {error}", flush=True)


def output_row(
    rid: str,
    label: str,
    promise_str: str,
    filtered: str,
    raw_timeline: str,
    error: str,
) -> dict[str, str]:
    return {
        "id": rid,
        "verification_timeline": label,
        "stage4_flow": FLOW_NAME,
        "stage1_promise_str": promise_str,
        "stage4_filtered": filtered,
        "stage4_raw_timeline": raw_timeline,
        "stage4_postprocess_rule": "",
        "stage4_error": error,
    }


def predict_row(
    index: int,
    row_count: int,
    row: dict[str, Any],
    rid: str,
    promise_str: str,
    args: argparse.Namespace,
    system_prompt: str,
    prompt_path: Path,
    raw_output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    raw = ""
    metadata: dict[str, Any] = {}
    error = ""
    try:
        prompt = build_prompt(system_prompt, str(row.get(args.data_col, "")))
        raw, metadata = predict_one(args.codex_bin, args.model, prompt, args.timeout)
        label, error = normalize_timeline(raw)
    except Exception as exc:  # Keep row alignment even when Codex fails.
        label = "N/A"
        error = f"codex_error:{type(exc).__name__}"
        metadata = {"error": str(exc)}

    raw_output_path = raw_output_dir / f"{index:04d}_{safe_file_part(rid)}.json"
    raw_record = {
        "id": rid,
        "raw_prediction": raw,
        "metadata": metadata,
        "run_id": run_id,
        "prompt_path": str(prompt_path),
    }
    usage_row = {
        "index": index,
        "id": rid,
        "model": args.model,
        "prompt_path": str(prompt_path),
        "raw_output_path": str(raw_output_path),
        "run_id": run_id,
        **extract_usage(metadata.get("events", [])),
    }
    return {
        "index": index,
        "row_count": row_count,
        "rid": rid,
        "label": label,
        "error": error,
        "raw_output_path": raw_output_path,
        "raw_record": raw_record,
        "usage_row": usage_row,
        "output_row": output_row(rid, label, promise_str, "no", label, error),
    }


def main() -> None:
    args = build_parser().parse_args()

    if args.question is not None:
        run_single_question(args)
        return

    if args.data is None or args.output is None:
        build_parser().error("--data and --output are required when not using single-question mode.")

    rows = read_json_or_csv(str(resolve_path(args.data)))
    if args.limit is not None:
        rows = rows[: args.limit]

    prompt_path = resolve_path(args.prompt_path)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    stage1 = load_stage1_gate(args.stage1_csv, args.stage1_gate_col) if args.stage1_csv else {}
    raw_output_dir = resolve_path(args.raw_output_dir)
    token_usage_output = resolve_path(args.token_usage_output)
    start_from = max(1, args.start_from)
    if start_from == 1 and token_usage_output.exists():
        token_usage_output.unlink()
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"stage4_codex_{args.model}_{prompt_path.stem}"
    workers = max(1, args.workers)

    output_rows_by_index: dict[int, dict[str, str]] = {}
    prediction_tasks: list[tuple[int, dict[str, Any], str, str]] = []
    predicted_count = 0
    error_count = 0
    for index, row in enumerate(rows, start=1):
        rid = str(row.get("id", index)).strip()
        promise_str = stage1.get(rid, "")
        if stage1 and not passes_stage1(promise_str):
            output_rows_by_index[index] = output_row(rid, "N/A", promise_str, "yes", "N/A", "")
            if index >= start_from:
                print(f"[{index}/{len(rows)}] {rid} -> N/A (filtered)", flush=True)
            continue

        # Recover already-processed rows from existing raw JSON files.
        if index < start_from:
            raw_output_path = raw_output_dir / f"{index:04d}_{safe_file_part(rid)}.json"
            recovered_label = "N/A"
            recovered_error = "error_skipped"
            try:
                if raw_output_path.exists() and raw_output_path.stat().st_size > 0:
                    raw_data = json.loads(raw_output_path.read_text(encoding="utf-8"))
                    recovered_label, recovered_error = normalize_timeline(raw_data.get("raw_prediction", ""))
            except Exception:
                pass
            output_rows_by_index[index] = output_row(
                rid,
                recovered_label,
                promise_str,
                "no",
                recovered_label,
                recovered_error,
            )
            if recovered_error:
                error_count += 1
            else:
                predicted_count += 1
            continue

        prediction_tasks.append((index, row, rid, promise_str))

    if workers == 1:
        for index, row, rid, promise_str in prediction_tasks:
            result = predict_row(
                index, len(rows), row, rid, promise_str, args, system_prompt, prompt_path, raw_output_dir, run_id
            )
            write_json(result["raw_output_path"], result["raw_record"])
            append_jsonl(token_usage_output, result["usage_row"])
            output_rows_by_index[index] = result["output_row"]
            if result["error"]:
                error_count += 1
            else:
                predicted_count += 1
            print(f"[{index}/{len(rows)}] {rid} -> {result['label']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    predict_row,
                    index,
                    len(rows),
                    row,
                    rid,
                    promise_str,
                    args,
                    system_prompt,
                    prompt_path,
                    raw_output_dir,
                    run_id,
                )
                for index, row, rid, promise_str in prediction_tasks
            ]
            for future in as_completed(futures):
                result = future.result()
                write_json(result["raw_output_path"], result["raw_record"])
                append_jsonl(token_usage_output, result["usage_row"])
                output_rows_by_index[result["index"]] = result["output_row"]
                if result["error"]:
                    error_count += 1
                else:
                    predicted_count += 1
                print(f"[{result['index']}/{len(rows)}] {result['rid']} -> {result['label']}", flush=True)

    output = resolve_path(args.output)
    output_rows = [output_rows_by_index[index] for index in range(1, len(rows) + 1)]
    write_csv(str(output), output_rows)
    summary = {
        "output": str(output),
        "rows": len(output_rows),
        "filtered_rows": sum(1 for row in output_rows if row["stage4_filtered"] == "yes"),
        "predicted_rows": predicted_count,
        "errors": error_count,
        "flow": FLOW_NAME,
        "model": args.model,
        "prompt_path": str(prompt_path),
        "token_usage_output": str(token_usage_output),
        "raw_output_dir": str(raw_output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
