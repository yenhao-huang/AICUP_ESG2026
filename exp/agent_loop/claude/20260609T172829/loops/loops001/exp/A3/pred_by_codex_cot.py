#!/usr/bin/env python3
"""Stage 4 prediction script for CoT prompt variants (V3, V6).

Modified from core/human/predict/stage4/pred_by_codex.py to:
1. NOT append the 'only output label' instruction (allows CoT output).
2. Extract label after 'output:' when CoT format is detected.
3. Skip raw file output to save disk space.
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
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/esg_contest")
sys.path.insert(0, str(ROOT))

from core.human.predict.stage4 import schema


DEFAULT_PROMPT_PATH = ROOT / "configs" / "prompt" / "stage4" / "codex" / "few_shot_cot_v3.txt"
CODEX_RUN_DIR = Path("/tmp")
FLOW_NAME = "filter_pred_by_codex_cot"
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


def extract_cot_label(text: str) -> str | None:
    """Extract label after CoT marker in CoT format output."""
    match = re.search(r'輸出[:：]\s*(\S+)', text)
    if match:
        candidate = match.group(1).strip().rstrip('。.,;;；')
        if candidate in ALLOWED_TIMELINES:
            return candidate
    return None


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

    # Try CoT extraction first (for format)
    cot_label = extract_cot_label(text)
    if cot_label:
        return cot_label, ""

    # Fall back to exact match
    if text in ALLOWED_TIMELINES:
        return text, ""

    # Fall back to alias matching on full lowered text
    lowered = text.lower()
    aliases = [
        ("between_2_and_5_years", ("between_2_and_5_years", "between 2 and 5", "2-5", "three", "four", "five")),
        ("more_than_5_years", ("more_than_5_years", "more_than_5", "more than 5", "more than five")),
        ("within_2_years", ("within_2_years", "within 2")),
        ("already", ("already",)),
    ]
    hits = [label for label, keys in aliases if any(key.lower() in lowered for key in keys)]
    if len(set(hits)) == 1:
        return hits[0], ""
    return "N/A", "invalid_codex_label"


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


def build_prompt_cot(system_prompt: str, data: str) -> str:
    """Build prompt for CoT variants - does NOT append 'only output label' instruction."""
    return (
        f"{system_prompt}\n\n"
        "只能根據下方 DATA 判斷 verification_timeline；"
        "不要使用 evidence_string、"
        "promise_string、標註欄位或任何外部資料。\n\n"
        f"DATA:\n{data}"
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
        "returncode": result.returncode,
        "raw_stderr": result.stderr[:500] if result.stderr else "",
    }
    if result.returncode != 0:
        raise RuntimeError(
            "Codex CLI failed "
            f"(exit={result.returncode}): {(result.stderr or result.stdout).strip()}"
        )
    return last_message or result.stdout.strip(), metadata


def load_existing_output(path: Path) -> dict[str, dict[str, str]]:
    """Load existing output CSV rows keyed by id, for resuming."""
    if not path.exists():
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return {row["id"]: row for row in csv.DictReader(f)}
    except Exception:
        return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--stage1-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--stage1-gate-col", default="promise_str")
    parser.add_argument("--data-col", default="data")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--start-from", type=int, default=1, metavar="N",
                        help="Resume from this 1-based row index; earlier rows recovered from existing output CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.data is None or args.stage1_csv is None or args.output is None:
        build_parser().error("--data, --stage1-csv, and --output are required.")

    rows = read_json_or_csv(str(resolve_path(args.data)))
    if args.limit is not None:
        rows = rows[: args.limit]

    prompt_path = resolve_path(args.prompt_path)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    stage1 = load_stage1_gate(args.stage1_csv, args.stage1_gate_col)
    run_id = f"stage4_codex_cot_{args.model}_{prompt_path.stem}"

    start_from = max(1, args.start_from)
    output_path = resolve_path(args.output)
    existing = load_existing_output(output_path) if start_from > 1 else {}

    output_rows: list[dict[str, str]] = []
    predicted_count = 0
    error_count = 0
    for index, row in enumerate(rows, start=1):
        rid = str(row.get("id", index)).strip()

        if index < start_from:
            if rid in existing:
                recovered = existing[rid]
                output_rows.append(recovered)
                if recovered.get("stage4_filtered") != "yes":
                    predicted_count += 1
            else:
                promise_str = stage1.get(rid, "")
                output_rows.append({
                    "id": rid,
                    "verification_timeline": "N/A",
                    "stage4_flow": FLOW_NAME,
                    "stage1_promise_str": promise_str,
                    "stage4_filtered": "yes",
                    "stage4_raw_timeline": "N/A",
                    "stage4_postprocess_rule": "",
                    "stage4_error": "recovered_missing",
                })
            print(f"[{index}/{len(rows)}] {rid} -> recovered", flush=True)
            continue

        promise_str = stage1.get(rid, "")
        if not passes_stage1(promise_str):
            output_rows.append(
                {
                    "id": rid,
                    "verification_timeline": "N/A",
                    "stage4_flow": FLOW_NAME,
                    "stage1_promise_str": promise_str,
                    "stage4_filtered": "yes",
                    "stage4_raw_timeline": "N/A",
                    "stage4_postprocess_rule": "",
                    "stage4_error": "",
                }
            )
            print(f"[{index}/{len(rows)}] {rid} -> N/A (filtered)", flush=True)
            # Write incrementally every 50 rows for checkpoint
            if index % 50 == 0:
                write_csv(str(output_path), output_rows)
            continue

        raw = ""
        metadata: dict[str, Any] = {}
        error = ""
        try:
            prompt = build_prompt_cot(system_prompt, str(row.get(args.data_col, "")))
            raw, metadata = predict_one(args.codex_bin, args.model, prompt, args.timeout)
            label, error = normalize_timeline(raw)
        except Exception as exc:
            label = "N/A"
            error = f"codex_error:{type(exc).__name__}"
            metadata = {"error": str(exc)}

        if error:
            error_count += 1
        else:
            predicted_count += 1

        output_rows.append(
            {
                "id": rid,
                "verification_timeline": label,
                "stage4_flow": FLOW_NAME,
                "stage1_promise_str": promise_str,
                "stage4_filtered": "no",
                "stage4_raw_timeline": raw[:100].replace("\n", " ") if raw else "",
                "stage4_postprocess_rule": "cot_extract",
                "stage4_error": error,
            }
        )
        print(f"[{index}/{len(rows)}] {rid} -> {label}", flush=True)
        # Write incrementally every 10 predicted rows for checkpoint
        if predicted_count % 10 == 0:
            write_csv(str(output_path), output_rows)

    output = resolve_path(args.output)
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
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
