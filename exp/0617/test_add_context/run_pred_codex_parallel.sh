#!/usr/bin/env bash
# Parallel Stage 3 Codex predictor for the 0616/test_add_context 600-row probe.
#
# Splits DATA into WORKERS JSON shards, runs core/human/predict/stage3/pred_by_codex.py
# concurrently, then merges shard CSVs back in the original DATA row order.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
EXP="$ROOT/exp/integrated_stage_predictions/0616/test_add_context/stage3"

DATA="${DATA:-$EXP/data/val_ctxhit.json}"
GATE="${GATE:-$EXP/data/val_ctxhit.json}"
MODEL="${MODEL:-gpt-5.5}"
CODEX_BIN="${CODEX_BIN:-codex}"
PROMPT_PATH="${PROMPT_PATH:-configs/prompt/stage3/codex/clear_notclear_with_context.txt}"
WORKERS="${WORKERS:-4}"
TIMEOUT="${TIMEOUT:-300}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-1}"
PRINT_PROMPT="${PRINT_PROMPT:-0}"                 # 1/true: print first passed prompt per shard to logs
ADD_CONTEXT="${ADD_CONTEXT:-0}"                   # 1/true: augment shard data with same-page OCR context
CONTEXT_BUDGET="${CONTEXT_BUDGET:-800}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-after_biased}"  # after_biased or symmetric
CROSS_PAGE="${CROSS_PAGE:-1}"
PREFIX_CHARS="${PREFIX_CHARS:-18}"
RUN_ID="${RUN_ID:-codex_parallel_$(date '+%Y%m%d_%H%M%S')}"

OUT_DIR="$EXP/preds/codex_parallel/$RUN_ID"
RAW_DIR="$EXP/raw/codex_parallel/$RUN_ID"
LOG_DIR="$EXP/logs/codex_parallel/$RUN_ID"
TMP_DIR="$EXP/tmp/codex_parallel/$RUN_ID"
MERGED="$EXP/preds/st3_codex_parallel_${RUN_ID}.csv"

mkdir -p "$OUT_DIR" "$RAW_DIR" "$LOG_DIR" "$TMP_DIR" "$EXP/preds"

echo "[$(date '+%H:%M:%S')] split DATA=$DATA workers=$WORKERS limit=${LIMIT:-none} run_id=$RUN_ID"
"$PY" - "$DATA" "$TMP_DIR" "$WORKERS" "${LIMIT:-}" "$ADD_CONTEXT" "$CONTEXT_BUDGET" "$CONTEXT_WINDOW" "$CROSS_PAGE" "$PREFIX_CHARS" <<'PY'
import json
import sys
from pathlib import Path

from core.human.predict.stage3.pred_by_qwen import augment_data, build_context_index

data_path = Path(sys.argv[1])
tmp_dir = Path(sys.argv[2])
workers = int(sys.argv[3])
limit = int(sys.argv[4]) if sys.argv[4] else None
add_context = sys.argv[5] in {"1", "true", "True"}
context_budget = int(sys.argv[6])
context_window = sys.argv[7]
cross_page = sys.argv[8] in {"1", "true", "True"}
prefix_chars = int(sys.argv[9])

rows = json.loads(data_path.read_text(encoding="utf-8"))
if limit is not None:
    rows = rows[:limit]
if workers < 1:
    raise SystemExit("WORKERS must be >= 1")
if context_window not in {"after_biased", "symmetric"}:
    raise SystemExit("CONTEXT_WINDOW must be after_biased or symmetric")

context_index = build_context_index(
    Path("data/generated/raw_doc_table.jsonl"),
    Path("data/generated/raw_page_table.jsonl"),
) if add_context else None
hit_counts = {}
if add_context:
    augmented_rows = []
    for row in rows:
        out = dict(row)
        original = str(out.get("data", ""))
        augmented, hit = augment_data(
            out,
            original,
            context_index,
            context_budget,
            context_window,
            cross_page,
            prefix_chars,
        )
        out["data"] = augmented
        out["_codex_context_hit"] = hit
        hit_counts[hit] = hit_counts.get(hit, 0) + 1
        augmented_rows.append(out)
    rows = augmented_rows

tmp_dir.mkdir(parents=True, exist_ok=True)
manifest = {
    "rows": len(rows),
    "workers": workers,
    "add_context": add_context,
    "context_budget": context_budget if add_context else None,
    "context_window": context_window if add_context else None,
    "cross_page": cross_page if add_context else None,
    "prefix_chars": prefix_chars if add_context else None,
    "context_hit_counts": hit_counts,
    "shards": [],
}
for shard in range(workers):
    shard_rows = rows[shard::workers]
    path = tmp_dir / f"shard_{shard:02d}.json"
    path.write_text(json.dumps(shard_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["shards"].append({"shard": shard, "path": str(path), "rows": len(shard_rows)})
(tmp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

pids=()
for shard in $(seq 0 $((WORKERS - 1))); do
  shard_id="$(printf '%02d' "$shard")"
  shard_json="$TMP_DIR/shard_${shard_id}.json"
  shard_out="$OUT_DIR/shard_${shard_id}.csv"
  shard_raw="$RAW_DIR/shard_${shard_id}"
  shard_usage="$OUT_DIR/token_usage_shard_${shard_id}.jsonl"
  shard_log="$LOG_DIR/shard_${shard_id}.log"

  if [[ ! -s "$shard_json" ]]; then
    continue
  fi

  resume_args=()
  if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
    resume_args=(--resume)
  fi

  echo "[$(date '+%H:%M:%S')] start shard=$shard_id"
  if [[ "$PRINT_PROMPT" == "1" || "$PRINT_PROMPT" == "true" ]]; then
    "$PY" - "$shard_json" "$GATE" "$PROMPT_PATH" "$shard_log.prompt.txt" <<'PY'
import sys
from pathlib import Path

from core.human.predict.stage3.pred_by_codex import (
    build_prompt,
    load_stage2_gate,
    load_system_prompt,
    read_data_rows,
)

data_path = Path(sys.argv[1])
gate_path = Path(sys.argv[2])
prompt_path = Path(sys.argv[3])
out_path = Path(sys.argv[4])

rows = read_data_rows(data_path)
stage2 = load_stage2_gate(gate_path, "evidence_status")
system_prompt = load_system_prompt(prompt_path)

chosen = None
for index, row in enumerate(rows, start=1):
    rid = str(row.get("id", index)).strip()
    if stage2.get(rid) == "Yes":
        chosen = (index, rid, row)
        break

if chosen is None:
    text = "No stage2-passed row in this shard; no Codex prompt would be sent.\n"
else:
    index, rid, row = chosen
    prompt = build_prompt(system_prompt, str(row.get("data", "")))
    text = (
        f"shard_prompt_index={index}\n"
        f"id={rid}\n"
        "----- PROMPT START -----\n"
        f"{prompt}\n"
        "----- PROMPT END -----\n"
    )

out_path.write_text(text, encoding="utf-8")
print(text)
PY
  fi
  PYTHONUNBUFFERED=1 "$PY" core/human/predict/stage3/pred_by_codex.py \
    --data "$shard_json" \
    --stage2-csv "$GATE" \
    --stage2-gate-col evidence_status \
    --output "$shard_out" \
    --model "$MODEL" \
    --codex-bin "$CODEX_BIN" \
    --prompt-path "$PROMPT_PATH" \
    --timeout "$TIMEOUT" \
    --run-id "${RUN_ID}_shard_${shard_id}" \
    --raw-output-dir "$shard_raw" \
    --token-usage-output "$shard_usage" \
    "${resume_args[@]}" \
    >"$shard_log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" != "0" ]]; then
  echo "[$(date '+%H:%M:%S')] one or more shards failed; see $LOG_DIR" >&2
  exit 1
fi

echo "[$(date '+%H:%M:%S')] merge shards -> $MERGED"
"$PY" - "$DATA" "$OUT_DIR" "$MERGED" "${LIMIT:-}" <<'PY'
import csv
import json
import sys
from pathlib import Path

data_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
merged_path = Path(sys.argv[3])
limit = int(sys.argv[4]) if sys.argv[4] else None

rows = json.loads(data_path.read_text(encoding="utf-8"))
if limit is not None:
    rows = rows[:limit]
row_ids = [str(row.get("id", index + 1)).strip() for index, row in enumerate(rows)]

by_id = {}
fieldnames = None
for csv_path in sorted(out_dir.glob("shard_*.csv")):
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = fieldnames or reader.fieldnames
        for rec in reader:
            by_id[str(rec.get("id", "")).strip()] = rec

missing = [rid for rid in row_ids if rid not in by_id]
if missing:
    raise SystemExit(f"missing {len(missing)} predictions, first={missing[:10]}")
if fieldnames is None:
    raise SystemExit("no shard CSV outputs found")

merged_path.parent.mkdir(parents=True, exist_ok=True)
with merged_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for rid in row_ids:
        writer.writerow(by_id[rid])
print(json.dumps({"output": str(merged_path), "rows": len(row_ids)}, ensure_ascii=False, indent=2))
PY

echo "[$(date '+%H:%M:%S')] done"
echo "merged: $MERGED"
echo "logs:   $LOG_DIR"
echo "raw:    $RAW_DIR"
