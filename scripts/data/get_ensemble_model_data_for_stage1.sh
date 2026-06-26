#!/usr/bin/env bash
# Prepare Stage 1 ensemble model data and per-seed stratified splits.
#
# Run from anywhere:
#   bash scripts/data/get_ensemble_model_data_for_stage1.sh
#
# This mirrors exp/ensemble/stage1/ensemble/make_splits.sh:
# - build the Stage 1 ensemble input dataset
# - generate one deterministic train/val split per seed
# - write outputs under data/ensemble_data/stage1/ensemble/seed<S>/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ===== PARAMETER SPACE (override with env vars) ===============================
SEEDS=(${SEEDS:-42 7 123 2024 31337})
VAL_RATIO="${VAL_RATIO:-0.2}"
LABEL_KEY="${LABEL_KEY:-promise_status}"
INPUT_DIR="${INPUT_DIR:-data/synthesis_data/stage1}"
OUTPUT_DIR="${OUTPUT_DIR:-data/ensemble_data/stage1}"
SYNTHESIS_FILE="${SYNTHESIS_FILE:-top1_a3_b1.json}"
VAL_INPUT="${VAL_INPUT:-data/raw_data/vpesg4k_val_1000.json}"
NAME="${NAME:-a3_b1_add_val}"
PYTHON="${PYTHON:-.venv/bin/python}"
# =============================================================================

if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

SYNTHESIS_INPUT="$INPUT_DIR/$SYNTHESIS_FILE"
RAW_OUT="$OUTPUT_DIR/raw_data/$SYNTHESIS_FILE"
INPUT="$OUTPUT_DIR/${NAME}.json"

[ -f "$SYNTHESIS_INPUT" ] || { echo "[error] missing synthesis input: $SYNTHESIS_INPUT" >&2; exit 1; }
[ -f "$VAL_INPUT" ] || { echo "[error] missing validation input: $VAL_INPUT" >&2; exit 1; }

mkdir -p "$(dirname "$RAW_OUT")"

echo "### stage1 ensemble data: synthesis=$SYNTHESIS_INPUT val=$VAL_INPUT"
"$PYTHON" - "$SYNTHESIS_INPUT" "$VAL_INPUT" "$RAW_OUT" "$INPUT" <<'PY'
import json
import os
import shutil
import sys
from collections import Counter

synthesis_input, val_input, raw_out, combined_out = sys.argv[1:]

with open(synthesis_input, encoding="utf-8") as f:
    synthesis_rows = json.load(f)
with open(val_input, encoding="utf-8") as f:
    val_rows = json.load(f)

if not isinstance(synthesis_rows, list):
    raise SystemExit(f"[error] {synthesis_input} is not a JSON list")
if not isinstance(val_rows, list):
    raise SystemExit(f"[error] {val_input} is not a JSON list")

os.makedirs(os.path.dirname(raw_out), exist_ok=True)
os.makedirs(os.path.dirname(combined_out), exist_ok=True)
shutil.copyfile(synthesis_input, raw_out)

combined_rows = synthesis_rows + val_rows
with open(combined_out, "w", encoding="utf-8") as f:
    json.dump(combined_rows, f, ensure_ascii=False, indent=2)

def dist(rows):
    return dict(sorted(Counter(str(r.get("promise_status", "__missing__")) for r in rows).items()))

print(f"synthesis: {len(synthesis_rows)} rows dist={dist(synthesis_rows)} -> {raw_out}")
print(f"val      : {len(val_rows)} rows dist={dist(val_rows)}")
print(f"combined : {len(combined_rows)} rows dist={dist(combined_rows)} -> {combined_out}")
PY

echo "### ensemble splits ST1: seeds=[${SEEDS[*]}] val_ratio=$VAL_RATIO label=$LABEL_KEY"
for seed in "${SEEDS[@]}"; do
  OUT="$OUTPUT_DIR/ensemble/seed${seed}"
  mkdir -p "$OUT"
  echo "=== [seed=$seed] -> $OUT ==="
  "$PYTHON" - "$INPUT" "$OUT/${NAME}.train.json" "$OUT/${NAME}.val.json" \
    "$LABEL_KEY" "$VAL_RATIO" "$seed" <<'PY'
import json
import os
import random
import sys
from collections import Counter, defaultdict

input_path, train_out, val_out, label_key, val_ratio, seed = sys.argv[1:]
val_ratio = float(val_ratio)
seed = int(seed)

if not 0.0 < val_ratio < 1.0:
    raise SystemExit(f"[error] val_ratio must be in (0,1), got {val_ratio}")

with open(input_path, encoding="utf-8") as f:
    records = json.load(f)
if not isinstance(records, list):
    raise SystemExit(f"[error] {input_path} is not a JSON list")

buckets = defaultdict(list)
for row in records:
    buckets[str(row.get(label_key, "__missing__"))].append(row)

rng = random.Random(seed)
train_rows, val_rows = [], []
for label in sorted(buckets):
    rows = buckets[label][:]
    rng.shuffle(rows)
    n_val = round(len(rows) * val_ratio)
    val_rows.extend(rows[:n_val])
    train_rows.extend(rows[n_val:])

rng.shuffle(train_rows)
rng.shuffle(val_rows)

for path, rows in ((train_out, train_rows), (val_out, val_rows)):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def dist(rows):
    return dict(sorted(Counter(str(r.get(label_key, "__missing__")) for r in rows).items()))

total = len(records)
print(f"input : {total} rows  dist={dist(records)}")
print(f"train : {len(train_rows)} rows ({len(train_rows)/total:.1%})  dist={dist(train_rows)} -> {train_out}")
print(f"val   : {len(val_rows)} rows ({len(val_rows)/total:.1%})  dist={dist(val_rows)} -> {val_out}")
PY
done

echo "### stage1 ensemble data DONE"
