#!/usr/bin/env bash
# fix_stage4_gen_na.sh
# ---------------------------------------------------------------------------
# 目的：修 submit_5 stage4 的 cascade bug。
#   stage4/tmp/stage4_codex_predictions.csv 的 N/A 是綁舊 ST1 烤出來的。
#   submit_5 換了新 ST1 後，有一批 ST1=Yes 的列仍殘留 codex 的 N/A
#   （單向 gate 無法還原）。本腳本「特別」針對這些列，用 boundary_rules_v4
#   prompt + 多 worker 重新預測，產生具體 verification_timeline。
#
# 目標列 = (新 ST1 promise_status == Yes)  AND  (tmp codex ST4 == N/A)
#   → 這些是唯一需要被救的列；ST1=No 的 N/A 不動（cascade 本來就該 N/A）。
#
# 輸出：stage4/tmp/stage4_codex_predictions_fixed.csv（不覆蓋原始 tmp）。
#   之後你再自行跑 apply_stage1_gate_to_stage4.py + merge_pipeline。
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../../../.." && pwd)"
cd "$ROOT"

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv/bin/python"

# ---- config (可用環境變數覆寫) ----
DATA="${DATA:-$ROOT/data/raw_data/vpesg4k_test_2000.json}"
ST1="${ST1:-$HERE/stage1/bert_focal_g3_w4.csv}"
TMP_ST4="${TMP_ST4:-$HERE/stage4/tmp/stage4_codex_predictions.csv}"
PROMPT="${PROMPT:-$ROOT/configs/prompt/stage4/codex/boundary_rules_v4.txt}"
MODEL="${MODEL:-gpt-5.5}"
WORKERS="${WORKERS:-8}"
TIMEOUT="${TIMEOUT:-300}"
OUT="${OUT:-$HERE/stage4/tmp/stage4_codex_predictions_fixed.csv}"
WORKDIR="${WORKDIR:-$HERE/stage4/tmp/na_fix}"

echo "[cfg] DATA=$DATA"
echo "[cfg] ST1=$ST1"
echo "[cfg] TMP_ST4=$TMP_ST4"
echo "[cfg] PROMPT=$PROMPT"
echo "[cfg] MODEL=$MODEL  WORKERS=$WORKERS"
echo "[cfg] OUT=$OUT"

mkdir -p "$WORKDIR"

# ---------------- Step 1: 算目標列並分片 ----------------
echo "[$(date '+%H:%M:%S')] Step1: build target subset + shard into $WORKERS"
"$VENV" - "$DATA" "$ST1" "$TMP_ST4" "$WORKDIR" "$WORKERS" <<'PY'
import csv, json, sys, math
data_path, st1_path, tmp_path, workdir, workers = sys.argv[1:6]
workers = int(workers)

with open(data_path, encoding="utf-8") as f:
    data = {str(r["id"]).strip(): r for r in json.load(f)}

def load_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return {str(r["id"]).strip(): r for r in csv.DictReader(f)}

st1 = load_csv(st1_path)
tmp = load_csv(tmp_path)

target = [rid for rid in tmp
          if str(tmp[rid].get("verification_timeline", "")).strip() == "N/A"
          and str(st1.get(rid, {}).get("promise_status", "")).strip() == "Yes"]
target.sort(key=lambda x: int(x) if x.isdigit() else x)

print(f"  ST1=Yes 且 tmp ST4=N/A 的目標列 = {len(target)}")
if not target:
    print("  沒有需要修的列，結束。")
    # still write empty shards so the bash loop is a no-op
for k in range(workers):
    shard = [{"id": rid, "data": data[rid].get("data", "")}
             for i, rid in enumerate(target) if i % workers == k]
    with open(f"{workdir}/shard_{k}.json", "w", encoding="utf-8") as f:
        json.dump(shard, f, ensure_ascii=False)
    print(f"  shard_{k}: {len(shard)} rows")
PY

# ---------------- Step 2: 多 worker 平行 codex 預測 ----------------
echo "[$(date '+%H:%M:%S')] Step2: launch $WORKERS codex workers (prompt=$(basename "$PROMPT"))"
pids=()
for k in $(seq 0 $((WORKERS - 1))); do
    SHARD="$WORKDIR/shard_${k}.json"
    # 跳過空 shard
    if [ "$("$VENV" -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$SHARD")" = "0" ]; then
        continue
    fi
    PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage4/pred_by_codex.py \
        --data "$SHARD" \
        --output "$WORKDIR/shard_${k}.csv" \
        --prompt-path "$PROMPT" \
        --model "$MODEL" \
        --timeout "$TIMEOUT" \
        --raw-output-dir "$WORKDIR/raw_${k}" \
        --token-usage-output "$WORKDIR/token_${k}.jsonl" \
        --run-id "fix_na_shard_${k}" \
        >"$WORKDIR/worker_${k}.log" 2>&1 &
    pids+=($!)
    echo "  worker $k pid=$! -> $WORKDIR/worker_${k}.log"
done

fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "  WORKER pid=$pid FAILED" >&2
        fail=1
    fi
done
[ "$fail" = "0" ] || { echo "有 worker 失敗，檢查 $WORKDIR/worker_*.log"; exit 1; }
echo "[$(date '+%H:%M:%S')] Step2 done"

# ---------------- Step 3: 合併回 tmp，產生 fixed csv ----------------
echo "[$(date '+%H:%M:%S')] Step3: merge predictions -> $OUT"
"$VENV" - "$TMP_ST4" "$OUT" "$WORKDIR" "$WORKERS" <<'PY'
import csv, glob, sys, collections
tmp_path, out_path, workdir, workers = sys.argv[1:5]
workers = int(workers)

# new predictions from shards
new = {}
for k in range(workers):
    p = f"{workdir}/shard_{k}.csv"
    try:
        with open(p, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                new[str(r["id"]).strip()] = str(r.get("verification_timeline", "")).strip()
    except FileNotFoundError:
        pass

with open(tmp_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames or ["id", "verification_timeline"]
    rows = list(reader)

patched = 0
still_na = []
for r in rows:
    rid = str(r["id"]).strip()
    if rid in new:
        label = new[rid]
        if label and label != "N/A":
            r["verification_timeline"] = label
            patched += 1
        else:
            still_na.append(rid)

with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

after = collections.Counter(str(r["verification_timeline"]).strip() for r in rows)
print(f"  目標列(送預測) = {len(new)}")
print(f"  成功補上具體 timeline = {patched}")
print(f"  codex 仍回 N/A(沒救到) = {len(still_na)}  {still_na if still_na else ''}")
print(f"  fixed 檔 N/A 總數 = {after['N/A']}  (原 tmp = 345)")
print(f"  fixed 分布 = {dict(after)}")
print(f"  輸出 = {out_path}")
PY

echo "[$(date '+%H:%M:%S')] DONE"
echo "下一步（自行執行）："
echo "  1) gate:   $VENV $HERE/scripts/apply_stage1_gate_to_stage4.py --stage1 $ST1 --stage4 $OUT --output $HERE/stage4/stage4_codex_gated.csv"
echo "  2) merge:  $VENV core/human/predict/merge_pipeline.py --pipeline-dir $HERE --output $HERE/submission.csv"
