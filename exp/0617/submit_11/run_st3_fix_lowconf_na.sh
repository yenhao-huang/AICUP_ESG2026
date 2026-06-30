#!/usr/bin/env bash
# submit_11 ST3 "fix": 對「multitask 信心度 < CONF 且 codex 該列沒有實際 clarity 標籤
# (codex=N/A / stage2_filter)」的列，用 codex 強制重新預測一個 Clear/Not Clear/
# Misleading 標籤，補進 codex 預測檔，讓 run_st3_mt_codex_fallback.sh 的 fallback
# 能真的吃到 codex 結果。
#
# 流程:
#   1. 由 stage3/tmp/softvote_raw.csv (multitask) + st3_codex_..._scoped.csv (codex)
#      動態挑出 target ids (低信心 & codex 無實際標籤)。
#   2. 從 submit_8 的 offset-context augmented JSON 取這些 id 的 data (NUL 去除)。
#   3. 建一份 forced-Yes gate，讓 pred_by_codex.py 不被 stage2 短路成 N/A。
#   4. 切 shard 平行跑 core/human/predict/stage3/pred_by_codex.py (同 scoped prompt)。
#   5. 合併 -> st3_codex_lowconf_nafix.csv，並 patch 進 scoped 檔的副本
#      -> st3_codex_test2000_offsetctx_scoped_fixed.csv。
#
# 跑完後把 fallback 腳本的 ST3_CODEX 指到 *_fixed.csv 再重跑即可 (見結尾提示)。
# Data-use: prompt/context 只由 `data` 推得 (offset OCR 同頁上下文已預先注入 data)。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
SUB8="$ROOT/exp/integrated_stage_predictions/0616/submit_8"

CONF="${CONF:-0.60}"
AUG_DATA="${AUG_DATA:-$SUB8/stage3/data/test2000_offsetctx.json}"
PROMPT_PATH="${PROMPT_PATH:-configs/prompt/stage3/codex/clear_notclear_with_context_scoped.txt}"
MODEL="${MODEL:-gpt-5.5}"
CODEX_BIN="${CODEX_BIN:-codex}"
WORKERS="${WORKERS:-8}"
TIMEOUT="${TIMEOUT:-300}"
RESUME="${RESUME:-1}"
RUN_ID="${RUN_ID:-st3_lowconf_nafix}"

MT="$HERE/stage3/tmp/softvote_raw.csv"
CODEX_SCOPED="$HERE/stage3/tmp/st3_codex_test2000_offsetctx_scoped.csv"
WORKDIR="$HERE/stage3/fix/$RUN_ID"
WORK_DATA="$WORKDIR/subset.json"
WORK_GATE="$WORKDIR/gate_forced_yes.csv"
TMP_DIR="$WORKDIR/shards"
OUT_DIR="$WORKDIR/preds"
RAW_DIR="$WORKDIR/raw"
LOG_DIR="$WORKDIR/logs"
FIX_CSV="$HERE/stage3/tmp/st3_codex_lowconf_nafix.csv"
CODEX_FIXED="$HERE/stage3/tmp/st3_codex_test2000_offsetctx_scoped_fixed.csv"

mkdir -p "$WORKDIR" "$TMP_DIR" "$OUT_DIR" "$RAW_DIR" "$LOG_DIR"

for f in "$MT" "$CODEX_SCOPED" "$AUG_DATA" "$PROMPT_PATH"; do
    [ -f "$f" ] || { echo "[error] missing input: $f" >&2; exit 1; }
done

# ---------------- 1+2+3. 選 target ids -> subset JSON + forced-Yes gate ----------------
echo "[$(date '+%H:%M:%S')] select low-conf(<$CONF) & codex-no-real-label ids"
"$PY" - "$MT" "$CODEX_SCOPED" "$AUG_DATA" "$CONF" "$WORK_DATA" "$WORK_GATE" <<'PY'
import csv, re, json, sys
from pathlib import Path
mt_p, cx_p, aug_p, conf_s, out_data, out_gate = sys.argv[1:7]
conf_th = float(conf_s)
SR = re.compile(r"score_[a-z_]+\s*=\s*([0-9.eE+-]+)")
REAL = {"Clear", "Not Clear", "Misleading"}

mt = list(csv.DictReader(open(mt_p, encoding="utf-8-sig")))
cx = {r["id"]: r for r in csv.DictReader(open(cx_p, encoding="utf-8-sig"))}
target = []
for r in mt:
    conf = max(float(x) for x in SR.findall(r["evidence_quality_reason"]))
    if conf >= conf_th:
        continue
    c = cx.get(r["id"])
    if c is None or c.get("evidence_quality") not in REAL:
        target.append(r["id"])
target_set = set(target)

aug = json.loads(Path(aug_p).read_text(encoding="utf-8"))
sel, nul = [], 0
for row in aug:
    rid = str(row.get("id", "")).strip()
    if rid in target_set:
        d = str(row.get("data", "")); nul += d.count("\x00")
        row = dict(row); row["data"] = d.replace("\x00", ""); sel.append(row)
got = {str(r.get("id", "")).strip() for r in sel}
missing = target_set - got
if missing:
    raise SystemExit(f"{len(missing)} target ids missing in aug data, first={sorted(missing)[:10]}")
Path(out_data).write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding="utf-8")

with open(out_gate, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "evidence_status"]); w.writeheader()
    for rid in target: w.writerow({"id": rid, "evidence_status": "Yes"})
print(json.dumps({"target_ids": len(target), "subset_rows": len(sel), "stripped_nul": nul}, ensure_ascii=False))
PY

# ---------------- 4. 切 shard 平行跑 codex ----------------
echo "[$(date '+%H:%M:%S')] split subset workers=$WORKERS"
"$PY" - "$WORK_DATA" "$TMP_DIR" "$WORKERS" <<'PY'
import json, sys
from pathlib import Path
data_path, tmp_dir, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = json.loads(data_path.read_text(encoding="utf-8"))
workers = min(workers, max(1, len(rows)))
tmp_dir.mkdir(parents=True, exist_ok=True)
for shard in range(workers):
    (tmp_dir / f"shard_{shard:02d}.json").write_text(
        json.dumps(rows[shard::workers], ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"rows": len(rows), "workers": workers}, ensure_ascii=False))
PY

pids=()
for shard_json in "$TMP_DIR"/shard_*.json; do
  [[ -s "$shard_json" ]] || continue
  sid="$(basename "$shard_json" .json | sed 's/shard_//')"
  resume_args=(); [[ "$RESUME" == "1" || "$RESUME" == "true" ]] && resume_args=(--resume)
  echo "[$(date '+%H:%M:%S')] start shard=$sid"
  PYTHONUNBUFFERED=1 "$PY" core/human/predict/stage3/pred_by_codex.py \
    --data "$shard_json" \
    --stage2-csv "$WORK_GATE" \
    --stage2-gate-col evidence_status \
    --output "$OUT_DIR/shard_${sid}.csv" \
    --model "$MODEL" \
    --codex-bin "$CODEX_BIN" \
    --prompt-path "$PROMPT_PATH" \
    --timeout "$TIMEOUT" \
    --run-id "${RUN_ID}_shard_${sid}" \
    --raw-output-dir "$RAW_DIR/shard_${sid}" \
    --token-usage-output "$OUT_DIR/token_usage_shard_${sid}.jsonl" \
    "${resume_args[@]}" \
    >"$LOG_DIR/shard_${sid}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == "0" ]] || { echo "[$(date '+%H:%M:%S')] shard(s) failed; see $LOG_DIR" >&2; exit 1; }

# ---------------- 5. 合併 -> FIX_CSV，patch 進 scoped 副本 -> CODEX_FIXED ----------------
echo "[$(date '+%H:%M:%S')] merge -> $FIX_CSV ; patch -> $CODEX_FIXED"
"$PY" - "$WORK_DATA" "$OUT_DIR" "$FIX_CSV" "$CODEX_SCOPED" "$CODEX_FIXED" <<'PY'
import csv, json, sys, collections
from pathlib import Path
data_path, out_dir, fix_path, scoped_path, fixed_path = map(Path, sys.argv[1:6])
row_ids = [str(r.get("id")).strip() for r in json.loads(data_path.read_text(encoding="utf-8"))]
by_id, fields = {}, None
for p in sorted(out_dir.glob("shard_*.csv")):
    with p.open(newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f); fields = fields or rd.fieldnames
        for rec in rd: by_id[str(rec.get("id", "")).strip()] = rec
missing = [i for i in row_ids if i not in by_id]
if missing:
    raise SystemExit(f"missing {len(missing)} fix preds, first={missing[:10]}")
with fix_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for i in row_ids: w.writerow(by_id[i])

# patch scoped 副本: 把這些 id 的 label 換成 fix 結果
scoped = list(csv.DictReader(open(scoped_path, encoding="utf-8-sig")))
sfields = scoped[0].keys() if scoped else fields
patched = 0
for r in scoped:
    rid = str(r.get("id", "")).strip()
    if rid in by_id:
        r["evidence_quality"] = by_id[rid].get("evidence_quality", r.get("evidence_quality"))
        if "evidence_quality_raw" in r: r["evidence_quality_raw"] = by_id[rid].get("evidence_quality", "")
        if "evidence_quality_source" in r: r["evidence_quality_source"] = "codex_nafix"
        if "evidence_quality_reason" in r: r["evidence_quality_reason"] = "lowconf_nafix_codex"
        patched += 1
with fixed_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(sfields)); w.writeheader(); w.writerows(scoped)
print(json.dumps({"fix_rows": len(row_ids), "patched_into_scoped": patched,
                  "fix_label_dist": dict(collections.Counter(by_id[i]["evidence_quality"] for i in row_ids))},
                 ensure_ascii=False))
PY

echo "[$(date '+%H:%M:%S')] done"
echo "  fix preds      : $FIX_CSV"
echo "  patched codex  : $CODEX_FIXED"
echo
echo "套用 fix 後重跑 fallback (用補好的 codex 檔):"
echo "  ST3_CODEX_OVERRIDE='$CODEX_FIXED' ; # 見下方"
echo "  在 run_st3_mt_codex_fallback.sh 裡把 ST3_CODEX 指到上面這個檔，或直接:"
echo "  ST3_CODEX=\"$CODEX_FIXED\" CONF=$CONF \"$HERE/run_st3_mt_codex_fallback.sh\""
