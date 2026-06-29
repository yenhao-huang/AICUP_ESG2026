#!/usr/bin/env bash
# submission_12 = ST1/ST2/ST4 沿用既有結果，僅 ST3 為：
#
#   ST3 multitask BERT 預測 (stage3/tmp/softvote_raw.csv, source=bert_multitask)
#     -> 對 max-softmax 信心度 < CONF (0.60) 的列，fallback 成已預先算好的
#        codex ST3 標籤 (stage3/tmp/add_context_test2000_codex.csv)
#     -> 用既有 ST1 (stage1/bert_gemma.csv) + ST2 (stage2/softvote_gated.csv)
#        做 cascade gate -> stage3/softvote_gated.csv
#     -> 重新 merge 出 submission.csv
#
# Fallback 規則 (data-only, 兩個輸入皆只由 `data` 推得):
#   - 信心度 = evidence_quality_reason 內 score_* 的最大值。
#   - 只有當 codex 該列給出實際 clarity 標籤 (Clear / Not Clear / Misleading)
#     時才取代；codex 為 N/A 的列保留 multitask 標籤，由本流程的 ST1+ST2 gate
#     統一決定 N/A。
#   - 注意: add_context_test2000_codex.csv 對全 2000 列皆有實際標籤 (無 N/A)，
#     故本版不需 submit_11 的 na-fix 步驟。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"   # -> /workspace/esg_contest
cd "$ROOT"
VENV="$ROOT/.venv/bin/python"
SCRIPTS="$ROOT/exp/integrated_stage_predictions/0617/submit_9/scripts"

CONF="${CONF:-0.60}"   # multitask max-softmax 低於此值 -> fallback 到 codex

ST3_MT="$HERE/stage3/tmp/softvote_raw.csv"                              # multitask BERT 預測
ST3_CODEX="${ST3_CODEX:-$HERE/stage3/tmp/add_context_test2000_codex.csv}"  # 預先算好的 codex 預測 (可用 ST3_CODEX 覆寫)
ST3_FALLBACK="$HERE/stage3/tmp/mt_codex_fallback_raw.csv"               # fallback 後 (ungated)
ST3_OUT="$HERE/stage3/softvote_gated.csv"                               # gate 後最終 ST3
ST1_OUT="$HERE/stage1/bert_gemma.csv"
ST2_OUT="$HERE/stage2/softvote_gated.csv"

mkdir -p "$HERE/stage3/tmp" "$HERE/logs"

for f in "$ST3_MT" "$ST3_CODEX" "$ST1_OUT" "$ST2_OUT"; do
    [ -f "$f" ] || { echo "[error] missing input: $f" >&2; exit 1; }
done

# ---------------- ST3: multitask -> codex low-conf fallback ----------------
echo "[$(date '+%H:%M:%S')] START st3 multitask -> codex fallback (conf<$CONF)"
PYTHONUNBUFFERED=1 "$VENV" - "$ST3_MT" "$ST3_CODEX" "$ST3_FALLBACK" "$CONF" <<'PY' 2>&1 | tee "$HERE/logs/st3_mt_codex_fallback.log"
import csv, re, sys, collections
mt_path, codex_path, out_path, conf_s = sys.argv[1:5]
conf_th = float(conf_s)
SCORE_RE = re.compile(r"score_[a-z_]+\s*=\s*([0-9.eE+-]+)")
REAL = {"Clear", "Not Clear", "Misleading"}

def max_prob(reason):
    vals = [float(x) for x in SCORE_RE.findall(reason or "")]
    return max(vals) if vals else 1.0   # 無分數視為高信心，不 fallback

codex = {r["id"]: r for r in csv.DictReader(open(codex_path, encoding="utf-8-sig"))}
with open(mt_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)

low = changed = skipped_no_codex = 0
for r in rows:
    conf = max_prob(r.get("evidence_quality_reason", ""))
    if conf >= conf_th:
        continue
    low += 1
    c = codex.get(r["id"])
    if c is None or c.get("evidence_quality") not in REAL:
        skipped_no_codex += 1
        continue
    if c["evidence_quality"] != r["evidence_quality"]:
        changed += 1
    r["evidence_quality"] = c["evidence_quality"]
    r["evidence_quality_raw"] = c.get("evidence_quality", r.get("evidence_quality_raw", ""))
    r["evidence_quality_source"] = "codex_fallback"
    r["evidence_quality_reason"] = f"mt_conf={conf:.6f}<{conf_th};codex={c['evidence_quality']}"

with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

print(f"  rows={len(rows)} low_conf(<{conf_th})={low} replaced_by_codex={low-skipped_no_codex} "
      f"label_changed={changed} kept_mt(no real codex)={skipped_no_codex}")
print(f"  ungated label 分布 = {dict(collections.Counter(r['evidence_quality'] for r in rows))}")
PY
echo "[$(date '+%H:%M:%S')] DONE  st3 fallback -> $ST3_FALLBACK"

# ---------------- ST3: gate by ST1 + ST2 ----------------
echo "[$(date '+%H:%M:%S')] START st3 gate (ST1+ST2)"
PYTHONUNBUFFERED=1 "$VENV" "$SCRIPTS/apply_stage12_gate_to_stage3.py" \
    --stage1 "$ST1_OUT" --stage2 "$ST2_OUT" --stage3 "$ST3_FALLBACK" --output "$ST3_OUT" \
    2>&1 | tee "$HERE/logs/st3_gate.log"
echo "[$(date '+%H:%M:%S')] DONE  st3 gate -> $ST3_OUT"

# ---------------- Merge ----------------
echo "[$(date '+%H:%M:%S')] START merge"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/merge_pipeline.py \
    --pipeline-dir "$HERE" --output "$HERE/submission.csv"
echo "[$(date '+%H:%M:%S')] DONE  merge -> $HERE/submission.csv"
