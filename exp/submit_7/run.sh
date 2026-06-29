#!/usr/bin/env bash
# submit_7 pipeline (0615 ensemble soft-vote):
#   ST1 = soft-vote over the 5 ST1 ensemble members (focal_g3_w4)
#   ST2 = soft-vote over the 5 ST2 members, then gated by ST1 (ST1=No -> N/A)
#   ST3 = soft-vote over the 5 ST3 multitask members, then gated by ST1+ST2
#   ST4 = pre-computed codex predictions, gated by ST1 (ST1=No -> N/A)
#
# Cascade gating (CLAUDE.md task dependency) is applied AFTER each soft vote,
# because the soft-vote handles emit ungated head labels for every row. Each
# stage dir root holds exactly ONE non-delta csv (merge_pipeline picks it
# non-recursively); raw/ungated intermediates live under stageN/tmp/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"   # -> repo root /workspace/esg_contest
cd "$ROOT"
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv/bin/python"
SCRIPTS="$HERE/scripts"

DATA="${DATA:-$ROOT/data/raw_data/vpesg4k_test_2000.json}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# ----- ensemble member checkpoints (quoted so bash does NOT expand the glob;
#       soft_vote.py expands it internally) -----
ST1_GLOB="${ST1_GLOB:-/models/ensemble_st1_a3_b1_focal_g3_w4_seed*/best_st1.pt}"
ST2_GLOB="${ST2_GLOB:-/models/ensemble_st2_mix_a2_b3_seed*/best_st2.pt}"
ST3_GLOB="${ST3_GLOB:-/models/ensemble_mt_st123_w1_8_30_mlval_seed*/best_multitask_st3.pt}"

# ----- pre-computed (ungated) stage4 codex predictions -----
ST4_CODEX_RAW="${ST4_CODEX_RAW:-$ROOT/exp/integrated_stage_predictions/0615/ensemble/submit/stage4/tmp/stage4_codex_predictions_fixed.csv}"
# ----- pre-computed na-fix patch: 14 selected N/A rows re-predicted by codex
#       (boundary_rules_v4). Only rows still N/A after the ST1 gate are patched. -----
ST4_NA_FIX_14="${ST4_NA_FIX_14:-$ROOT/exp/integrated_stage_predictions/0615/fix_stage4_predictions/stage4_codex_predictions_14.csv}"

mkdir -p "$HERE/stage1/tmp" "$HERE/stage2/tmp" "$HERE/stage3/tmp" "$HERE/stage4/tmp" "$HERE/logs"

ST1_OUT="$HERE/stage1/softvote.csv"
ST2_RAW="$HERE/stage2/tmp/softvote_raw.csv"
ST2_OUT="$HERE/stage2/softvote_gated.csv"
ST3_RAW="$HERE/stage3/tmp/softvote_raw.csv"
ST3_OUT="$HERE/stage3/softvote_gated.csv"
ST4_OUT="$HERE/stage4/codex_gated.csv"

# ---------------- Stage1: soft vote ----------------
echo "[$(date '+%H:%M:%S')] START st1 soft-vote"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage1/soft_vote.py \
    --data "$DATA" --ckpt-glob "$ST1_GLOB" \
    --output "$ST1_OUT" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    2>&1 | tee "$HERE/logs/st1_softvote.log"
# soft_vote.py also drops softvote.members.csv into stage1/; move it under tmp/
# so merge_pipeline's find_main_csv sees exactly ONE csv in stage1/.
[ -f "$HERE/stage1/softvote.members.csv" ] && mv "$HERE/stage1/softvote.members.csv" "$HERE/stage1/tmp/"
echo "[$(date '+%H:%M:%S')] DONE  st1 soft-vote"

# ---------------- Stage2: soft vote -> gate by ST1 ----------------
echo "[$(date '+%H:%M:%S')] START st2 soft-vote"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage2/soft_vote.py \
    --data "$DATA" --ckpt-glob "$ST2_GLOB" \
    --output "$ST2_RAW" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    2>&1 | tee "$HERE/logs/st2_softvote.log"
PYTHONUNBUFFERED=1 "$VENV" "$SCRIPTS/apply_stage1_gate_to_stage2.py" \
    --stage1 "$ST1_OUT" --stage2 "$ST2_RAW" --output "$ST2_OUT" \
    2>&1 | tee "$HERE/logs/st2_gate.log"
echo "[$(date '+%H:%M:%S')] DONE  st2 soft-vote + gate"

# ---------------- Stage3: soft vote -> gate by ST1+ST2 ----------------
echo "[$(date '+%H:%M:%S')] START st3 soft-vote"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage3/soft_vote.py \
    --data "$DATA" --ckpt-glob "$ST3_GLOB" \
    --output "$ST3_RAW" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    2>&1 | tee "$HERE/logs/st3_softvote.log"
PYTHONUNBUFFERED=1 "$VENV" "$SCRIPTS/apply_stage12_gate_to_stage3.py" \
    --stage1 "$ST1_OUT" --stage2 "$ST2_OUT" --stage3 "$ST3_RAW" --output "$ST3_OUT" \
    2>&1 | tee "$HERE/logs/st3_gate.log"
echo "[$(date '+%H:%M:%S')] DONE  st3 soft-vote + gate"

# ---------------- Stage4: codex predictions -> gate by ST1 -> patch 14 na-fix ----------------
echo "[$(date '+%H:%M:%S')] START st4 codex gate"
[ -f "$ST4_CODEX_RAW" ] || { echo "[error] missing stage4 codex preds: $ST4_CODEX_RAW" >&2; exit 1; }
ST4_GATED_PRE14="$HERE/stage4/tmp/codex_gated_pre14.csv"
PYTHONUNBUFFERED=1 "$VENV" "$SCRIPTS/apply_stage1_gate_to_stage4.py" \
    --stage1 "$ST1_OUT" --stage4 "$ST4_CODEX_RAW" --output "$ST4_GATED_PRE14" \
    2>&1 | tee "$HERE/logs/st4_gate.log"
echo "[$(date '+%H:%M:%S')] DONE  st4 codex gate"

# --- patch the 14 pre-computed na-fix rows (only those still N/A after the gate;
#     all 14 are ST1=Yes, so補成具體 timeline 不破壞 cascade) ---
echo "[$(date '+%H:%M:%S')] START st4 na-fix patch (14 rows)"
[ -f "$ST4_NA_FIX_14" ] || { echo "[error] missing na-fix csv: $ST4_NA_FIX_14" >&2; exit 1; }
PYTHONUNBUFFERED=1 "$VENV" - "$ST4_GATED_PRE14" "$ST4_NA_FIX_14" "$ST4_OUT" <<'PY' 2>&1 | tee "$HERE/logs/st4_na_fix14.log"
import csv, sys, collections
base_path, patch_path, out_path = sys.argv[1:4]
patch = {r["id"]: r["verification_timeline"] for r in csv.DictReader(open(patch_path, encoding="utf-8-sig"))}
with open(base_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
applied = 0
for r in rows:
    rid = r["id"]
    if rid in patch and r["verification_timeline"] == "N/A" and patch[rid] not in ("", "N/A"):
        r["verification_timeline"] = patch[rid]
        if "stage4_raw_timeline" in r: r["stage4_raw_timeline"] = patch[rid]
        if "stage4_postprocess_rule" in r: r["stage4_postprocess_rule"] = "na_fix_14_codex"
        applied += 1
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print(f"  na-fix patched = {applied}/{len(patch)}")
print(f"  timeline 分布 = {dict(collections.Counter(r['verification_timeline'] for r in rows))}")
PY
echo "[$(date '+%H:%M:%S')] DONE  st4 na-fix patch -> $ST4_OUT"

# ---------------- Merge ----------------
echo "[$(date '+%H:%M:%S')] START merge"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/merge_pipeline.py \
    --pipeline-dir "$HERE" --output "$HERE/submission.csv"
echo "[$(date '+%H:%M:%S')] DONE  merge -> $HERE/submission.csv"
