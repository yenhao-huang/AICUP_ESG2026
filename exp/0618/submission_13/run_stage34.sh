#!/usr/bin/env bash
# Stage3 + Stage4 + Merge — pairs with run_stage12.sh.
# PREREQUISITE: run_stage12.sh has already produced stage1/bert_gemma.csv and
# stage2/softvote_gated.csv (this script gates ST3/ST4 against them).
#
#   ST3 = single multitask BERT (submit_5 model w0_2_0_3_0_5), then gated by ST1+ST2
#   ST4 = EXISTING codex predictions (v6 add-context, NOT re-predicted here) ->
#         apply ST1 gate (ST1=No -> N/A)
#   Merge -> submission.csv
#
# ST4 does NO codex inference — it only re-applies the (new) ST1 gate to the
# already-computed v6 predictions. Point ST4_CODEX_RAW at a different raw codex
# CSV to gate a different ST4 source.
#
# Data-use: ST3 multitask reads only `data`. ST4 codex preds were produced
# separately (add-context exceeds data-only; user-approved probe).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"   # -> repo root /workspace/esg_contest
cd "$ROOT"
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv/bin/python"
SCRIPTS="$HERE/scripts"

DATA="${DATA:-$ROOT/data/raw_data/vpesg4k_test_2000.json}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# ----- ST3: single multitask BERT (submit_5 model), gated by ST1+ST2 -----
STAGE3_CKPT="${STAGE3_CKPT:-/models/submit_5_mt_st123_twsweep/w0_2_0_3_0_5/best_multitask_st3.pt}"
STAGE3_BERT_MODEL="${STAGE3_BERT_MODEL:-hfl/chinese-roberta-wwm-ext-large}"

# ----- ST4: EXISTING codex predictions to gate (no re-predict) -----
#   v6 add-context, NOGATE (raw, every row carries a timeline) — gated here by the new ST1.
ST4_CODEX_RAW="${ST4_CODEX_RAW:-$HERE/stage4/preds/codex/st4_add_context2_v6_test2000_nogate_codex.csv}"

# ----- prerequisite stage1/stage2 outputs (from run_stage12.sh) -----
ST1_OUT="$HERE/stage1/bert_gemma.csv"
ST2_OUT="$HERE/stage2/softvote_gated.csv"
for f in "$ST1_OUT" "$ST2_OUT"; do
  [ -f "$f" ] || { echo "[error] missing $f — run run_stage12.sh first" >&2; exit 1; }
done
[ -f "$ST4_CODEX_RAW" ] || { echo "[error] missing ST4 codex preds: $ST4_CODEX_RAW" >&2; exit 1; }

mkdir -p "$HERE/stage3/tmp" "$HERE/stage4/tmp" "$HERE/logs"

ST3_RAW="$HERE/stage3/tmp/softvote_raw.csv"
ST3_OUT="$HERE/stage3/softvote_gated.csv"
ST4_OUT="$HERE/stage4/codex_gated.csv"

# ---------------- Stage3: single multitask BERT -> gate by ST1+ST2 ----------------
echo "[$(date '+%H:%M:%S')] START st3 multitask bert (single model)"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage3/pred_by_bert_multitask.py \
    --data "$DATA" --finetune-path "$STAGE3_CKPT" --model "$STAGE3_BERT_MODEL" \
    --output "$ST3_RAW" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    2>&1 | tee "$HERE/logs/st3_bert.log"
PYTHONUNBUFFERED=1 "$VENV" "$SCRIPTS/apply_stage12_gate_to_stage3.py" \
    --stage1 "$ST1_OUT" --stage2 "$ST2_OUT" --stage3 "$ST3_RAW" --output "$ST3_OUT" \
    2>&1 | tee "$HERE/logs/st3_gate.log"
echo "[$(date '+%H:%M:%S')] DONE  st3 multitask bert + gate"

# ---------------- Stage4: apply ST1 gate to EXISTING codex preds (no re-predict) ----------------
echo "[$(date '+%H:%M:%S')] START st4 codex gate (source=$(basename "$ST4_CODEX_RAW"))"
PYTHONUNBUFFERED=1 "$VENV" "$SCRIPTS/apply_stage1_gate_to_stage4.py" \
    --stage1 "$ST1_OUT" --stage4 "$ST4_CODEX_RAW" --output "$ST4_OUT" \
    2>&1 | tee "$HERE/logs/st4_gate.log"
echo "[$(date '+%H:%M:%S')] DONE  st4 codex gate -> $ST4_OUT"

# ---------------- Merge ----------------
echo "[$(date '+%H:%M:%S')] START merge"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/merge_pipeline.py \
    --pipeline-dir "$HERE" --output "$HERE/submission.csv"
echo "[$(date '+%H:%M:%S')] DONE  merge -> $HERE/submission.csv"
