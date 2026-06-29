#!/usr/bin/env bash
# submit_5 final pipeline:
#   ST1 = focal_g3_w4 BERT + Gemma fallback (threshold=0.9)
#   ST2 = loop001_st2_synth_A2_b3_c1_25_add_val BERT + Gemma local fallback (no-rag, threshold=0.8)
#   ST3 = codex handle (load stage3/tmp/ 2000 preds, gate ST1+ST2 -> N/A)
#   ST4 = codex handle (load stage4/tmp/ 2000 preds, gate ST1 -> N/A)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../../../.." && pwd)"
cd "$ROOT"

HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="${DATA:-$ROOT/data/raw_data/vpesg4k_test_2000.json}"
VENV="$ROOT/.venv/bin/python"

mkdir -p "$HERE/stage1/tmp" "$HERE/stage2/tmp" "$HERE/stage2/raw" \
         "$HERE/stage3/tmp" "$HERE/stage4/tmp" "$HERE/logs"

SCRIPTS="$HERE/scripts"

ST1_BERT_RAW="$HERE/stage1/tmp/bert_raw.csv"
ST1_OUT="$HERE/stage1/bert_focal_g3_w4.csv"
ST2_BERT_RAW="$HERE/stage2/tmp/bert_raw.csv"
ST2_OUT="$HERE/stage2/bert.csv"

# Stage3: multitask BERT predict only (no codex fallback) -> gate via ST2.
# Stage4: codex handle (load pre-computed 2000-row codex from tmp/) -> gate.
# Each stage dir root holds exactly ONE csv (merge_pipeline picks it non-recursively).
ST3_MAIN="$HERE/stage3/stage3_bert.csv"                           # bert-only ST3 → merge
ST4_CODEX_RAW="$HERE/stage4/tmp/stage4_codex_predictions.csv"      # pre-computed 2000-row codex ST4
ST4_MAIN="$HERE/stage4/stage4_codex_gated.csv"                     # gated ST4 → merge

# ---------------- Stage1 config ----------------
STAGE1_BERT_CKPT="${STAGE1_BERT_CKPT:-/models/submit_5_a3_b1_add_val_focal_g3_w4/best_st1.pt}"
STAGE1_BERT_MODEL="${STAGE1_BERT_MODEL:-hfl/chinese-roberta-wwm-ext-large}"
STAGE1_CONF_THRESHOLD="${STAGE1_CONF_THRESHOLD:-0.6}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-auto}"
DEVICE_GEMMA="${DEVICE_GEMMA:-$DEVICE}"
GEMMA_BASE="${GEMMA_BASE:-/models/unsloth-gemma-4-12b}"
GEMMA_ADAPTER="${GEMMA_ADAPTER:-/models/gemma4_st12_mix}"
GEMMA_MAX_NEW_TOKENS="${GEMMA_MAX_NEW_TOKENS:-1000}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

# --- ST1: BERT + Gemma fallback ---
echo "[$(date '+%H:%M:%S')] START st1 bert+gemma_fallback"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage1/pred_by_bert.py \
    --data "$DATA" \
    --finetune-path "$STAGE1_BERT_CKPT" \
    --model "$STAGE1_BERT_MODEL" \
    --mode finetune \
    --output "$ST1_BERT_RAW" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    2>&1 | tee "$HERE/logs/st1_bert.log"

PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage1/pred_by_bert_gemma.py \
    --bert-pred "$ST1_BERT_RAW" \
    --data "$DATA" \
    --output "$ST1_OUT" \
    --gemma-base "$GEMMA_BASE" \
    --gemma-adapter "$GEMMA_ADAPTER" \
    --device "$DEVICE_GEMMA" \
    --conf-threshold "$STAGE1_CONF_THRESHOLD" \
    --max-new-tokens "$GEMMA_MAX_NEW_TOKENS" \
    --run-id "${RUN_ID}_stage1" \
    2>&1 | tee "$HERE/logs/st1_gemma.log"
echo "[$(date '+%H:%M:%S')] DONE  st1 bert+gemma_fallback"

# ---------------- Stage2 config ----------------
STAGE2_CKPT="${STAGE2_CKPT:-/models/esg_contest/loop001_st2_synth_A2_b3_c1_25_add_val_ce/best_st2.pt}"
STAGE2_BERT_MODEL="${STAGE2_BERT_MODEL:-hfl/chinese-roberta-wwm-ext-large}"
STAGE2_THRESHOLD="${STAGE2_THRESHOLD:-0.9}"

# --- ST2a: BERT ---
echo "[$(date '+%H:%M:%S')] START st2a bert"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage2/pred_by_bert.py \
    --data "$DATA" \
    --finetune-path "$STAGE2_CKPT" \
    --model "$STAGE2_BERT_MODEL" \
    --stage1-csv "$ST1_OUT" \
    --stage1-gate-col promise_status \
    --output "$ST2_BERT_RAW" \
    --text-mode data \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    2>&1 | tee "$HERE/logs/st2_bert.log"
echo "[$(date '+%H:%M:%S')] DONE  st2a bert"

# --- ST2b: Gemma local fallback ---
echo "[$(date '+%H:%M:%S')] START st2b gemma_local (threshold=$STAGE2_THRESHOLD)"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage2/pred_by_bert_gemma.py \
    --bert-pred "$ST2_BERT_RAW" \
    --data "$DATA" \
    --output "$ST2_OUT" \
    --threshold "$STAGE2_THRESHOLD" \
    --backend local \
    --gemma-base "$GEMMA_BASE" \
    --gemma-adapter "$GEMMA_ADAPTER" \
    --device "$DEVICE_GEMMA" \
    --max-new-tokens "$GEMMA_MAX_NEW_TOKENS" \
    --no-rag \
    --run-id "${RUN_ID}_stage2" \
    --raw-output-dir "$HERE/stage2/raw" \
    --token-usage-output "$HERE/stage2/tmp/token_usage.jsonl" \
    2>&1 | tee "$HERE/logs/st2_gemma.log"
echo "[$(date '+%H:%M:%S')] DONE  st2b gemma_local"

# ---------------- Stage3: MultiTask BERT only (no codex fallback) ----------------
STAGE3_CKPT="${STAGE3_CKPT:-/models/submit_5_mt_st123_twsweep/w0_2_0_3_0_5/best_multitask_st3.pt}"
STAGE3_BERT_MODEL="${STAGE3_BERT_MODEL:-hfl/chinese-roberta-wwm-ext-large}"

# --- ST3: MultiTask BERT predict (cascade-aware via ST2 gate) ---
echo "[$(date '+%H:%M:%S')] START st3 multitask bert (no fallback)"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage3/pred_by_bert_multitask.py \
    --data "$DATA" \
    --finetune-path "$STAGE3_CKPT" \
    --model "$STAGE3_BERT_MODEL" \
    --stage2-csv "$ST2_OUT" \
    --stage2-gate-col evidence_status \
    --output "$ST3_MAIN" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    2>&1 | tee "$HERE/logs/st3_bert.log"
echo "[$(date '+%H:%M:%S')] DONE  st3 multitask bert"

# ---------------- Stage4 codex handle: load 2000 codex preds + gate ----------------
# Loads the pre-computed ungated codex predictions from stage4/tmp/, then drifts
# ST1=No rows to N/A via the gate script. Gated output -> merge.
echo "[$(date '+%H:%M:%S')] START st4 codex gate"
PYTHONUNBUFFERED=1 "$VENV" "$SCRIPTS/apply_stage1_gate_to_stage4.py" \
    --stage1 "$ST1_OUT" \
    --stage4 "$ST4_CODEX_RAW" \
    --output "$ST4_MAIN" \
    2>&1 | tee "$HERE/logs/st4_codex_gate.log"
echo "[$(date '+%H:%M:%S')] DONE  st4 codex gate"

# ---------------- Merge ----------------
echo "[$(date '+%H:%M:%S')] START merge"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/merge_pipeline.py \
    --pipeline-dir "$HERE" \
    --output "$HERE/submission.csv"
echo "[$(date '+%H:%M:%S')] DONE  merge → $HERE/submission.csv"
