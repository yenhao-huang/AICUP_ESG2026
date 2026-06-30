#!/usr/bin/env bash
# submission_13 = ST1/ST2/ST4 沿用 submission_12，僅 ST3 改為：
#
#   ST3 = 直接用 multitask BERT checkpoint 預測 evidence_quality
#         模型: /models/submit_5_mt_st123_twsweep/w0_2_0_3_0_5/best_multitask_st3.pt
#         gate: 在 ST2 的 evidence_status==Yes 上預測；其餘列 -> N/A
#         (不做 submission_12 的 codex low-conf fallback)
#     -> stage3/bert_multitask_gated.csv   (單一模型，非 soft-vote)
#     -> merge ST1+ST2+ST3+ST4 -> submission.csv
#
# 為何 --stage2-csv 即等同完整 cascade gate：stage2/softvote_gated.csv 已是
# ST1-gated（ST1=No 的列 evidence_status 已為 N/A），故「evidence_status==Yes」
# 同時排除 ST1=No 與 ST2=No，輸出與 apply_stage12_gate_to_stage3 一致。
#
# Data-use（CLAUDE.md）: ST3 模型輸入只有 `data` 文字欄（pred_by_bert_multitask
# text_mode 固定為 "data"）；gate 來自上一階段「預測」CSV，非標註欄位。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"   # -> /workspace/esg_contest
cd "$ROOT"
VENV="$ROOT/.venv/bin/python"

DATA="${DATA:-$ROOT/data/raw_data/vpesg4k_test_2000.json}"
STAGE3_CKPT="${STAGE3_CKPT:-/models/submit_5_mt_st123_twsweep/w0_2_0_3_0_5/best_multitask_st3.pt}"
STAGE3_BERT_MODEL="${STAGE3_BERT_MODEL:-hfl/chinese-roberta-wwm-ext-large}"
DEVICE="${DEVICE:-cuda:1}"          # GPU0 常被占用，預設用空閒的 GPU1
BATCH_SIZE="${BATCH_SIZE:-8}"

ST1_OUT="$HERE/stage1/bert_gemma.csv"
ST2_OUT="$HERE/stage2/softvote_gated.csv"        # 已 ST1-gated 的 stage2 結果，當 ST3 gate
ST3_OUT="$HERE/stage3/bert_multitask_gated.csv"  # 單一 multitask 模型 + gate 後最終 ST3（非 soft-vote）

mkdir -p "$HERE/stage3/tmp" "$HERE/logs"

for f in "$STAGE3_CKPT" "$DATA" "$ST1_OUT" "$ST2_OUT" "$HERE/stage4/codex_gated.csv"; do
    [ -e "$f" ] || { echo "[error] missing input: $f" >&2; exit 1; }
done

# ---------------- ST3: multitask BERT，gate by ST2 evidence_status==Yes ----------------
echo "[$(date '+%H:%M:%S')] START st3 multitask bert (gate: ST2 evidence_status=Yes)"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/stage3/pred_by_bert_multitask.py \
    --data "$DATA" \
    --finetune-path "$STAGE3_CKPT" \
    --model "$STAGE3_BERT_MODEL" \
    --stage2-csv "$ST2_OUT" \
    --stage2-gate-col evidence_status \
    --output "$ST3_OUT" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    2>&1 | tee "$HERE/logs/st3_multitask.log"
echo "[$(date '+%H:%M:%S')] DONE  st3 -> $ST3_OUT"

# ---------------- Merge ST1+ST2+ST3+ST4 ----------------
echo "[$(date '+%H:%M:%S')] START merge"
PYTHONUNBUFFERED=1 "$VENV" core/human/predict/merge_pipeline.py \
    --pipeline-dir "$HERE" --output "$HERE/submission.csv"
echo "[$(date '+%H:%M:%S')] DONE  merge -> $HERE/submission.csv"
