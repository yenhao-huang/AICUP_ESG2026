# submit_11 vs submit_10 差異分析

- Data: `vpesg4k_test_2000.json`（blind, 2000 列, 無 gold）
- 兩者唯一差別在 **ST3 方法**；ST1/ST2/ST4 完全相同。

============================================================
逐欄差異 (submission.csv, 2000 列)
============================================================

  promise_status (ST1)        :   0 列不同   ✅ 對齊
  evidence_status (ST2)       :   0 列不同   ✅ 對齊
  verification_timeline (ST4) :   0 列不同   ✅ 對齊
  evidence_quality (ST3)      : 190 列不同   ← 唯一差異
    方向 (submit_10 → submit_11):
      Not Clear → Clear        104
      Clear     → Not Clear     86

============================================================
ST3 方法對照
============================================================

  submit_10 : Codex 全量預測 (prompt = clear_notclear_with_context
              + add_page_abstract context) → gate by ST1+ST2
              來源: codex × 2000

  submit_11 : multitask BERT 全量 → 信心度 < 0.60 的 152 列 fallback 到 Codex
              (prompt = clear_notclear_with_context_scoped, offset OCR ctx;
               其中 46 列原 codex=N/A 已用 na-fix 強制重判)
              來源: bert_multitask × 1848 + codex_fallback × 152

============================================================
ST3 class_distribution (final)
============================================================

  class        submit_10   submit_11   Δ(11−10)
  Clear           1124        1142        +18
  Not Clear        252         234        −18
  Misleading         0           0          0
  N/A              624         624          0
  TOTAL           2000        2000

  → submit_11 整體更偏 Clear（multitask head 比 add_page_abstract codex 更常判 Clear）。

============================================================
190 差異列來源拆解 (active 列, 依 submit_11 的 ST3 來源)
============================================================

  bert_multitask  : 173 列
    → submit_11 用 multitask BERT、submit_10 用 codex，兩個「不同模型」對同一列判不同。
      這是 190 列差異的主體 (91%)。
  codex_fallback  :  17 列
    → 這 152 個低信心列兩邊「都是 codex」，但 prompt/context 不同
      (submit_11=offsetctx_scoped vs submit_10=add_page_abstract)，
      fallback 區 active 108 列中有 17 列因此判不同。

============================================================
結論 / 注意
============================================================
- ST1/ST2/ST4 三 stage 與 submit_10（亦即 submit_9）完全一致，差異純粹來自 ST3。
- 差異主因是「模型不同」(multitask BERT vs codex) 而非門檻：1848 個高信心列直接採 multitask，
  與 submit_10 的 codex 在 173 列上分歧。
- 次要來源是低信心 152 列兩邊 codex 的 prompt/context 不同，貢獻 17 列。
- 兩版 Misleading 皆為 0。
- Blind set 無 gold → 無法直接判定誰較準；如需裁決，建議在 val.json 上用同一套
  ST1/ST2 gate 分別跑 submit_10 與 submit_11 的 ST3，比較 ST3 Macro-F1。
