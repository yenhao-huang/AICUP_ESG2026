submit_5/submit/
├── run.sh
├── submission.csv                          ★ 最終輸出(2000 列, 5 欄)
│
├── scripts/
│   ├── apply_stage12_gate_to_stage3.py
│   └── apply_stage1_gate_to_stage4.py
│
├── stage1/
│   ├── bert_focal_g3_w4.csv                ★ ST1 最終(BERT+Gemma)→ merge 取這個
│   └── tmp/
│       └── bert_raw.csv                    ★ ST1 BERT 原始(Gemma 前)
│
├── stage2/
│   ├── bert.csv                            ★ ST2 最終(BERT+Gemma)→ merge 取這個
│   ├── raw/                                ★ Gemma 逐筆原始回覆(--raw-output-dir)
│   │   └── …                               ★
│   └── tmp/
│       ├── bert_raw.csv                    ★ ST2 BERT 原始(Gemma 前)
│       └── token_usage.jsonl               ★ Gemma token 用量
│
├── stage3/
│   ├── stage3_codex_gated.csv              ★ ST3 gated(N/A 漂移後)→ merge 取這個
│   └── tmp/
│       └── stage3_codex_predictions_merge.csv   (離線已存在, 2000 筆 ungated codex)
│
├── stage4/
│   ├── stage4_codex_gated.csv              ★ ST4 gated(N/A 漂移後)→ merge 取這個
│   └── tmp/
│       └── stage4_codex_predictions.csv         (離線已存在, 2000 筆 ungated codex)
│
└── logs/                                   ★
    ├── st1_bert.log   st1_gemma.log
    ├── st2_bert.log   st2_gemma.log
    ├── st3_codex_gate.log
    └── st4_codex_gate.log