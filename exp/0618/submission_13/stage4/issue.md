CSV 欄位拆解:13494, except, between_2_and_5, codex_ctx, model_output_error:invalid_label
- codex 其實輸出了 between_2_and_5(raw 欄),但少了 _years 後綴
- parser 嚴格比對合法 label 失敗 → 標 invalid_label → final 設成 except 哨兵值

語意上 codex 想說的就是 between_2_and_5_years(= submission.csv 的原值)。你要我用 codex 重跑這筆,我來做 —— 建單筆 data,用同一條 v6 pipeline 跑: