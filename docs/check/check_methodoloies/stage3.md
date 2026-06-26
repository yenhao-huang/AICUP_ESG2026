# Stage 3 Output Comparison Check

檢查目標：比較 `stage3` multitaskbert 的 `local` 與 `submit` 兩種模式產生的 prediction output 差別。

不比較 checkpoint 目錄、檔案 layout、檔名差異；只比較最後輸出的 CSV 結果。

## Generate Outputs

```bash
MODE=submit bash scripts/predict/predict_multitaskbert_for_stage3.sh
MODE=local bash scripts/predict/predict_multitaskbert_for_stage3.sh
```

Expected outputs:

```text
results/predict/stage3/multitaskbert/submit/prediction.csv
results/predict/stage3/multitaskbert/local/prediction.csv
```

## Current Observed Result

Current output comparison:

```text
submit_rows 2000
local_rows 2000
matched_ids 2000
id_mismatch 0
submit_dist {'Clear': 1449, 'Not Clear': 551}
local_dist {'Clear': 1440, 'Not Clear': 560}
changed_labels 35
changed_ratio 1.7500% (35/2000)
changed_by_pair {('Clear', 'Not Clear'): 22, ('Not Clear', 'Clear'): 13}
score_clear mean_abs_diff 0.01569203 max_abs_diff 0.20726257
score_not_clear mean_abs_diff 0.01568271 max_abs_diff 0.20764509
score_misleading mean_abs_diff 0.00016014 max_abs_diff 0.00176477
changed_ids_first20 12040,12143,12179,12222,12227,12433,12578,12588,12590,12614,12686,12763,12942,12948,12981,13197,13230,13258,13285,13377
```

## Decision Rule

主要看 `changed_ratio`、`changed_labels` 與 `changed_by_pair`。如果 label 幾乎不變，只看 score drift；如果 label 有明顯變動，回頭抽查 changed ids 對應的 confidence。
