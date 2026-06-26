# Stage 1 Output Comparison Check

檢查目標：比較 `stage1` 的 `local` 與 `submit` 兩種模式產生的 prediction output 差別。

不比較 checkpoint 目錄、檔案 layout、檔名差異；只比較最後輸出的 CSV 結果。

## Generate Outputs

```bash
MODE=submit bash scripts/predict/predict_ensemble_model_for_stage1.sh
MODE=local bash scripts/predict/predict_ensemble_model_for_stage1.sh
```

Expected outputs:

```text
results/predict/stage1/ensemble/submit/softvote.csv
results/predict/stage1/ensemble/local/softvote.csv
```

## Current Observed Result

Current output comparison:

```text
submit_rows 2000
local_rows 2000
matched_ids 2000
id_mismatch 0
submit_dist {'No': 341, 'Yes': 1659}
local_dist {'No': 354, 'Yes': 1646}
changed_labels 27
changed_ratio 1.3500% (27/2000)
changed_by_pair {('No', 'Yes'): 7, ('Yes', 'No'): 20}
score_yes mean_abs_diff 0.02300808 max_abs_diff 0.18210610
score_no mean_abs_diff 0.02300808 max_abs_diff 0.18210610
changed_ids_first20 12059,12061,12105,12359,12362,12457,12490,12582,12587,12647,12709,12816,12820,12830,12894,12955,13098,13169,13272,13313
```

## Decision Rule

主要看 `changed_ratio`、`changed_labels` 與 `changed_by_pair`。如果 label 幾乎不變，只看 score drift；如果 label 有明顯變動，回頭抽查 changed ids 對應的 confidence。
