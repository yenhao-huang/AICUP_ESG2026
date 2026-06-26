# Stage 2 Output Comparison Check

檢查目標：比較 `stage2` 的 `local` 與 `submit` 兩種模式產生的 prediction output 差別。

不比較 checkpoint 目錄、檔案 layout、檔名差異；只比較最後輸出的 CSV 結果。

## Generate Outputs

```bash
MODE=submit bash scripts/predict/predict_ensemble_model_for_stage2.sh
MODE=local bash scripts/predict/predict_ensemble_model_for_stage2.sh
```

Expected outputs:

```text
results/predict/stage2/ensemble/submit/softvote.csv
results/predict/stage2/ensemble/local/softvote.csv
```

## Current Observed Result

Current output comparison:

```text
submit_rows 2000
local_rows 2000
matched_ids 2000
id_mismatch 0
submit_dist {'No': 299, 'Yes': 1701}
local_dist {'No': 296, 'Yes': 1704}
changed_labels 3
changed_ratio 0.1500% (3/2000)
changed_by_pair {('No', 'Yes'): 3}
score_yes mean_abs_diff 0.00109571 max_abs_diff 0.11008665
score_no mean_abs_diff 0.00109571 max_abs_diff 0.11008662
changed_ids 13233,13249,13916
```

## Decision Rule

主要看 `changed_ratio`、`changed_labels` 與 `changed_by_pair`。如果 label 幾乎不變，只看 score drift；如果 label 有明顯變動，回頭抽查 changed ids 對應的 confidence。
