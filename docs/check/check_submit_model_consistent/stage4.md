# Stage 4 Submit Model Consistency

檢查 `check_submit_model_consistent/submit/stage4` 與 `results/predict/stage4` 是否對齊。

可執行檢查：

```bash
python3 docs/check/check_submit_model_consistent/check_submit_outputs.py --stage stage4
```

Current result:

```text
check_path docs/check/check_submit_model_consistent/submit/stage4/stage4_codex_predictions_fixed.csv
results_path results/predict/stage4/codex/all_rows/20260627_123651/stage4_codex_predictions.csv
check_rows 2000
results_rows 2000
matched_ids 2000
id_mismatch_count 0
prediction_columns verification_timeline,stage4_flow,stage1_promise_str,stage4_filtered,stage4_raw_timeline,stage4_postprocess_rule,stage4_error
prediction_aligned false
prediction_mismatch_rows 111
full_csv_exact_aligned false
full_csv_mismatch_columns {"verification_timeline": 101, "stage4_raw_timeline": 103}
```

判斷：row 數與 id 對齊，但 prediction 不完全對齊；主要差異在 `verification_timeline` 與 `stage4_raw_timeline`。
