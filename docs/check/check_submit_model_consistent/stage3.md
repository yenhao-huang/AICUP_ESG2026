# Stage 3 Submit Model Consistency

檢查 `check_submit_model_consistent/submit/stage3` 與 `results/predict/stage3/multitaskbert/submit` 是否對齊。

可執行檢查：

```bash
python3 docs/check/check_submit_model_consistent/check_submit_outputs.py --stage stage3
```

Current result:

```text
check_path docs/check/check_submit_model_consistent/submit/stage3/multitask.csv
results_path results/predict/stage3/multitaskbert/submit/prediction.csv
check_rows 2000
results_rows 2000
matched_ids 2000
id_mismatch_count 0
prediction_columns evidence_quality,evidence_quality_raw,evidence_quality_source,evidence_quality_reason
prediction_aligned true
prediction_mismatch_rows 0
full_csv_exact_aligned true
full_csv_mismatch_columns {}
```

判斷：prediction 對齊，full CSV 也完全對齊。
