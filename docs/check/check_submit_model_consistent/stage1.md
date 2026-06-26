# Stage 1 Submit Model Consistency

檢查 `check_submit_model_consistent/submit/stage1` 與 `results/predict/stage1/ensemble/submit` 是否對齊。

可執行檢查：

```bash
python3 docs/check/check_submit_model_consistent/check_submit_outputs.py --stage stage1
```

Current result:

```text
check_path docs/check/check_submit_model_consistent/submit/stage1/softvote_raw.csv
results_path results/predict/stage1/ensemble/submit/softvote.csv
check_rows 2000
results_rows 2000
matched_ids 2000
id_mismatch_count 0
prediction_columns promise_status,score_yes,score_no,raw_prediction
prediction_aligned true
prediction_mismatch_rows 0
full_csv_exact_aligned false
full_csv_mismatch_columns {'finetune_path': 2000, 'run_id': 2000}
```

判斷：prediction 對齊。full CSV 不完全一致只來自 metadata 欄位 `finetune_path` 與 `run_id`，不影響 `promise_status`、score 與 raw prediction。
