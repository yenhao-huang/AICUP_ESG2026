# ensemble (formerly `bagging`)

Renamed from `bagging` → `ensemble`: the method here is **not** bootstrap
aggregating (no sampling-with-replacement). Instead, the ONE raw dataset of each
stage is cut into **5 different stratified train/val partitions** (one per seed),
and **5 BERT members** are trained — member `<seed>` on split `<seed>`, each with
its own RNG `--seed`. So every member sees a different data slice *and* a
different seed → an ensemble with both data- and seed-diversity.

## Per stage

| stage | raw input | stratify label | trainer | model |
|-------|-----------|----------------|---------|-------|
| stage1 | `data/a3_b1_add_val.json` (2500) | `promise_status` | `train_bert.py --stage st1` | `best_st1.pt` |
| stage2 | `data/mix_a2_b3_add_val.json` (2548) | `evidence_status` | `train_bert.py --stage st2` | `best_st2.pt` |
| stage3 | `data/vpesg_4k_train_1000_add_val.json` (2000) | `evidence_quality` (Misleading forced→val) | `train_multitaskbert_stage3.py` | `best_multitask_st3.pt` |

Seeds: `42 7 123 2024 31337` (edit at the top of each script).

## Run (from repo root `/workspace/esg_contest`)

```bash
# 1. make the 5 splits  ->  stageN/data/ensemble/seed<S>/
bash exp/integrated_stage_predictions/0615/ensemble/stage1/ensemble/make_splits.sh
# 2. train the 5 members ->  /models/ensemble_*_seed<S>/  + stageN/ensemble/train_results/
bash exp/integrated_stage_predictions/0615/ensemble/stage1/ensemble/train.sh
```

(same two commands for `stage2` / `stage3`). `make_splits.sh` is CPU-only and
idempotent; `train.sh` needs a GPU (`GPU=<id>` env to pick the device). Members
are independent — combine their predictions (vote / mean-prob) at inference.
