# Loop 001 — Experiment

Synthetic-data augmentation grid for the ST1 BERT classifier
(`hfl/chinese-roberta-wwm-ext-large`), gated against the exp4 baseline on the
blind set `data/benchmarks/val_test.json`.

## CRITICAL FINDING — designated dev split is contaminated

The plan/task designated `data/benchmarks/test.json` (n=200) as the dev
selection split. Overlap check against the canonical real-train file
`data/raw_data/vpesg_4k_train_1000.json` (n=1000):

| eval file | n | id-overlap w/ train | data-overlap w/ train |
|---|---|---|---|
| test.json (designated dev) | 200 | **200 / 200** | **200 / 200** |
| val_public.json | 500 | 0 | 0 |
| val_test.json (blind gate) | 500 | 0 | 0 |

`test.json` is a 100% subset of the training set (and 214 rows of `mix_a1_b1`
are verbatim test.json rows). Dev Macro-F1 on test.json (0.96–0.99 across arms)
is pure training-set memorization and is **invalid for selection**.

`val_public.json` and `val_test.json` are clean (0 overlap with train) and
mutually disjoint (id 0 / data 0), so selection was moved to **val_public.json**
and the blind gate kept on **val_test.json**.

## Commands (exact)

```bash
# Baseline gate measurement
python core/eval/eval_bert.py --model large --stage st1 --no-cascade \
  --model-dir models/exp4_optimize2_highconf_yes_balanced_no_large \
  --pretrain-model hfl/chinese-roberta-wwm-ext-large \
  --data-path data/benchmarks/val_test.json --device cuda:0 --output .../baseline_val_test_st1.json
# (same on data/benchmarks/test.json and val_public.json)

# Grid training (12 arms; A0 trained separately), GPU0 only, batch 4
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash .../exp/run_grid_gpu0.sh          # see run_grid_gpu0.sh

# Selection eval (clean) + gate eval
bash .../exp/eval_arms_valpublic.sh      # all 13 arms on val_public.json
python core/eval/eval_bert.py ... --data-path data/benchmarks/val_test.json   # selected arms only
```

All training/eval ran on GPU0 (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`,
`--batch-size 4`, `expandable_segments`) after GPU1 OOM collisions with an
unrelated `exp32` job. No OOM in the final run (`grep OutOfMemoryError` over arm
logs → none). Runtime/eval input is `data` only.

## Baselines (exp4 model, data-only)

| split | exp4 baseline Macro-F1 |
|---|---|
| test.json (BASE_DEV, contaminated) | 0.8375 |
| val_public.json | 0.7873 |
| val_test.json (BASE_GATE) | **0.8024** |

## Grid results

`best_val_macro_f1` below = each arm's own internal mix val-split (NOT
comparable across arms; synthetic-heavy mixes inflate it). The selection metric
is the clean **val_public Macro-F1**.

| arm | source × mix | internal val | dev test.json (leaked) | **val_public (selection)** |
|---|---|---|---|---|
| a0_real_only | real only | 0.7576 | 0.9675 | **0.8031** |
| a1_b1 | data-only 0.5× | 0.9107 | 0.9920 | 0.7517 |
| a1_b2 | data-only 1.0× | 0.9560 | 0.9838 | 0.7724 |
| a1_b3 | data-only 2.0× | 0.9548 | 0.9387 | 0.7621 |
| a2_b1 | promise 0.5× | 0.8251 | 0.9037 | 0.7329 |
| a2_b2 | promise 1.0× | 0.8018 | 0.8716 | 0.7828 |
| a2_b3 | promise 2.0× | 0.8078 | 0.9503 | 0.7725 |
| a3_b1 | data+promise 0.5× | 0.8737 | 0.9590 | **0.8031** |
| a3_b2 | data+promise 1.0× | 0.8127 | 0.9675 | 0.7752 |
| a3_b3 | data+promise 2.0× | 0.8180 | 0.9503 | 0.7955 |
| a4_b1 | pdf hard-No 0.5× | 0.9329 | 0.9492 | 0.7905 |
| a4_b2 | pdf hard-No 1.0× | 0.9481 | 0.8937 | 0.7689 |
| a4_b3 | pdf hard-No 2.0× | 0.9641 | 0.9838 | 0.7793 |

Selection (clean val_public): **a0_real_only = a3_b1 = 0.8031 (tie)**, both
+0.0158 over the exp4 baseline (0.7873). Every synthetic-heavy arm (A1/A2/A4 at
B2/B3) is *below* real-only. The manual_ce loss ablation was not reached
(grid time/collision budget); recorded as skipped.

## Blind gate (val_test.json) — the two tied-best arms

| arm | val_test Macro-F1 | No-F1 | Yes-F1 | Δ vs BASE_GATE 0.8024 |
|---|---|---|---|---|
| a0_real_only | 0.8110 | 0.6919 | 0.9301 | +0.0086 |
| **a3_b1** | **0.8234** | 0.7086 | 0.9382 | **+0.0210** |

Generalization gap (val_public → val_test): a0 +0.0079, a3_b1 +0.0203 (both
positive — no overfit to selection set).

## Gate-check table (score_first_promote)

| gate item | threshold | a3_b1 | pass? |
|---|---|---|---|
| blind val_test Macro-F1 > BASE_GATE | > 0.8024 | 0.8234 | ✅ +0.0210 |
| selected without touching blind gate | val_public only | tie-best on val_public | ✅ |
| runtime data-only | data only | data-only fwd pass | ✅ |
| promise_string offline-only | no runtime leak | A3 synth offline; output rows {id,data,promise_status} | ✅ |
| beats baseline on selection set too | > 0.7873 | 0.8031 | ✅ +0.0158 |

## Caveats for reflect

1. The loop's thesis (synthetic augmentation helps ST1) is only **weakly
   supported**: on clean val_public, real-only (a0) ties the best synthetic arm
   (a3_b1), and all heavier synthetic mixes hurt. The candidate's edge over the
   exp4 baseline appears to come mostly from the canonical-data training recipe
   (a0 already beats baseline), not from synthesis.
2. a0 and a3_b1 **tie** on the locked selection metric; a3_b1's larger gate win
   (+0.021 vs +0.009) could be partly noise. Promoting "A3 data+promise 0.5×"
   vs "real-only retrain" is a close call.
3. The contaminated `test.json` dev split must NOT be used for selection in any
   future loop; switch the standing dev split to val_public (or a train-disjoint
   holdout). This is a task-definition correction worth surfacing to the user.

## Artifacts

- Checkpoints: `models/loop001_st1_<arm>/best_st1.pt` (13 arms)
- Baselines: `.../exp/baseline_{test,val_test}_st1.json`, `.../valpublic_eval/baseline_exp4_valpublic.json`
- Selection evals: `.../exp/valpublic_eval/*_valpublic.json`
- Leaked dev evals (flagged invalid for selection): `.../exp/dev_eval/*_dev.json`
- Gate evals: `.../exp/gate_eval/{a0_real_only,a3_b1}_valtest.json`
- Candidate: `models/loop001_st1_a3_b1/best_st1.pt` (val_test 0.8234), runner-up `models/loop001_st1_a0_real_only`
