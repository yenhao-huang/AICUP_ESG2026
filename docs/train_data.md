# Training Data

## Stage 1

Stage 1 uses `promise_status` as the training label. The current ensemble
dataset is:

```text
data/synthesis_data/stage1/a3_b1_add_val.json
```

This dataset has 2,500 records:

| Source | ID range / pattern | Records | Label distribution | Notes |
| --- | --- | ---: | --- | --- |
| Original train data | `10001`-`11000` | 1,000 | `Yes=814`, `No=186` | Projected from `data/raw_data/vpesg_4k_train_1000.json` to the Stage 1 fields `id`, `data`, and `promise_status`. |
| Original validation data | `11001`-`12000` | 1,000 | `Yes=813`, `No=187` | Copied from `data/raw_data/vpesg4k_val_1000.json`; these records match the source file exactly. |
| A3 synthetic positives | `syn_a3_*` | 500 | `Yes=500`, `No=0` | Synthetic commitment-style positive examples generated from train-set parent records. |

Overall label distribution:

```text
total = 2500
Yes   = 2127
No    = 373
```

The A3 synthetic records are all positive examples. Their parent IDs come from
the original train range `10001`-`11000`; the 500 synthetic records are sampled
from 300 unique parent records, so some parents appear twice through separate
synthetic examples.

### How to generate

The Stage 1 synthetic-data design comes from the Claude agent-loop records under:

```text
exp/agent_loop/claude/20260608T152150/loops/
```

The relevant loop is `loops001`, which tested offline synthetic-data
augmentation for the Stage 1 BERT classifier. The runtime model path stayed
strictly data-only: synthetic generation could use extra annotation fields
offline, but training/evaluation/inference consumed only the `data` text plus the
training target.

The loop defined two axes:

| Axis | Values | Purpose |
| --- | --- | --- |
| Synthesis source | `A0`, `A1`, `A2`, `A3`, `A4` | Compare real-only training against several offline synthetic sources. |
| Synthetic mix weight | `B1=0.5x`, `B2=1.0x`, `B3=2.0x` | Add synthetic rows relative to the 1,000 real Stage 1 training rows. |

The synthesis sources were:

| Source | Description | Label behavior | Data-use boundary |
| --- | --- | --- | --- |
| `A0_real_only` | Control arm with no synthetic rows. | Uses original `promise_status`. | No synthetic data. |
| `A1_data_only` | LLM paraphrases existing `data` text. | Carries the original row's `promise_status`. | Uses only `data` text for synthesis. |
| `A2_promise_derived` | LLM or deterministic templates rewrite `promise_string` spans into standalone commitment sentences. | All `Yes`. | `promise_string` is offline-only and is not written into runtime inputs. |
| `A3_data_plus_promise` | LLM combines real `data` context with the row's `promise_string` span into a clear commitment-style sentence. | All `Yes`. | `promise_string` is offline-only; emitted rows are projected to `id`, `data`, `promise_status`. |
| `A4_pdf_derived` | Reuses a pre-materialized PDF/data-derived pool to add hard negatives. | All `No` in the net-new pool. | PDF/page text is offline-only; emitted rows use runtime-safe `data`. |

The loop normalized every candidate mix to one canonical real-training base:

```text
data/raw_data/vpesg_4k_train_1000.json
```

For each source and mix weight, the canonical mix was:

```text
real 1000 rows + synthetic N rows

B1: N =  500
B2: N = 1000
B3: N = 2000
```

When a synthetic pool was smaller than the target `N`, rows were sampled with
replacement and duplicate IDs were suffixed to keep row IDs unique. Every
canonical training row was projected to the Stage 1 schema:

```text
id
data
promise_status
```

The experiment found that the designated `test.json` dev split was contaminated:
all 200 rows overlapped the real train data. Selection was therefore moved to
the clean `val_public.json` split, while `val_test.json` remained the blind
gate.

Key Stage 1 loop result:

| Arm | Meaning | `val_public` Macro-F1 | `val_test` Macro-F1 | Note |
| --- | --- | ---: | ---: | --- |
| `A0_real_only` | Real-only canonical retrain | 0.8031 | 0.8110 | Tied best on clean selection. |
| `A3_B1` | Data + promise synthetic positives at 0.5x | 0.8031 | 0.8234 | Best synthetic arm on blind gate. |
| Exp4 baseline | Previous Stage 1 baseline | 0.7873 | 0.8024 | Reference baseline. |

The reflect decision was `defer`, not a clean promotion, because `A0_real_only`
and `A3_B1` tied on the clean selection split. The evidence supports two
practical conclusions:

- A clean canonical retrain improved over the old exp4 baseline.
- `A3_B1` was the strongest synthetic arm, but the incremental benefit of
synthetic augmentation over real-only retraining was not proven conclusively.

The current `a3_b1_add_val.json` dataset follows the useful `A3_B1` data design:
it keeps the 1,000 original train rows, adds 500 A3 synthetic positive examples,
and then folds in the 1,000 original validation rows for the final ensemble
training pool.

### Ensemble Splits

Stage 1 ensemble train/validation splits are generated from the single 2,500-row
dataset above:

```text
data/synthesis_data/stage1/a3_b1_add_val.json
  -> scripts/data/get_ensemble_model_data_for_stage1.sh
  -> core/service/data/split_train_val_by_class.py
  -> data/ensemble_data/stage1/a3_b1_add_val/seed*/
```

The splitter performs a deterministic stratified split by `promise_status` with
`VAL_RATIO=0.2`. The current ensemble seeds are:

```text
42 7 123 2024 31337
```

Each seed produces:

```text
2000 train records
 500 val records
```

The same split files are mirrored under:

```text
exp/ensemble/stage1/data/ensemble/seed*/
```

The root files `exp/ensemble/stage1/data/a3_b1_add_val.train.json` and
`exp/ensemble/stage1/data/a3_b1_add_val.val.json` correspond to the `seed42`
split.

## Stage 2

Stage 2 uses `evidence_status` as the training label. The current ensemble
dataset is:

```text
data/synthesis_data/stage2/mix_a2_b3_add_val.json
```

This file matches the experiment copy at:

```text
exp/ensemble/stage2/data/mix_a2_b3_add_val.json
```

This dataset has 2,548 records:

| Source | ID range / pattern | Records | Label distribution | Notes |
| --- | --- | ---: | --- | --- |
| Original train data | `10001`-`11000` | 1,000 | `Yes=677`, `No=137`, `""=186` | Projected from `data/raw_data/vpesg_4k_train_1000.json` to the Stage 2 fields `id`, `data`, `esg_type`, `promise_status`, and `evidence_status`. |
| Original validation data | `11001`-`12000` | 1,000 | `Yes=668`, `No=145`, `N/A=187` | Projected from `data/raw_data/vpesg4k_val_1000.json` to the same Stage 2 fields. |
| Stage 2 synthetic data | `syn_st2_*` | 548 | `Yes=274`, `No=274` | Synthetic evidence-status examples mixed into `mix_a2_b3`; all have `promise_status=Yes`. |

Overall `evidence_status` distribution:

```text
total = 2548
Yes   = 1619
No    =  556
N/A   =  187
""    =  186
```

The synthetic records use these ID patterns:

```text
syn_st2_a2yes_* = 274
syn_st2_a2no_*  = 233
syn_st2_a2no2_* =  41
```

Together they are balanced for Stage 2: `Yes=274`, `No=274`. Their parent IDs
come from the original train range `10001`-`11000`; the 548 synthetic records
come from 460 unique parent records, so some parents appear twice through
separate synthetic examples.

The pre-add-val raw mix is:

```text
exp/ensemble/stage2/data/raw_data/mix_a2_b3.json
```

It has 1,548 records: the 1,000 projected train records plus the 548 synthetic
records. The final `mix_a2_b3_add_val.json` appends the 1,000 projected
validation records.

### Ensemble Splits

Stage 2 ensemble train/validation splits are generated from the single
2,548-row dataset above:

```text
data/synthesis_data/stage2/mix_a2_b3_add_val.json
  -> scripts/data/get_ensemble_model_data_for_stage2.sh
  -> core/service/data/split_train_val_by_class.py
  -> data/ensemble_data/stage2/mix_a2_b3_add_val/seed*/
```

The splitter performs a deterministic stratified split by `evidence_status` with
`VAL_RATIO=0.2`. Empty string labels and `N/A` labels are treated as separate
classes by the splitter. The current ensemble seeds are:

```text
42 7 123 2024 31337
```

Each seed produces:

```text
2039 train records
 509 val records
```

The per-seed split distribution is stable:

```text
train: Yes=1295, No=445, N/A=150, ""=149
val:   Yes= 324, No=111, N/A= 37, ""= 37
```

The same split files are mirrored under:

```text
exp/ensemble/stage2/data/ensemble/seed*/
```

The root files `exp/ensemble/stage2/data/mix_a2_b3_add_val.train.json` and
`exp/ensemble/stage2/data/mix_a2_b3_add_val.val.json` correspond to the `seed42`
split.

## Stage 3

TODO
