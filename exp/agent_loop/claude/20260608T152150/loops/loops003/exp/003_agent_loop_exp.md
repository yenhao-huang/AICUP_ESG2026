# Loop 003 — Experiment (promotion gate for loop001 a0)

Promotion of the loop001 "a0 real-only recipe" Stage 1 BERT as the new ST1
baseline. This loop runs the CLAUDE.md mandatory external Claude review and the
integrated weighted re-check required by that review.

## Candidate
- Artifact: `models/loop001_st1_a0_real_only/best_st1.pt` (1.30 GB)
- Recipe: `hfl/chinese-roberta-wwm-ext-large` via
  `core/train/train_bert.py --model large --stage st1` on
  `data/raw_data/vpesg_4k_train_1000.json` (1000 real rows, Yes 814 / No 186),
  no synthetic data. Runtime input = `data` only.

## Standalone ST1 (data-only, from loop001 gate evals)
| split | exp4 baseline | a0 candidate | Δ |
|---|---|---|---|
| val_public (selection) | 0.7873 | 0.803069 | +0.0158 |
| val_test (blind gate) | 0.802383 | 0.810977 | **+0.008594** |

## Integrated cascade weighted re-check (val_test, ST2-4 = exp23_train_json_st2_st3_st4_large)

Command (per arm; symlinked cascade dirs mix ST1 with shared ST2-4):
```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python core/eval/eval_bert.py --model large \
  --model-dir <cascade_dir> --pretrain-model hfl/chinese-roberta-wwm-ext-large \
  --data-path data/benchmarks/val_test.json --device cuda:0 \
  --output <out>.json
```

| ST1 used | ST1 | ST2 | ST3 | ST4 | weighted (0.2/0.3/0.35/0.15) |
|---|---|---|---|---|---|
| exp4 (baseline) | 0.8024 | 0.6996 | 0.4635 | 0.3934 | **0.591591** |
| a0 (candidate) | 0.8110 | 0.7242 | 0.4837 | 0.4091 | **0.610112** |
| Δ | +0.0086 | +0.0246 | +0.0202 | +0.0157 | **+0.018521** |

Swapping ONLY the ST1 checkpoint improved every downstream stage (a more
accurate ST1 gate reduces cascade penalties on ST2-4). No stage regresses.

Artifacts:
- `loops/loops003/exp/integrated_baseline_st1exp4_val_test.json`
- `loops/loops003/exp/integrated_candidate_st1a0_val_test.json`
- `loops/loops003/exp/cascade_dirs/{baseline_st1exp4,candidate_st1a0}/` (symlinks)
- External review: `loops/loops003/reflect/external_claude_review_a0_promote.md`

## Gate-check table
| gate item | requirement | result | pass |
|---|---|---|---|
| blind val_test ST1 > BASE_GATE | > 0.802383 | 0.810977 (+0.0086) | ✅ |
| disciplined selection | val_public only, gate touched once | yes | ✅ |
| runtime data-only | input = data | data-only fwd pass; no synthesis | ✅ |
| integrated weighted ≥ baseline | ≥ 0.591591 | 0.610112 (+0.0185) | ✅ |
| no stage regresses | ST1-4 Δ ≥ 0 | all positive | ✅ |
| external Claude review stored | required | stored (conditional support-promote) | ✅ |
| methods.md + state updated same session | required | done | ✅ |
| test.json retired | required | documented in methods.md + state | ✅ |
