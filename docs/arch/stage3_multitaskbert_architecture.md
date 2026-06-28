# Stage 3 Multitask BERT Architecture

This note records where the Stage 3 multitask BERT architecture and prediction
code live, with source line numbers.

## Prediction Entrypoint

Default wrapper:

```bash
bash scripts/predict/predict_multitaskbert_for_stage3.sh
```

Relevant lines:

- `scripts/predict/predict_multitaskbert_for_stage3.sh:21-31`: default env vars, including `DATA`, `PREDICTOR`, `MODEL`, `MAX_LEN`, `BATCH_SIZE`, `DEVICE`, `NC_TAU`.
- `scripts/predict/predict_multitaskbert_for_stage3.sh:34-50`: checkpoint selection. `MODE=submit` uses `models/submission/stage3/w0_2_0_3_0_5/best_multitask_st3.pt`; `MODE=local` uses the local multitask checkpoint.
- `scripts/predict/predict_multitaskbert_for_stage3.sh:63-72`: calls `core/service/predict/stage3/pred_by_multitask.py` with `--data`, `--finetune-path`, `--output`, `--model`, `--max-len`, `--batch-size`, `--device`, and `--run-id`.
- `scripts/predict/predict_multitaskbert_for_stage3.sh:79-80`: optional `NC_TAU` threshold passed as `--nc-tau`.
- `scripts/predict/predict_multitaskbert_for_stage3.sh:83-90`: runtime logging. It explicitly reports `stage2_gate=disabled`.

## Prediction Architecture Class

File:

```text
core/service/predict/stage3/pred_by_multitask.py
```

Architecture class:

```text
MultiTaskBertClassifier
```

Relevant lines:

- `core/service/predict/stage3/pred_by_multitask.py:78-93`: defines `MultiTaskBertClassifier`, a shared `AutoModel` BERT encoder plus `head_st1`, `head_st2`, and `head_st3`.
- `core/service/predict/stage3/pred_by_multitask.py:95-98`: forward pass uses `[CLS]` from `out.last_hidden_state[:, 0]`, applies dropout, and returns only `head_st3(pooled)` for prediction.

Copied code:

```python
class MultiTaskBertClassifier(nn.Module):
    """Shared BERT encoder with ST1, ST2, and ST3 classification heads."""

    def __init__(self, model_name: str, st3_classes: int, dropout: float = 0.1, local_files_only: bool = True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            _fast_init=False,
            local_files_only=local_files_only,
        )
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_st1 = nn.Linear(hidden, ST1_CLASSES)
        self.head_st2 = nn.Linear(hidden, ST2_CLASSES)
        self.head_st3 = nn.Linear(hidden, st3_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        return self.head_st3(pooled)
```

Important behavior:

- The prediction class instantiates all three heads so full multitask checkpoints can load.
- The prediction forward only emits Stage 3 logits. Stage 1 and Stage 2 heads are present for checkpoint compatibility, not used for output.
- The predictor does not apply cascade gating; every input row receives an `evidence_quality` prediction.

## Prediction Data Flow

Relevant lines in `core/service/predict/stage3/pred_by_multitask.py`:

- `53-75`: `TextDataset` tokenizes each row's `data` field and returns `id`, `input_ids`, and `attention_mask`.
- `126-140`: `read_state_dict()` loads the checkpoint. If the checkpoint is an exported ST3-compatible checkpoint with `classifier.weight`, it remaps that to `head_st3.weight`.
- `143-151`: `st3_class_count()` infers whether the ST3 head has 2 classes or 3 classes from `head_st3.weight`.
- `154-158`: `load_multitask_checkpoint()` loads weights with `strict=False`, but requires BERT and ST3 weights.
- `172-179`: `decide_label()` chooses the predicted class. If `--nc-tau` is set, class 1 (`Not Clear`) can be selected by threshold.
- `182-225`: `predict_rows()` loads the tokenizer/model/checkpoint, runs batches, softmaxes ST3 logits, maps class ids to labels, and emits CSV rows.
- `237-294`: CLI `main()` parses arguments, loads rows, optionally applies `--limit`, calls `predict_rows()`, writes CSV, and prints a JSON summary.

Output fields are defined through `schema.OUTPUT_COLUMNS` and each prediction row includes:

- `id`
- `evidence_quality`
- `evidence_quality_raw`
- `evidence_quality_source=bert_multitask`
- `evidence_quality_reason`, containing per-class scores

## Training Architecture Class

File:

```text
core/service/train/train_multitaskbert_stage3.py
```

Architecture class:

```text
MultiTaskBertST3
```

Relevant lines:

- `73-85`: label maps and task metadata. ST1 has 2 classes, ST2 has 2 classes, ST3 defaults to 3 classes: `Clear`, `Not Clear`, `Misleading`.
- `104-119`: `build_multitask_samples()` maps each row into labels for ST1/ST2/ST3 and masks non-applicable labels with task-specific ignore indices.
- `122-141`: `MTDataset` tokenizes `data` and returns `input_ids`, `attention_mask`, and task labels.
- `146-156`: `MultiTaskBertST3` defines a shared `AutoModel` encoder and three linear heads: `head_st1`, `head_st2`, `head_st3`.
- `158-162`: forward pass uses `[CLS]`, dropout, and returns logits for requested tasks as `{task: logits}`.

Copied code:

```python
class MultiTaskBertST3(nn.Module):
    def __init__(self, pretrain_model, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrain_model)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # head names chosen so head_st3 -> classifier remap is trivial for export
        self.head_st1 = nn.Linear(hidden, NUM_LABELS["st1"])
        self.head_st2 = nn.Linear(hidden, NUM_LABELS["st2"])
        self.head_st3 = nn.Linear(hidden, NUM_LABELS["st3"])
        self._heads = {"st1": self.head_st1, "st2": self.head_st2, "st3": self.head_st3}

    def forward(self, input_ids, attention_mask, tasks=TASKS):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]          # [CLS], matches BertClassifier
        pooled = self.dropout(pooled)
        return {t: self._heads[t](pooled) for t in tasks}
```

- `191-209`: `TaskLoss` wraps CE/weighted CE/manual CE and skips batches where all labels for a task are ignored.
- `214-239`: `evaluate()` computes per-task macro-F1 while ignoring masked labels.
- `244-254`: `export_best_st3()` writes a BERT-classifier-compatible ST3 checkpoint by mapping `head_st3` to `classifier`.
- `294-302`: optional `--st3-drop-misleading` mutates ST3 into a 2-class head.
- `373-383`: training instantiates `MultiTaskBertST3` and defines `best_multitask_st3.pt` plus exported `best_st3.pt` checkpoint paths.
- `390-399`: training loop computes logits for all tasks and optimizes weighted sum of task losses.
- `409-420`: best checkpoint is selected by Stage 3 validation F1 and saved as both full multitask and ST3-compatible checkpoints.

## Architecture Summary

```text
input data text
  -> AutoTokenizer
  -> shared BERT encoder
  -> [CLS] hidden state
  -> dropout
  -> head_st1: promise_status logits
  -> head_st2: evidence_status logits
  -> head_st3: evidence_quality logits
```

Training uses all three heads and a weighted multitask loss:

```text
loss = lambda_st1 * loss_st1 + lambda_st2 * loss_st2 + lambda_st3 * loss_st3
```

Prediction loads the same shared encoder and heads, then returns only Stage 3
`evidence_quality` from `head_st3`.
