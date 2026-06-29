Optimize ESG Stage 1 (Promise Identification, output `Yes`/`No`, metric Macro-F1).

Pipeline goal: a BERT-style classifier trained on synthetic + real data, with
low-confidence predictions optionally escalated to an LLM-RAG fallback
(endpoint http://192.168.1.79:3134, RAG over data-only retrieved examples).
The orchestrator decides when to introduce the LLM fallback; loop 1 establishes
a clean BERT + synthetic-data method first.

Datasets (exact paths):
- Train (real):   data/raw_data/vpesg_4k_train_1000.json  (n=1000; Yes 814 / No 186)
- PDF sources for synthesis: data/generated/raw_doc_table.jsonl, data/generated/raw_page_table.jsonl
- Tune / dev split (threshold + hyperparameter search ONLY here): data/benchmarks/test.json (n=200; Yes 162 / No 38)
- Test / report split: data/benchmarks/val_public.json (n=500; Yes 410 / No 90)
- FINAL blind gate (must beat baseline; never tune on this): data/benchmarks/val_test.json (n=500; Yes 403 / No 97)

Baseline: models/exp4_optimize2_highconf_yes_balanced_no_large
(hfl/chinese-roberta-wwm-ext-large). Its ST1 Macro-F1 on the OLD val.json was
0.7950. The TRUE gate baseline = this model's ST1 Macro-F1 measured on
data/benchmarks/val_test.json — measure it once in loop 1 and reuse as the gate
baseline for all loops.

Optimization directions allowed: (1) synthetic dataset generation (from `data`,
from promise_string, from data+promise, or from the data's source PDF content),
(2) mixing synthetic + real data with chosen weights, (3) model architecture,
(4) loss function, (5) hyperparameters, (6) confidence-threshold tuning,
(7) LLM fallback prompt design.

Harness rules:
- NO rule/keyword/rule-based postprocess of any kind.
- LLM fallback (low-confidence BERT prediction handed to the LLM) IS allowed.
- A method is accepted only if it beats the baseline Macro-F1 on the blind gate
  data/benchmarks/val_test.json.

Data-use rules (CLAUDE.md, with one explicit user override):
- RUNTIME / inference input is `data` ONLY. Never feed promise_string,
  evidence_string, extracted promise/evidence, ground-truth labels, or any
  derived annotation field into the model, prompt, RAG retrieval, or features at
  inference time.
- OVERRIDE (user-approved): promise_string MAY be used OFFLINE only, solely to
  generate synthetic training data. It must never appear in the runtime path.
  Any artifact that uses promise_string at runtime is INVALID.
- Ground-truth labels may be used only for offline scoring, never as input.
- Tune thresholds/hyperparameters only on the dev split (data/benchmarks/test.json);
  never on val_public.json or val_test.json.

Relevant existing code (reuse, don't reinvent): core/train/train_bert.py,
core/eval/eval_bert.py, core/e2e/stage1.py, configs/train/bert.yml,
configs/eval/roberta.yml, core/human/predict/stage1/pred_by_bert_codex_rag.py.
