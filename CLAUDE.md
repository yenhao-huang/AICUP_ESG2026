# Claude Instructions

Follow `AGENT.md` and the standard project convention for all file placement.

## Environment

- Use the project-local `.venv/` for all Python commands.
- Run Python as `.venv/bin/python`.
- Install packages with `.venv/bin/python -m pip ...` or `uv pip ...` inside the active `.venv`.
- Do not install packages into system Python or user/global site-packages.


## Problem Definition

The competition contains four ESG verification stages. All stages must infer
their outputs from the same allowed raw input field: `data`.

Data-use rules:

- For every stage, the only allowed raw input field is `data`.
- Do not use `evidence_string`, `promise_string`, extracted evidence, extracted promise, ground-truth labels, or any other derived/annotation field as model input, prompt input, rule input, postprocess input, or feature input.
- Stage optimization, harness experiments, prompts, classifiers, and deterministic postprocess rules must be designed as data-only unless the user explicitly changes the problem definition.
- If an existing experiment or artifact used fields other than `data`, mark it as invalid for the current problem definition before comparing or promoting it.

### Task Dependency

```text
All data -> ST1 (promise_status)
              |-- Yes -> ST2 (evidence_status)
              |             |-- Yes -> ST3 (evidence_quality)
              |             `-- No  -> ST3 = ""
              |          ST4 (verification_timeline)
              `-- No  -> ST2 / ST3 / ST4 = ""
```

### Stage 1: Promise Statement Identification

Goal: determine whether the given sentence expresses a clear corporate
commitment to future action.

Output labels:

- `Yes`: the statement contains an explicit commitment.
- `No`: the statement is a general statement and does not contain a commitment.

Metric: Macro-F1.

Examples:

- Promise: `我們承諾在 2030 年前達成碳中和目標`
- Non-promise: `我們重視環境保護的重要性`

### Stage 2: Supporting Evidence Link

Goal: determine whether an identified promise has a concrete implementation
plan or supporting evidence.

Output labels:

- `Yes`: the promise has concrete supporting evidence.
- `No`: the promise lacks concrete supporting evidence.
- `N/A`: the sentence is not a promise, so evidence judgment is not applicable.

Metric: Macro-F1 for semantic relevance judgment.

Example:

- Promise: `推動低碳價值鏈轉型，持續強化供應商節電、減碳、省水及減廢輔導`
- Evidence pattern: `要求訂定中長期減量目標並提出具體行動`
- Expected judgment: evidence supported.

### Stage 3: Clarity Classification

Goal: assess whether the promise is semantically clear and verifiable, and
identify potential greenwashing risk.

Output labels:

- `Clear`: semantically clear and verifiable.
- `Not Clear`: vague or difficult to quantify.
- `Misleading`: potentially misleading statement.
- `N/A`: the sentence is not a promise, so clarity judgment is not applicable.

Metric: Macro-F1 over the four classes.

Practical value: help identify potential corporate greenwashing and improve ESG
report credibility.

### Stage 4: Expected Verification Timeline

Goal: infer the expected completion or verification timing of the promise.

Output labels:

- `Already`: the promise has already been implemented and can be verified in
  the current reporting period.
- `Within 2 years`: short-term target.
- `Between 2 and 5 years`: medium-term target.
- `More than 5 years`: long-term target.
- `N/A`: the sentence is not a promise, so timeline judgment is not applicable.

Metric: Macro-F1 over the five timeline classes.


## Dataset

Dataset name: VeriPromiseESG4K.
Dataset path: ./data/raw_data/<>.json (e.g., vpesg_4k_train_1000.json)

VeriPromiseESG4K is the first large-scale Traditional Chinese sustainability
promise verification annotated dataset. It is built from real ESG reports from
Taiwan 50 Index constituent companies, covering 15 industries.

Dataset characteristics:

- Taiwan leading companies: sourced from Taiwan 50 Index (0050) constituent
  companies, covering real sustainability reports from the top 50 listed
  companies in Taiwan.
- Cross-industry diversity: covers 15 industries, including technology,
  finance, manufacturing, energy, and other sectors, providing diverse industry
  perspectives.
- High-quality annotation: annotated through collaboration between teams from
  National Taipei University and University of Taipei, with multi-stage quality
  control and Krippendorff's Alpha used to ensure annotation consistency.

Dataset scale:

- Dataset name: `VeriPromiseESG4K`
- Full description: the first large-scale Traditional Chinese sustainability
  promise verification dataset.
- Total size: 4,000 high-quality annotated records.
- Data source: Taiwan 50 Index (0050) constituent companies, covering the top
  50 listed companies.
- Industry coverage: 15 industries, including technology, finance,
  manufacturing, and energy.
- Annotation dimensions: four subtasks: promise identification, evidence
  support, clarity assessment, and verification timeline.
- Data split: training set plus test sets, including public and private test
  splits.

Annotation process:

1. Initial annotation
   - Professional annotation team performs initial labeling.
   - Annotation standards and guidelines are established.
   - Annotator training is conducted.
2. Cross-validation
   - Multiple annotators label independently.
   - Inter-annotator agreement is calculated.
   - Annotation disagreements are resolved.
3. Expert review
   - Domain experts perform final review.
   - Quality control and corrections are applied.
   - Dataset is released after review.