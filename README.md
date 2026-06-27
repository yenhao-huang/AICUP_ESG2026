# AICUP_ESG2026


## Requiremnets
```bash
pip install -r requirements.txt
```

## Layout

- `core/service/`: business logic and pipelines
- `core/api/`: API, CLI, routes, and adapters
- `data/raw_data/`: user-provided raw data
- `data/externel_data/`: simulated or external generated data
- `data/synthesis_data/`: synthetic data
- `lib/`: shared utilities
- `test/`: tests
- `configs/`: config files and environment templates
- `ui/`: frontend or interface code
- `exp/`: experiments and research notes
- `exp/agent_loop`: the workspace agent runs experiments 
- `results/`: evaluation outputs
- `logs/`: runtime logs
- `external/`: third-party service wiring
- `docs/`: project documentation

## Architectures

### Predict

#### Stage 1

```mermaid
flowchart TD
    Input["Input ESG Report Text"] --> BertMembers["BERT Ensemble Members"]
    BertMembers --> SoftVote["Soft Voting"]
    SoftVote --> PrimaryLabel["Promise Status Prediction"]
    PrimaryLabel --> FallbackModule["Fallback Module<br/>(conf < k)"]

    Input --> LLM["LLM"]
    LLM --> FallbackLabel["Promise Status Prediction"]
    FallbackLabel --> FallbackModule

    FallbackModule --> Final["Stage 1 Output"]
```

#### Stage 2

```mermaid
flowchart TD
    Input["Input ESG Report Text"] --> BertMembers["BERT Ensemble Members"]
    BertMembers --> SoftVote["Soft Voting"]
    SoftVote --> PrimaryLabel["Evidence Status Prediction"]
    PrimaryLabel --> FallbackModule["Fallback Module<br/>(conf < k)"]

    Input --> LLM["LLM"]
    LLM --> FallbackLabel["Evidence Status Prediction"]
    FallbackLabel --> FallbackModule

    FallbackModule --> Final["Stage 2 Output"]
```

#### Stage 3

```mermaid
flowchart TD
    Input["Input ESG Report Text"] --> MultitaskBert["MultitaskBERT Components"]
    MultitaskBert --> PromiseStatus["Promise Status"]
    MultitaskBert --> EvidenceStatus["Evidence Status"]
    MultitaskBert --> EvidenceQuality["Evidence Quality"]
    EvidenceQuality --> Stage1Gate["Stage 1 Gate Filter"]
    Stage1Gate --> Stage2Gate["Stage 2 Gate Filter"]
    Stage2Gate --> Final["Stage 3 Output"]
```

#### Stage 4

```mermaid
flowchart TD
    Input["Input ESG Report Text"] --> PromptBuilder["Prompt Builder"]
    Prompt["Prompt<br/>(e.g., Boundary Rules)"] --> PromptBuilder
    PromptBuilder --> Reasoner["GPT 5.5"]
    Reasoner --> Candidate["Stage 4 Candidate Output"]
    Candidate --> Gate["Stage 1 Gate Filter"]
    Gate --> Final["Stage 4 Output"]
```

### Train

#### Stage 1

```mermaid
flowchart TD
    Raw["Stage 1 Training Data"] --> Subsampling["Subsampling"]
    Subsampling --> Data1["Ensemble Data 1"]
    Subsampling --> Data2["Ensemble Data 2"]
    Subsampling --> Data3["Ensemble Data 3"]
    Subsampling --> Data4["Ensemble Data 4"]
    Subsampling --> Data5["Ensemble Data 5"]
    Data1 --> Train1["BERT Training 1"]
    Data2 --> Train2["BERT Training 2"]
    Data3 --> Train3["BERT Training 3"]
    Data4 --> Train4["BERT Training 4"]
    Data5 --> Train5["BERT Training 5"]
    Train1 --> Ensemble["BERT Ensemble Members"]
    Train2 --> Ensemble
    Train3 --> Ensemble
    Train4 --> Ensemble
    Train5 --> Ensemble
```

#### Stage 2

```mermaid
flowchart TD
    Raw["Stage 2 Training Data"] --> Subsampling["Subsampling"]
    Subsampling --> Data1["Ensemble Data 1"]
    Subsampling --> Data2["Ensemble Data 2"]
    Subsampling --> Data3["Ensemble Data 3"]
    Subsampling --> Data4["Ensemble Data 4"]
    Subsampling --> Data5["Ensemble Data 5"]
    Data1 --> Train1["BERT Training 1"]
    Data2 --> Train2["BERT Training 2"]
    Data3 --> Train3["BERT Training 3"]
    Data4 --> Train4["BERT Training 4"]
    Data5 --> Train5["BERT Training 5"]
    Train1 --> Ensemble["BERT Ensemble Members"]
    Train2 --> Ensemble
    Train3 --> Ensemble
    Train4 --> Ensemble
    Train5 --> Ensemble
```

#### Stage 3

```mermaid
flowchart TD
    Raw["Stage 3 Training Data"] --> Encoder["Shared BERT Encoder"]
    Encoder --> Classifier["Different Classifier"]
```

#### Fallback Model

```mermaid
flowchart TD
    InstructionData["Stage 1 / 2 Instruction Data"] --> Finetune["Finetune LLM QLoRA"]
    Finetune --> Adapter["LLM Adapter"]
```

## Methodologies

要介紹 input/output

### Stage1

**Synthesis Data**


**Ensemble Data Collection**
```bash
bash scripts/data/get_ensemble_model_data_for_stage1.sh
```

**Train Ensemble Models**
```bash
bash scripts/train/train_ensemble_models_for_stage1.sh
```

**Predict**
```
bash scripts/predict/predict_ensemble_model_for_stage1.sh
```

### Stage2

**Ensemble Data Collection**
```bash
bash scripts/data/get_ensemble_model_data_for_stage2.sh
```

**Train Ensemble Models**
```bash
bash scripts/train/train_ensemble_models_for_stage2.sh
```

**Predict**
```bash
bash scripts/predict/predict_ensemble_model_for_stage2.sh
```

### Stage3
**Ensemble Data Collection**
```bash
bash scripts/data/get_multitask_model_for_stage3.sh
```

**Train Ensemble Models**
```bash
bash scripts/train/train_multitaskbert_for_stage3.sh
```

**Predict**
```bash
bash scripts/predict/predict_multitaskbert_for_stage3.sh
```

### Stage4

**Predict**
```bash
bash scripts/predict/predict_codex_for_stage4.sh
```


### Fallback Model: Gemma4

**Ensemble Data Collection**
```bash
bash scripts/data/get_gemma_data_for_stage12.sh
```

**Train Gemma12b**
```bash
bash scripts/train/train_gemma_for_stage12.sh
```

**Predict**
```bash
bash scripts/predict/predict_gemma_fallback_model.sh
```

### Fallback Model: GPT5.5


## Which insights does the agent generate?

### Design Synthesis Data for Stage1
exp/agent_loop/claude/20260608T152150/loops/loops02

### Find Best Prompt for Stage4
exp/agent_loop/claude/20260609T172829/loops/loops001/


## Harness Engineering

### 設定專案規則
**AGENT.md/CLAUDE.md**
* 定義解決的任務
* 介紹資料集
* 介紹評分方式
* 限制每個 


### 設定進度檔
* init.sh
* Progress.md
* Decisions.md
* Task.md

Use `init.sh` as the checklist for agent handoff.

```bash
./init.sh start
./init.sh finish
```

```mermaid
flowchart TD
    A[Run ./init.sh start<br/>Read Task.md, Progress.md, Decisions.md<br/>Inspect git status]
    A --> B[Do assigned work]
    B --> C[Run validation]
    C --> D[Run ./init.sh finish<br/>Update Progress.md and Decisions.md<br/>Inspect git status]
    D --> E[Create checkpoint commit when appropriate]
```

### 設定工具
