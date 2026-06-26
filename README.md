# AICUP_ESG2026

Project initialized with the standard project tree.

## Layout

- `core/service/`: business logic and pipelines
- `core/api/`: API, CLI, routes, and adapters
- `data/raw_data/`: user-provided raw data
- `data/externel_data/`: simulated or external generated data
- `lib/`: shared utilities
- `test/`: tests
- `configs/`: config files and environment templates
- `ui/`: frontend or interface code
- `exp/`: experiments and research notes
- `results/`: evaluation outputs
- `logs/`: runtime logs
- `external/`: third-party service wiring
- `docs/`: project documentation

## Agent Workflow

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
