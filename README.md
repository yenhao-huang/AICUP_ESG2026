# AICUP_ESG2026

Project initialized with the standard project tree.

## Layout

- `core/service/`: business logic and pipelines
- `core/api/`: API, CLI, routes, and adapters
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
    A[Start agent work] --> B[Run ./init.sh start]
    B --> C[Read Task.md]
    C --> D[Read Progress.md]
    D --> E[Read Decisions.md]
    E --> F[Inspect git status]
    F --> G[Do the assigned work]
    G --> H[Run relevant validation]
    H --> I[Run ./init.sh finish]
    I --> J[Update Progress.md]
    J --> K[Record major decisions in Decisions.md]
    K --> L[Inspect git status]
    L --> M[Create checkpoint commit when appropriate]
```
