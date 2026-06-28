# File Tree Spec

Agents must read this file before creating, moving, or deleting project files. Follow this tree unless the user explicitly asks for a different structure.

```text
AICUP_ESG2026/
├── core/                         # Application code
│   ├── service/                  # Business logic, pipelines, data processing, model training, prediction
│   │   └── data/                 # Data service code, including materialization and synthesis implementations
│   └── api/                      # API, CLI, routes, and adapters
├── data/                         # Project data inputs used by pipelines
│   ├── raw_data/                 # User-provided raw data
│   ├── externel_data/            # Simulated or external generated data
│   └── synthesis_data/           # Intentional checked/staged training inputs only
├── lib/                          # Shared utilities that are reused across domains
├── test/                         # Tests
├── scripts/                      # Thin shell wrappers or orchestration entrypoints only
├── configs/                      # Config files and environment templates
├── ui/                           # Frontend or interface code
├── exp/                          # Experiments and research notes
├── results/                      # Evaluation outputs and generated artifacts
│   └── reproduce_inputs/         # Default materialized training inputs for reproduce runs
├── logs/                         # Runtime logs
├── external/                     # Third-party service wiring
└── docs/                         # Project documentation
```

## Placement Rules

- Put Python implementation code under `core/service/` or `core/api/`.
- Put data materialization, synthesis, and transformation logic under `core/service/data/`.
- Keep `scripts/` for thin shell wrappers or command orchestration only; do not place core Python implementation there.
- Put reusable helpers in `lib/` only when they are shared across multiple service domains.
- Put generated outputs under `results/` and runtime logs under `logs/`.
- Put materialized reproduce inputs under `results/reproduce_inputs/` by default.
- Write to `data/synthesis_data/` only when the user explicitly asks for data placement or the command uses an explicit flag such as `--output-root data`.
- Put methodology, file tree, and process documentation under `docs/`.
