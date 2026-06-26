# Agent Instructions

Follow the standard project convention:

- Keep application logic in `core/service/`.
- Keep API, CLI, routes, and adapters in `core/api/`.
- Keep shared utilities in `lib/`.
- Keep tests in `test/`.
- Keep configs in `configs/`.
- Keep frontend/interface code in `ui/`.
- Keep experiments in `exp/`.
- Keep outputs in `results/` and logs in `logs/`.
- Do not store models, datasets, secrets, logs, or generated outputs in git.

## Environment

- Use the project-local `.venv/` for all Python commands.
- Run Python as `.venv/bin/python`.
- Install packages with `.venv/bin/python -m pip ...` or `uv pip ...` inside the active `.venv`.
- Do not install packages into system Python or user/global site-packages.
