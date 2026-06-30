#!/usr/bin/env bash
# Driver: run every Stage 4 variant in sequence on the current SET, then summarize.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$HERE/run_vanilla.sh"
bash "$HERE/run_add_context.sh"
bash "$HERE/run_add_promise.sh"
bash "$HERE/run_add_image.sh"
bash "$HERE/run_all.sh"
echo
echo "=== Stage 4 variant summary (SET=${SET:-100}, BACKEND=${BACKEND:-codex}) ==="
"${PY:-/workspace/esg_contest/.venv/bin/python}" "$HERE/summarize.py" --set "${SET:-100}" --backend "${BACKEND:-codex}"
