#!/usr/bin/env bash
# Run all five Stage 3 ablations on the 100-row smoke set, in sequence.
# Quick wiring check: LIMIT=3 ./run_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in run_add_evidence_string run_add_image run_add_evidence_promise run_add_context run_add_context_window; do
  bash "$HERE/$s.sh"
done
echo "all experiments done -> $(cd "$HERE/../.." && pwd)/preds/"
