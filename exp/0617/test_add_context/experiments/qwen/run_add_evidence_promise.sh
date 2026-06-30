#!/usr/bin/env bash
# Ablation: inject <evidence-string> + <promise-string> (no same-page-context).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_exp add_evidence_promise --no-add-context --add-evidence-string --add-promise-string
