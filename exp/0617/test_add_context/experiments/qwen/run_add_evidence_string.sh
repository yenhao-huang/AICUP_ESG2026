#!/usr/bin/env bash
# Ablation: inject <evidence-string> only (no same-page-context, no image).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_exp add_evidence_string --no-add-context --add-evidence-string
