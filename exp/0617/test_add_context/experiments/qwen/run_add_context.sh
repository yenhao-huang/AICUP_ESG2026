#!/usr/bin/env bash
# Ablation: same-page-content, mode=all (whole matched page).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_exp add_context --add-context --context-mode all
