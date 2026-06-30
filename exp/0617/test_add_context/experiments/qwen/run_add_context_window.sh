#!/usr/bin/env bash
# Ablation: same-page-content, mode=hit_exact_window_norm_window only.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_exp add_context_window --add-context --context-mode hit_exact_window_norm_window
