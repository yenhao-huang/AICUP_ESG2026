#!/usr/bin/env bash
# Baseline: vanilla.txt (= boundary_rules_v4.txt), data-only.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_variant vanilla vanilla.txt --no-add-context
