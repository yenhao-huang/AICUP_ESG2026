#!/usr/bin/env bash
# Probe: add_context_2 gated v3 — <data-prompt> primary, <same-page-context> only to
# understand this promise's semantics; forward-commitment thrust -> between, 2024
# actuals count as progress evidence only. DATA-USE: same-page-context exceeds the
# `data`-only default; probe-only, not promotable.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_variant add_context2_gated_v3 add-context-2-gated-v3.txt --add-context --context-mode all
