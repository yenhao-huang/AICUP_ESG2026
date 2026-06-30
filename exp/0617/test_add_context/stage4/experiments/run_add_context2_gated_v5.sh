#!/usr/bin/env bash
# Probe: add_context_2 gated v5 — same rules as v4, but the decision is restructured
# as a short->long horizon cascade (4 gates: already -> within_2 -> between -> more5,
# stop on first hit) instead of v4's "has-year vs no-year" split. Tests whether the
# explicit short-first sieve changes accuracy. DATA-USE: same-page-context exceeds the
# `data`-only default; probe-only.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_variant add_context2_gated_v5 add-context-2-gated-v5.txt --add-context --context-mode all
