#!/usr/bin/env bash
# Probe: add_context_2 gated v6 — same rules as v4, but the decision is a 2-level
# hierarchy: first coarse split short-term vs long-term, then sub-classify
# (short -> already / within_2_years; long -> between / more_than_5_years). Tests the
# hierarchical tree vs v4's year-split and v5's flat short->long cascade.
# DATA-USE: same-page-context exceeds the `data`-only default; probe-only.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_variant add_context2_gated_v6 add-context-2-gated-v6.txt --add-context --context-mode all
