#!/usr/bin/env bash
# Probe: add_context_2 gated v4 — v3's rules grafted onto the baseline within_2 rule.
# Restores vanilla's strong "2025/2026 (even with 持續/逐步 framing) -> within_2"
# rule (drops v3's strict guard that over-suppressed within_2 recall on full), and
# makes year-based step 2 take precedence over the intent-first already/between
# logic. Keeps v3's intent-first split for the year-less case + context-use policy.
# DATA-USE: same-page-context exceeds the `data`-only default; probe-only.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_variant add_context2_gated_v4 add-context-2-gated-v4.txt --add-context --context-mode all
