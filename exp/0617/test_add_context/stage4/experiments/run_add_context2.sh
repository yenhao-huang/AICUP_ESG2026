#!/usr/bin/env bash
# Probe family: add_context_2 — <data-prompt> is the sole subject; <same-page-context>
# is used ONLY to understand this promise's semantics, then judge timeline. Three
# context-use policies (gated / paraphrase / surgical) share the same 4-step core;
# only the policy header differs, so this is a clean ablation of "how much may
# same-page-content drive the timeline judgment".
#
# DATA-USE NOTE: same-page-context exceeds the CLAUDE.md `data`-only default; this is
# a user-approved probe and must NOT be promoted as a data-only runtime path.
#
# Reuses --add-context --context-mode all (whole matched-page OCR injection), same as
# run_add_context.sh; only the prompt changes between variants.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_variant add_context2_gated    add-context-2-gated.txt    --add-context --context-mode all
run_variant add_context2_para     add-context-2-para.txt     --add-context --context-mode all
run_variant add_context2_surgical add-context-2-surgical.txt --add-context --context-mode all
