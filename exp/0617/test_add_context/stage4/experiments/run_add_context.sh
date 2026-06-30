#!/usr/bin/env bash
# Probe: + same-page-context (whole matched page OCR text).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_variant add_context add-context.txt --add-context --context-mode all
