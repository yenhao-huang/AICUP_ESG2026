#!/usr/bin/env bash
# Probe: all inputs — same-page-context + page image + promise_string.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_variant all all.txt --add-context --context-mode all --add-image --add-promise-string
