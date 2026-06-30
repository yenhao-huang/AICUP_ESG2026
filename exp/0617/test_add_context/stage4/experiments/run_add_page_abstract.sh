#!/usr/bin/env bash
# Probe: same-page-context, but the prompt first abstracts the whole page down to
# this promise's targets/years/progress, then judges timeline from that abstract
# (noise-suppression vs raw add_context). Reuses --add-context whole-page injection.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_variant add_page_abstract add-page-abstract.txt --add-context --context-mode all
