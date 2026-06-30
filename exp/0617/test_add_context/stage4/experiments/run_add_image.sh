#!/usr/bin/env bash
# Probe: + same report page image (codex gpt-5.5 / qwen VLM read the page).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_variant add_image add-image.txt --no-add-context --add-image
