#!/usr/bin/env bash
# Ablation: attach the same report page image (VLM), no text context.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_exp add_image --no-add-context --add-image
