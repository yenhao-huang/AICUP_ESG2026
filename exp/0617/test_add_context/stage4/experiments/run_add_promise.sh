#!/usr/bin/env bash
# Probe: + promise_string (annotation field; data-use exceeds data-only).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_variant add_promise add-promise.txt --no-add-context --add-promise-string
