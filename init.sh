#!/usr/bin/env bash
set -euo pipefail

case "${1:-start}" in
  start)
    echo "Agent start checklist"
    echo "1. Read Task.md"
    echo "2. Read Progress.md"
    echo "3. Read Decisions.md"
    echo "4. Inspect git status"
    ;;
  finish)
    echo "Agent finish checklist"
    echo "1. Update Progress.md"
    echo "2. Add major decisions to Decisions.md"
    echo "3. Run relevant validation"
    echo "4. Inspect git status"
    ;;
  *)
    echo "Usage: ./init.sh [start|finish]" >&2
    exit 2
    ;;
esac
