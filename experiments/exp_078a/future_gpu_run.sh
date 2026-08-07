#!/usr/bin/env bash
set -euo pipefail
echo "EXP-078A rejected frozen tangent reuse before any physical macro or GPU Gate." >&2
echo "Do not run a GPU/kernel continuation for this experiment." >&2
exit 2
