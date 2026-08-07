#!/usr/bin/env bash
set -euo pipefail

echo "EXP-082A has not passed its structural lower-bound Gate; GPU execution is not authorized." >&2
exit 2
