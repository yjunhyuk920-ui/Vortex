#!/usr/bin/env bash
set -euo pipefail
echo "EXP-077A authorizes only the pinned CPU favorable-oracle Gate." >&2
echo "No sparse GPU kernel, target-server mutation, or 35B/122B download is registered." >&2
exit 2
