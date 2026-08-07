#!/usr/bin/env bash
set -euo pipefail
echo "EXP-076 authorizes only the pinned CPU causal reference Gate." >&2
echo "No vLLM/SGLang GPU backend, target-server mutation, 35B/122B download, or page scheduler is registered." >&2
exit 2
