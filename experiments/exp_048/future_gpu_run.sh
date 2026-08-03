#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
EXP-048 has not passed its small-checkpoint B3 early Gate and has no Phase-D backend.
Do not report GPU, CUDA, PCIe, SSD, 8 GiB, 70B, 405B, TTFT, or tokens/sec evidence.
Complete and freeze the current CPU small-checkpoint Gate first.
EOF
exit 2
