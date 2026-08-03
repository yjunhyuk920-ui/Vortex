#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
EXP-047R has no real-operation replacement backend and no Phase-D runner.
Current valid work is the CPU small-checkpoint offline oracle audit only.
GPU, CUDA, PCIe, SSD, 8 GiB VRAM, 70B, 405B, TTFT, and tokens/sec remain NOT TESTED.
EOF
exit 2
