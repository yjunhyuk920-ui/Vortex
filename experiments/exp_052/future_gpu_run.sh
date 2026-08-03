#!/usr/bin/env bash
set -euo pipefail

echo "EXP-052 has no Phase-D backend." >&2
echo "405B, <=8 GiB VRAM, CUDA, PCIe, SSD, TTFT, tokens/second, and physical advice lookup remain NOT TESTED." >&2
exit 2
