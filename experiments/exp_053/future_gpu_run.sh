#!/usr/bin/env bash
set -euo pipefail

echo "EXP-053 has no Phase-D or real-Transformer circuit backend." >&2
echo "405B, <=8 GiB VRAM, CUDA, PCIe, SSD, TTFT, tokens/second, and physical circuit traffic remain NOT TESTED." >&2
exit 2
