# Native rounding-aware decision geometry — bounded auxiliary

Read [report](REPORT.md) and [scoped obligations](LEDGER.md). Fixed mission unchanged; CORE_ADMISSION=false.

From a checked-out research branch, restore into a NEW EMPTY directory:

```sh
python experiments/native_decision_geometry_20260907/restore.py /tmp/vortex-decision --run
```

Requires Python, NumPy and a C compiler. Validated with Python3.13.5, NumPy2.3.5, GCC14.2.0. This is local replay, not Actions. Restores22 text files including full Korean proofs, preregistration, source, C reference, tests, exact original results/manifest and retained development versions/logs. Rebuilds native reference and regenerates51 scientific files. Checksums are verified before restore and after replay. SOURCE_*.b64 are four pieces of one archival XZ stream, NOT model compression or an inference data structure.

The full original scientific binaries (weights, inputs, generated indices, traces) are in the user ZIP and regenerate from the preserved sources to the original manifest. They are not all embedded in Git. Later split-wrapper restoration also passed; no science-file changes. Full HF/KV/softmax sampler/GPU/405B/8GiB/4BQ4/TTFT not tested.
