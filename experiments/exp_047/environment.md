# EXP-047 Environment

## Current GitHub environment

Expected:

- Linux GitHub-hosted CPU runner;
- Python 3.10 and 3.12 CI matrix;
- no target 8 GiB GPU guarantee;
- no 405B checkpoint;
- no CUDA/PCIe/SSD target measurements.

Current run classification:

```text
Phase A/B
E1 ceiling
Phase D: NOT TESTED
```

## Runtime dependencies

The Phase-B core and runner use Python standard library plus pytest for tests. No model download is required.

## Determinism

- experiment seed: from `config.json`;
- Python random permutations are seeded per case;
- summary timing is environment-dependent and not a target projection;
- logical decisions and case metrics must reproduce under the same Python semantics and config.

## Future hardware variables

```text
VORTEX_MODEL_PATH
VORTEX_MODEL_REVISION
VORTEX_CHECKPOINT_MANIFEST
CUDA_VISIBLE_DEVICES
```

The future hardware script must capture driver, CUDA, GPU memory, model manifest, disk, RAM, and Git commit before any real-operation measurement.
