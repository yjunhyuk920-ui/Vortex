# Selector-adjoint source gate (auxiliary, NOT a 405B engine)

Read [REPORT.md](REPORT.md); the user ZIP also includes the detailed Korean report in `docs/REPORT_KO.md`. The mission remains unachieved.

Requirements: Python 3, NumPy, C11 compiler `cc`, little-endian host, standard math library. No model downloads or GPU required.

```sh
python verify.py
```

This compiles the independent C rounding reference, runs 16 unit checks, regenerates 84 scientific files, and compares against the frozen expected manifest. Candidate evaluation is Python Boolean-circuit interpretation; C is the reference, not a GPU/fast candidate. Lanes are packed only to make verification efficient; this is not a batch-one inference speed result.

Source: `src/adjoint.py`, `src/reference.c`, `run.py`, `tests/test_adjoint.py`.
Evidence: `results/run/summary.json`, `scope.json`, frozen manifest and per-case data.
The source reconstructs its own input/output files deterministically. No trace of an actual pretrained model, CUDA, full KV/RNG, VRAM or 4B latency is included or claimed.

Committed source recreates all 84 scientific files. Raw binaries are in the user ZIP and regenerated locally, not all embedded in Git.
