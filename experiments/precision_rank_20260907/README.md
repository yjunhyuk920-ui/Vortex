# Reproduce the bounded precision-source experiment

Read [REPORT.md](REPORT.md) and [PREREGISTRATION.md](PREREGISTRATION.md).

With Python3 and NumPy2.3.5 installed, plus a C11 compiler:

```sh
python restore.py /tmp/vortex-precision --run
```

This verifies XZ and source-JSON SHA256, restores7 source/test/expected-result files, recompiles C, runs18tests, and regenerates603 scientific files to the frozen manifest. It refuses overwriting a different existing restored source. The capsule is archival compression only, not a model-compression result. Raw binary inputs/programs/output bits regenerate; they and detailed Korean report/development logs are in the downloadable ZIP, not all embedded here.

Global THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED; CORE_ADMISSION=false. Accuracy on the restricted integer primitive is not native HF/KV/RNG correctness or speed. No Actions/full-repository suite/target hardware run.
