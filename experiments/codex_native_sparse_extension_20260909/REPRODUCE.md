# Reproduce the focused evidence

Use Python 3.12 for the stdlib scalar/oracle files. For HF, use the exact installed
CPU environment torch 2.8.0+cpu and transformers 4.55.4 from the manifest.
The current verified interpreter was:
`C:\ChatOnSteroids\Vortex-output-envelope\experiments\native_global_transition_20260908\.venv\Scripts\python.exe`.
No package install or environment change is needed in this workspace.

Run from this package, with PYTHONDONTWRITEBYTECODE=1 and new output directories:

```powershell
python -c "from pathlib import Path; import run_scalar; run_scalar.OUT=Path('scalar_replay'); run_scalar.main()"
python integer_oracle_check.py --output-dir integer_replay
python hf_bridge.py --output-dir hf_replay
```

The recorded canonical directories already exist and must not be overwritten.
The first command reruns the scalar sweep; the second uses independent exact
integer arithmetic; the third actually runs native CPU HF and the compiled
replacement with dense/model-forward prohibition. No command runs CUDA or a
405B checkpoint. No timing/peak-memory acceptance is attached to these commands.

Compare scalar report bytes to scalar_results_v2/report.json and oracle summary
bytes to integer_results_v1/summary.json. Compare HF summary/manual/sample records
to hf_results_v5. The HF manifest includes environment details and may change if
the interpreter/build differs; do not suppress those differences. Hash the
results and check the checkpoint unchanged flag. Historical HF v2 is an expected
failure; scalar_v1_sources contains identities, not a runnable old implementation.
