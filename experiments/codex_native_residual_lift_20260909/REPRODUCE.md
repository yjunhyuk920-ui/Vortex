# Reproduction

Saved sources and hashes are authoritative. Run in a fresh copy/output location;
lift.py, integer_audit.py and torch_cpu_audit.py refuse existing result files.
Do not overwrite the preserved result package.

    python lift.py
    python integer_audit.py
    python check_admission.py

The first two use Python3.12.14/NumPy2.3.5 in the bundled runtime. Set
PYTHONUTF8=1 and PYTHONDONTWRITEBYTECODE=1. integer_audit.py reads the already
verified registered-shape inventory from the prior hardened FMM result.

    python torch_cpu_audit.py

Use the existing torch2.8.0+cpu environment for this separate small operator
audit. It explicitly uses CPU tensors. No GPU, full HF or Actions is required.

partial_source_v1.py is intentionally defective incomplete work; ACTUAL_RED.json
records the actual primary reproduction. It is not a passing implementation.
checksums.json covers all package files except itself.
# Stored evidence check

Run `python verify_artifacts.py` from this directory to compare the manifest,
executed source bindings, stored ordered cases and both native output WORD
arrays. This rereads evidence; it does not rerun the 24 native CPU calls.
