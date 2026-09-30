# Local validation receipt

- `validate.py` validation 01: PASS; original source preserved in results/source_01
- Bounded empty-domain refinement registered before validation 02
- `validate.py` validation 02: PASS on the same population
- `validate_crossing.py` crossing 01: PASS; original source retained
- Displayed three-level illustration corrected and asserted; crossing 02: PASS
- `python -m py_compile` on the four main Python files: PASS
- Markdown fenced-block parity, local Markdown link existence and trailing whitespace: PASS
- `sha256sum -c SHA256SUMS`: PASS when checked after sealing; regenerate the manifest
  whenever the integrating parent edits an artifact

All numerical checks use exact scalar references. No model run, native-library
replacement, kernel timing, GPU, HF state comparison, target VRAM, TTFT, 4B-Q4
baseline or GitHub Actions was executed in this package. The full native crossing
chart constructor is DERIVED only. No repository commit/push was performed by
this worker; the parent owns integration and durable handoff.
