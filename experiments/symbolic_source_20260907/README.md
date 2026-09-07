# Bounded symbolic source construction

[Full Korean proof, results, costs and O1-O6 ledger](REPORT.md).
Base: PR137 de11732ec478773164aeb2cf7de02efd0483c9e9. No core admission.

All 11 source/test/replay files are in CODE_CAPSULE.txt, a hash-checked XZ/Base64
archive of a UTF-8 JSON path-to-text map. This is research archival compression,
not an inference gain. proof_rules.py is also directly visible for inspection.
All original traces, historical versions and logs are in the user ZIP; the complete
56-file scientific result set is regenerated and compared to its frozen original
manifest hash. Binary traces are not all embedded in this Git commit.

Requirements: Linux Python3, NumPy, gcc, installed libz3.so.4 (recorded4.13.3.0).
No Python z3 package or network is used. Source extraction does not execute code:

```sh
python experiments/symbolic_source_20260907/restore.py /tmp/vortex-symbolic
```

For a fresh complete local build/test/regeneration with original hash verification:

```sh
python experiments/symbolic_source_20260907/restore.py /tmp/vortex-symbolic-replay --run
```

Destination must be empty. The wrapper creates results/ before invoking the original
runner; the first packaging replay failed on that missing directory and is preserved
in the user archive. Numeric source and original result hashes were unchanged.
A different solver/build can alter witnesses or UNKNOWN results; hash differences
are failures to reproduce, not silently accepted passes. The original tests include
UNKNOWN retention. No independent machine-checked UNSAT proof log is claimed.

See PREREGISTRATION.md and FOLLOWON.md for fixed inputs, candidates and limits.
Full Transformer/RNG/HF/CUDA/405B/8GiB/4BQ4/TTFT/Actions/full-suite: NOT TESTED.
