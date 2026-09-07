# Validation matrix — 2026-09-07

[Current report](experiments/symbolic_source_20260907/REPORT.md).
[Previous matrix unchanged](docs/research/history/pre_symbolic_source_20260907/VALIDATION_MATRIX.md).

| Item | Evidence and boundary |
|---|---|
| Constructor |Finite source-derived candidates, serialized AST, loaded C interpreter; no response truth table|
| Symbolic |7 scenarios;2 UNSAT rewrites;2 UNKNOWN retained; all finite BF16 words including signed0/subnormals|
| State |Toy successor roots included in miter; erasure counterexample; not full KV/RNG|
| Replay |28678 original/selected C checks,0 differences; unchanged baseline cases included|
| Follow-on |Exact-product guard proof, no BF16-store crossing; input-leaf type requirement|
| Matrix checks |4 matrices,448 queries,1664 FP32 and1664 BF16 coordinates; C and independent Fraction agree|
| Costs |4x32 expression ops252->128, but same128mult+124add; literal bytes492->492; code6512->4528B vs256B source weights|
| Numerical witness |x=3*2^-133, separate BF16 product stores give0x0001, merged coefficient gives0x0002|
| Reproducibility |22 tests,56 scientific files; original manifest SHA anchored in restore; fresh11-file source restore passes|
| Proof trust |Z3 UNSAT trusted, no independent proof-certificate checker; mathematical guard proof plus finite independent tests|
| Full theory |O1-O6 OPEN; no3 qualifying principles or core admission|
| HF/full Transformer/CUDA/GPU/405B/8GiB/4BQ4/TTFT |NOT TESTED|
| Actions/full repository suite |NOT RUN|

All source/test files are remotely archived and independently hash-read back with
the report. Scientific traces regenerate to original hashes and are all in the user
ZIP; no claim all binary/raw traces are embedded in Git. Archive compression is not
an inference result. Final branch/PR receipt is separate.
