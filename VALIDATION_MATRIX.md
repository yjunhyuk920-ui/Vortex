# Validation matrix — 2026-09-07

[Current report](experiments/residual_absorption_20260907/REPORT.md).
[Replay instructions](experiments/residual_absorption_20260907/README.md).
[Prior matrix unchanged](docs/research/history/pre_residual_absorption_20260907/VALIDATION_MATRIX.md).

| Item | Evidence and boundary |
|---|---|
| Constructor |Exact dyadic L1 scan; scalar metadata; original hashes preserved; native endpoint tests|
| Conditional source |Whole branch erased before matrix read, not approximate then replay|
| Causal state |K/V writers preserved; global erasure permits token KV/logit table; induction context<=32|
| Numerical trust |Fraction/C independent endpoint/projection checks; BF16 SiLU range exhaustive; stable exp an explicit ABI assumption|
| Replay |8 synthetic models,256 guarded steps/8192 logits,128 table steps/4096 logits; raw KV and own RNG equal|
| Coverage |4 high-scale favorable models qualify,4 do not; ordinary exact identities0; midscale true identities but proof misses|
| Query |Favorable source read7.001%/MAC6.383%; table192B vs96896B source, not measured latency|
| Full costs |Startup/precompute/KV actual FP32 storage separated;32-query bytes >=15.9% of selected source budget; target tables96.20GiB|
| Source restoration |7 exact code/test/replay files;19 tests;455 run files+2 derivations repeat original manifest|
| Universal theory |O1-O6 OPEN; CORE_ADMISSION=false; no3 qualifying universal principles|
| Public model/full HF/RoPE/CUDA/405B/8GiB/4BQ4/TTFT |NOT TESTED / not constructed|
| Actions/full repository suite |NOT RUN|

Remote sources/reports/manifest persist; scientific raw files regenerate to original
hashes. Complete raw binaries and development logs are in the user ZIP, not all
embedded in Git. Source archival compression is not inference compression. Commit,
scientific theorem and target hardware success are separate statuses.
