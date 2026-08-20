# EXP-091A latest result — Exact Jacobi Fixed-Point Gate

## Identity

- source commit: `aa324f2b99d53207a909e70049dd025b139b42b8`
- raw result: `results/exp_091a/aa324f2b99d53207a909e70049dd025b139b42b8/result.json`
- public checkpoint: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- checkpoint tensor SHA-256: `ee567358e6d45900612a1a276fd068d1c5fd999a451cd67bae55cb2c01446308`
- deterministic core: `0a5c0dc3822effee32a97ff9d569467d2f81a9915aa786f0a58035937f9bc14c`
- mechanism fingerprint: `official-target-jacobi/128-token-causal-block/three-frozen-causal-seeds/eight-sweeps/self-consistent-exact-prefix/build-selection/untouched-holdout`

## Authoritative decision

`REJECT_NATIVE_JACOBI_FIXED_POINT_AS_405B_CORE`

## Build selection

- selected seed: `prompt_suffix_cycle_16`
- minimum/total first-sweep exact release: `0 / 1`
- minimum/total best release over eight sweeps: `7 / 24`
- worst fixed-point sweep: `None`

## Untouched holdout

- first-sweep exact releases: `[0, 0, 0]`
- minimum/p50/maximum: `0 / 0.0 / 0`
- best releases over measured sweeps: `[7, 7, 9]`
- fixed-point sweeps: `[None, None, None]`
- trajectory-before-target control: `True`
- exact-prefix theorem controls: `True`
- fixed-point/AR controls: `True`

## Target equation

- required tokens per sweep, no compression: `85`
- required tokens per sweep, favorable lossless compression: `67`
- observed holdout minimum first-sweep release: `0`
- observed logical fraction, no compression: `inf`
- observed logical fraction, favorable compression: `inf`
- two-sweep full-block fraction, no compression: `1.562500000%`
- target arithmetic per block sweep: `100.000000000%`

## Meaning

A self-consistent prefix is exact by causality: the guessed token and unchanged target proposal agree at every released position, and every preceding released token is already exact. The ordinary AR continuation was delayed until all trajectories were complete and served only as an independent integrity control.

A rejection means the unchanged checkpoint does not release enough exact tokens per target sweep from any frozen causal seed to approach the 405B traffic target. It does not reject trained consistency models or arbitrary new exact token certificates; retraining remains outside the fixed mission.

## Claim boundary

TARGET-W acceptance, target KV/storage behavior, 8-GiB residency, physical streamed checkpoint execution, target arithmetic throughput, and same-machine 4B-class p50/p95 remain `NOT TESTED`.
