# EXP-094A latest result — Parareal Deep-Residual Transport Gate

## Identity

- source commit: `7bf744f73b725cee073ad68a8581be34d8243327`
- raw result: `results/exp_094a/7bf744f73b725cee073ad68a8581be34d8243327/result.json`
- public checkpoint: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- checkpoint tensor SHA-256: `ee567358e6d45900612a1a276fd068d1c5fd999a451cd67bae55cb2c01446308`
- deterministic core: `d5c96d01d57aaa6f077933a42eab83e9dc8ba03d0a6305594809e687018caad6`
- mechanism fingerprint: `one-official-k128-fine-guessed-sweep/one-official-first-layer-q4-head-coarse/position-aligned-float64-fine-minus-coarse-residual/causal-corrected-prefix-coarse-plus-frozen-residual/delayed-incremental-target/build-and-untouched-holdout`

## Authoritative decision

`REJECT_PARAREAL_RESIDUAL_TRANSPORT_AS_405B_BLOCK_SOURCE`

## Measured DEV-W Gate

- integrity controls: `True`
- build accepted prefixes: `[1, 2, 1]`
- untouched holdout accepted prefixes: `[1, 1, 1]`
- holdout exact positions: `[4, 3, 1]`
- transported holdout rank p50/p95/max: `4787.0 / 39824.74999999999 / 49105`
- transported holdout top-1/top-4/top-16: `3.385416667% / 5.208333333% / 8.854166667%`
- static fine-sweep holdout rank p95/max: `17096.299999999992 / 42541`
- Parareal p95 rank improvement: `-22728.45`
- Parareal top-16 coverage improvement: `-1.302083333%`
- exact-chain Gate: `False`
- branch-source Gate: `False`

## Favorable target projection

- full-block raw/compressed fine-sweep traffic/token: `0.781250000% / 0.619070788%`
- observed-min compressed source traffic/token: `79.241060816%`
- coarse parameter-equivalent arithmetic fraction: `1.305930208%`
- float64 residual buffer: `131,334,144` bytes
- projected total hot state: `8,094,798,848` bytes (`7.538869 GiB`)
- projected 8-GiB Gate: `True`
- fine dense arithmetic/token: `100%`
- sound token certificate: `NOT CONSTRUCTED`

## Meaning

The candidate is generated before target continuation from the fixed equation `F(g)+G(c)-G(g)`. A rank or acceptance gain therefore measures a real nonlinear causal transport effect, not a post-target fit. A pass still requires a sound certificate and physical executor.

## Claim boundary

TARGET-W acceptance, packed kernels, exact token commitment, physical VRAM, and same-machine 4B-class p50/p95 remain `NOT TESTED`.
