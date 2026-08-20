# EXP-095A latest result — Online Deep-Residual Affine-Hull Gate

## Identity

- source commit: `e284b482e7e64d6db42f3cae4037f1c6b61b87bf`
- raw result: `results/exp_095a/e284b482e7e64d6db42f3cae4037f1c6b61b87bf/result.json`
- public checkpoint: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- checkpoint tensor SHA-256: `ee567358e6d45900612a1a276fd068d1c5fd999a451cd67bae55cb2c01446308`
- deterministic core: `0575e9091485275f453019215ff96d3d0cf7188e92ec4c42c7f11cc1b800070c`
- mechanism fingerprint: `one-official-k128-fine-guessed-sweep/one-official-first-layer-q4-head/complete-online-float64-deep-residual-bank/eight-neighbor-affine-ridge-causal-transport/best-single-residual-oracle/full-residual-row-span-l2-oracle/delayed-incremental-target/build-and-untouched-holdout`

## Authoritative decision

`REJECT_ONLINE_DEEP_RESIDUAL_AFFINE_HULL_AS_405B_SOURCE`

## Measured DEV-W Gate

- integrity controls: `True`
- build accepted prefixes: `[1, 3, 1]`
- untouched holdout accepted prefixes: `[1, 1, 1]`
- untouched holdout exact positions: `[1, 1, 1]`
- causal affine holdout rank p50/p95/max: `2410.0 / 35475.2 / 47647`
- causal affine top-1/top-4/top-16: `1.822916667% / 2.604166667% / 8.854166667%`
- position-aligned EXP-094A-equivalent p95/max: `36921.549999999945 / 49111`
- best-single-residual oracle p95/max/top-16: `1475.9499999999991 / 23031 / 26.562500000%`
- full residual-span L2 oracle p50/p95/max: `6.0 / 1083.4499999999973 / 20337`
- full residual-span oracle top-1/top-4/top-16: `22.656250000% / 42.447916667% / 64.322916667%`
- holdout residual-bank effective ranks: `[128, 128, 128]`
- holdout projection relative-L2 values: `[0.17584267273363213, 0.1540593130914987, 0.267314375131764]`
- exact-chain / causal-branch / oracle-span Gates: `False / False / False`

## Favorable target projection

- full-block raw/compressed fine-sweep traffic/token: `0.781250000% / 0.619070788%`
- observed-min compressed source traffic/token: `79.241060816%`
- causal affine scalar-work fraction of 405B: `0.000285835%`
- added float64 hidden bank: `16,777,216` bytes
- added float64 Gram matrix: `131,072` bytes
- projected total hot state: `8,111,707,136` bytes (`7.554616 GiB`)
- projected 8-GiB Gate: `True`
- fine dense arithmetic/token: `100%`
- sound token certificate: `NOT CONSTRUCTED`

## Meaning

The causal chain uses only one guessed-context target sweep, the official first-layer/Q4 coarse state, and a branch-conditioned affine combination of the online residual bank. The single-row and complete-span oracles are computed only after the delayed target and are favorable capacity diagnostics, not deployable candidates.

## Claim boundary

TARGET-W acceptance, a sound coefficient/token certificate, exact successor commitment, physical VRAM, fine arithmetic reduction, and same-machine 4B-class p50/p95 remain `NOT TESTED`.
