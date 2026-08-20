# EXP-090A latest result — Causal Suffix-Action Cache Gate

## Identity

- source commit: `5dcd865cf4a0cc210234d23aa4225a33c8cb78ff`
- raw result: `results/exp_090a/5dcd865cf4a0cc210234d23aa4225a33c8cb78ff/result.json`
- public checkpoint: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- checkpoint tensor SHA-256: `ee567358e6d45900612a1a276fd068d1c5fd999a451cd67bae55cb2c01446308`
- deterministic core: `720d15f94111bc6b76864b80f9177549635419df4064395fc4c3a5fdb94ba412`
- mechanism fingerprint: `one-official-bf16-layer/rowwise-q4-logit-lens/frozen-verified-bf16-suffix-action-history/raw-carry-secant-nearest/128-token-chain/build-selection/untouched-holdout`

## Authoritative decision

`REJECT_CAUSAL_SUFFIX_ACTION_CACHE_AS_128_TOKEN_DRAFT_CORE`

## Build selection

- selected mode: `raw_q4`
- build minimum accepted length: `0 / 128`
- build total accepted length: `0`
- build true-path rank p95/max: `28963.349999999973 / 45133`

## Untouched holdout

- accepted lengths: `[0, 0, 0]`
- minimum/p50/maximum: `0 / 0.0 / 0`
- target-token rank p50/p95/max: `1227.5 / 15186.249999999996 / 41239`
- top-1/top-4/top-16 fractions: `6.770833333% / 10.937500000% / 16.666666667%`
- candidate-before-target control: `True`
- manual first-layer replay equality: `True`

## Resource screen

- projected hot state: `7,981,689,344` bytes / `7.433527470` GiB
- projected 8-GiB margin: `608,245,248` bytes
- hot-state screen: `True`
- actual-min-acceptance verification fraction, no compression: `inf%`
- favorable compressed fraction: `inf%`
- registered p50 fraction limit: `1.185185185%`
- verification arithmetic fraction: `100.000000000%`

## Meaning

The draft is causal and training-free: one official BF16 layer, a Q4 logit lens, and suffix actions from already verified prompt positions. Candidate chains were completed and hashed before the corresponding target continuation. Wrong candidates are not committed; a future exact target block verifier would retain the official successor state.

This Gate measures only whether the candidate source is strong enough to justify that verifier or a bounded candidate tree. It does not reduce the exact target arithmetic and does not establish target latency.

## Claim boundary

TARGET-W acceptance, packed Q4 execution, exact physical block verification, actual 8-GiB residency, SSD/PCIe/HBM traffic, target arithmetic throughput, and same-machine 4B-class p50/p95 remain `NOT TESTED`.
