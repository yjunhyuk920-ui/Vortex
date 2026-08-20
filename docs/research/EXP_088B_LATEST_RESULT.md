# EXP-088B latest result — oracle cross-layer program-sharing Gate

## Identity

- source commit: `991bae0f0bfe42e9f4b918499b1e055b293c2c02`
- raw result: `results/exp_088b/991bae0f0bfe42e9f4b918499b1e055b293c2c02/result.json`
- public checkpoint: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- selected complete layer window: `[14, 15]`
- mechanism fingerprint: `checkpoint-static/two-adjacent-complete-layers/linear-64x64-tiles/eight-cross-layer-page-families/exhaustive-bitmask/final-hidden-plus-local-KV-bit-exact/build-intersection/untouched-holdout`

## Authoritative decision

`REJECT_CROSS_LAYER_PAGE_MASK_PROGRAM_SHARING_AS_405B_CORE`

## Measured DEV-W result

- integrity controls: `True`
- build states: `3` distinct states
- untouched holdout states: `3 / 3` exact under the selected program
- exhaustive program language: `256` masks
- exact programs shared by all build states: `1`
- selected mask: `0xff`
- selected program SHA-256: `902ba9839b731fa92a30c08ebf0e15463aaa020fc20aec4e0d92b7b79aca6f44`
- selected retained linear bytes: `14,155,776 / 14,155,776`
- retained fraction: `100.000000000%`
- empty-program exact holdout states: `0`
- oracle program-sharing Gate: `False`

## Optimistic target projection

The projection grants that every 405B checkpoint linear page obtains the same retained fraction, while program metadata, non-linear state traffic, addressing, materialization, and sparse-kernel overhead are free.

- projected hot bytes/token: `810,000,000,000`
- registered hot-byte limit: `100,000,000`
- projected hot-byte Gate: `False`

This is `PROJECTED`, not a 405B or target-GPU measurement.

## Claim boundary

This experiment is a non-deployable oracle upper-bound over one frozen program language. The official DEV-W graph executed two adjacent complete Transformer layers, and exactness covered the pair output hidden state plus both layers' complete post-step K/V caches. The oracle evaluator intentionally used dense official operations to decide whether a sparse static program is worth implementing; its wall time is not an execution speed claim. TARGET-W, an actual sparse kernel, 128-step continuation, 8-GiB VRAM, and 4B-class p50/p95 remain `NOT TESTED`.
