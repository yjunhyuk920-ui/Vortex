# EXP-096A latest result — Online Residual Direct-Margin Certificate Gate

## Identity

- source commit: `33f3118a9e61471d67115d78b44300550c6f0d2d`
- raw result: `results/exp_096a/33f3118a9e61471d67115d78b44300550c6f0d2d/result.json`
- public checkpoint: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- checkpoint tensor SHA-256: `ee567358e6d45900612a1a276fd068d1c5fd999a451cd67bae55cb2c01446308`
- deterministic core: `1180056056650640e458c0db08066ffa45de7794d54c09f868afcb82e8b0dfd1`
- mechanism fingerprint: `one-official-k128-fine-guessed-sweep/official-first-layer-q4-head/complete-online-float64-residual-bank/target-seeing-minimum-l1-direct-token-margin-lp/highs-ds-active-set/full-vocabulary-fp64-roundoff-certificate/build-and-untouched-holdout/cheapest-kill-first`

## Authoritative decision

`PROMOTE_RESIDUAL_MARGIN_CERTIFICATE_TO_CAUSAL_COEFFICIENT_GATE`

## Measured DEV-W Gate

- integrity controls: `True`
- build certificates: `384 / 384` (`100.000000000%`)
- untouched holdout certificates: `384 / 384` (`100.000000000%`)
- build decisive infeasible cases: `[]`
- holdout decisive infeasible cases: `[]`
- build coefficient L1 p50/p95: `0.9344711119756743 / 5.624883277711435`
- holdout coefficient L1 p50/p95: `1.0767554152067471 / 3.4613174575481795`
- holdout coefficient Linf p50/p95: `0.5049333191730976 / 0.8765775745756487`
- holdout coefficient support p50/p95: `5.0 / 13.0`
- minimum holdout certified FP64 margin: `9.534520602220081e-07`
- promotion Gate: `True`

Every accepted row was scanned against the complete vocabulary with the registered `gamma_(2K+4)` FP64 error enclosure. Every rejected row reports HiGHS status `2` on an explicit constraint subset or full matrix.

## Favorable target projection

- coefficient block: `131,072` bytes
- residual scoring and full margin scan work: `0.004085191%` of 405B parameter-equivalent scalar work/token
- projected total hot state: `8,111,839,744` bytes (`7.554739 GiB`)
- projected 8-GiB Gate: `True`
- perfect-block raw/compressed fine-sweep traffic/token: `0.781250000% / 0.619070788%`
- fine dense arithmetic/token: `100%`
- causal coefficient generator: `NOT CONSTRUCTED`

## Meaning

This is a target-seeing coefficient-capacity Gate. A pass means the online residual bank contains finite FP64 coefficient words that certify every target token. It does not make those coefficient words available before target generation.

## Claim boundary

A causal coefficient generator, target-independent certification, TARGET-W execution, physical 8-GiB allocation, fine arithmetic reduction, CUDA/SASS, and same-machine 4B-class p50/p95 remain `NOT TESTED`.
