# EXP-093A latest result — One-Sweep True-Token Rank Gate

## Identity

- source commit: `fa35cf73dfcc544d07277cd716a467de44aca439`
- raw result: `results/exp_093a/fa35cf73dfcc544d07277cd716a467de44aca439/result.json`
- public checkpoint: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- checkpoint tensor SHA-256: `ee567358e6d45900612a1a276fd068d1c5fd999a451cd67bae55cb2c01446308`
- deterministic core: `d16b51ce653fd587508a99677ded722af9ebef74cbf227e4e11e88aa09e0fb5f`
- mechanism fingerprint: `one-official-guessed-context-sweep/frozen-prompt-suffix-cycle-16/complete-vocabulary-fp32-logits/stable-true-token-rank/static-top4-p95-top16-worst-case/target-after-sweep/build-and-untouched-holdout`

## Authoritative decision

`REJECT_ONE_SWEEP_STATIC_TOPK_BRANCH_SOURCE_AS_405B_CORE`

## Measured DEV-W Gate

- integrity controls: `True`
- frozen rank thresholds: p95 `<= 4`, maximum `<= 16`
- build rank p50/p95/max: `231.0 / 24139.799999999956 / 43083`
- build top-16 coverage: `23.697916667%`
- untouched holdout rank p50/p95/max: `1152.0 / 16703.899999999954 / 42864`
- untouched holdout top-1/top-4/top-16 coverage: `2.604166667% / 5.729166667% / 12.500000000%`
- holdout per-case maximum ranks: `[17115, 42864, 15604]`
- holdout per-case oracle static-path information bits: `[920.1296985438929, 1282.9276558845934, 1334.785921890988]`
- promotion Gate: `False`

## Favorable source accounting

- one-sweep raw logical checkpoint traffic/token: `0.781250000%`
- one-sweep favorable-compressed logical traffic/token: `0.619070788%`
- static top-16 token-ID bytes/block: `4,096`
- one-sweep dense arithmetic/token: `100%`
- branch successor construction and verification: `NOT MEASURED`

The target continuation is generated only after the guessed-context sweep completes. It is used to score the already-fixed logits and never to generate the source.

## Meaning

A rank above 16 proves that the exact token is absent from the frozen static top-16 source at that position. No tree layout, verifier, or kernel can recover that missing token without a new information source. A pass would still be only a necessary condition because alternate branch tokens change later hidden states and logits.

## Claim boundary

This Gate does not construct branch-dependent successor states, reduce dense arithmetic, execute TARGET-W, measure 8-GiB residency, or establish 4B-class latency. Those remain `NOT TESTED`.
