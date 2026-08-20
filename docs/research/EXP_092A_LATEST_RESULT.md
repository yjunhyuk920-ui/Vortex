# EXP-092A latest result — One-Sweep Block-Span Correction Closure Gate

## Identity

- source commit: `01abfe5e30f1e9063b62b4c6085844bd58cfddda`
- raw result: `results/exp_092a/01abfe5e30f1e9063b62b4c6085844bd58cfddda/result.json`
- public checkpoint: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- checkpoint tensor SHA-256: `ee567358e6d45900612a1a276fd068d1c5fd999a451cd67bae55cb2c01446308`
- deterministic core: `bc466a6c154135d8b7a8bc239cc3e4c7c56a38d8492129bf212aaa4ff89300c8`
- mechanism fingerprint: `one-guessed-target-block/complete-linear-operator-input-cache/exact-dyadic-row-span/192-coordinate-restriction/three-prime-rank-lower-bound/true-teacher-forced-oracle/first-middle-last-complete-layers`

## Authoritative decision

`INVALID_BLOCK_SPAN_CORRECTION_CONTROL_FAILURE`

## Measured DEV-W Gate

- integrity controls: `False`
- frozen extra-direction budget per operator: `8`
- untouched holdout reports: `36`
- holdout certified extra-rank lower bound p50/p95/max: `64.0 / 64.0 / 64`
- holdout certified extra-fraction lower bound p50/p95: `50.000000000% / 50.000000000%`
- holdout reports already over the frozen budget: `33 / 36`
- optimistic correction-operation multiplier lower bound: `1.500000000x`

## Favorable target projection

- sidecar at frozen eight-direction budget: `206,438,400` bytes
- sidecar lower bound at the observed maximum certified requirement: `1,651,507,200` bytes

The projection grants true target blocks, exact coefficients, basis discovery, nonlinear recomputation, routing, coefficient work, and native-order repair for free. It is `DERIVED`, not a TARGET-W measurement.

## Meaning

The first guessed block supplies 128 arbitrary projection-input basis rows. The Gate asks whether the exact teacher-forced block remains inside that span plus only eight static directions. A modular rank increase on a frozen 192-coordinate restriction is a certified lower bound on the full rational rank increase. Therefore a lower bound above eight cannot be repaired by a full-rank solver or kernel implementation.

## Claim boundary

This result covers first/middle/last complete DEV-W layers and four dominant linear-operator input roles. Full-coordinate ranks when the lower-bound Gate survives, a causal coefficient constructor, all TARGET-W layers, 8-GiB residency, physical traffic, block arithmetic, and same-machine 4B-class latency remain `NOT TESTED`.
