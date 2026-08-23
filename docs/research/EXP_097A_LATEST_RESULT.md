# EXP-097A latest result — Residual Bank Specificity Gate

- source commit: `72ca77abf81397fb69b5e3d402ff034f175fb3a3`
- decision: `REJECT_UNBOUNDED_RESIDUAL_MARGIN_CAPACITY_AS_MODEL_SPECIFIC_SOURCE`
- all banks frozen before target: `True`
- matched controls: `['cross_prompt', 'vocab_permuted']`

## Untouched holdout

### same
- certificates: `24/24` (`100.000000000%`)
- support p50/p95: `4.5 / 13.849999999999998`
- L1 p50/p95: `0.9076383996996851 / 3.831871688116445`

### cross_prompt
- certificates: `24/24` (`100.000000000%`)
- support p50/p95: `5.5 / 14.849999999999998`
- L1 p50/p95: `1.0399409455441027 / 4.744448973030837`

### vocab_permuted
- certificates: `24/24` (`100.000000000%`)
- support p50/p95: `9.5 / 17.0`
- L1 p50/p95: `2.8658957284330246 / 7.20327861954318`

## Claim boundary

This is a DEV-W specificity falsification Gate. A causal coefficient generator,
fine dense arithmetic reduction, TARGET-W, physical 8-GiB execution, CUDA/SASS,
and same-machine 4B-class latency remain `NOT TESTED`.
