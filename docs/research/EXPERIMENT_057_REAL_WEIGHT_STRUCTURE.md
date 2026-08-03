# EXP-057 — Pinned Real-Checkpoint Weight-Structure Extraction Gate

## Authority

- workflow `30824957941`
- source head `cf9d7099dc11b22ce24ba6e096712d5da1bc3729`
- workflow merge `0c70c5547a68ce3db4a584ac32fb0cbf9873d861`
- artifact `8860450501` (197667 bytes)
- artifact ZIP SHA-256 `7e2d91fb1af2d77c7cb87732557e8c42c22e23771264cfb000d29536d76172f0`
- config SHA-256 `e99e13b3c912f1567d010c1c60fa0c8ade0b2350bd8ce6cacc49e244c4df334e`

## MEASURED

- 3 unchanged revision-pinned TinyStories checkpoints;
- 327 learned tensors, 153 analyzed 2-D tensors, 54,205,312 named 2-D scalars;
- 144 dense-projection matrices in the primary Q4 Gate;
- zero unregistered 2-D tensors and zero reconstruction/control failures;
- exact repeated/sign-related dense matrices: 0 in FP32, Q8, and Q4;
- Q4 p50/p90 operations: 82.8918%/85.8398%;
- Q4 p50/p90 query bytes: 329.0244%/490.6845%;
- Q4 median/p90 exact residual density: 81.4087%/84.2834%;
- best real dense matrix: 70.2866% operations;
- maximum projected logical 405B-Q4 storage: 0.9300 TiB;
- maximum compile amortization: 377 queries.

## Decision

```text
REJECT_REAL_WEIGHT_EXACT_GROUPING_DICTIONARY_AS_CORE_RETAIN_MEASURED_AUXILIARY_ONLY
```

The measured models do not contain the exact repeated or sparse-residual column structure required by EXP-055/056. Q8/Q4 model-output preservation and operation replacement were not tested.
