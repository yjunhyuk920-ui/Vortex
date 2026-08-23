# Next Experiment — EXP-106A

## Depth-Complete Width-Thin Checkpoint Surrogate Reality Gate

### Why this is next

EXP-104A showed that a full-vocabulary head plus one complete target-width layer is already above the p50 scan budget. EXP-105A showed that an exact activation-ordered residual-bound head index cannot rescue that complete-layer path: on a legal late-decision head the combined charged traffic is `2,815,515,648` bytes/token versus a `2,400,000,000` limit. A head without a state source is incomplete.

The next Gate must therefore replace **one full-width layer per draft token** with a causal checkpoint-derived state source that is materially cheaper while still observing the target's full depth.

### Three materially different principles

1. **Depth-complete width-thin checkpoint surrogate — selected.** For every target layer, derive a narrow draft-only state and narrow checkpoint-static operators without training. The unchanged target remains authoritative. The draft must include explicit input/output bridges and a charged sub-full-head decision path.
2. **Bit-sliced all-layer state transducer.** Propagate only registered finite-word state bits through every layer. Reject before implementation if total bitplanes, nonlinear metadata, and head state cannot fit the 8-GiB and p50 ledgers.
3. **Cold-probe exact vocabulary decision index.** Keep bounded resident metadata and adaptively probe exact cold head rows/bitplanes. It is eligible only with a concrete causal state source and a measured exact fallback rate; a stronger index, not another Cauchy/block-size variant, is required.

### Selected representation must be concrete

Freeze before any checkpoint experiment:

```text
narrow state width m
all layer/operator shapes
checkpoint-derived compilation algorithm
input embedding bridge bytes
per-layer narrow weight bytes
nonlinear/normalization implementation
KV/state bytes
output-head or exact index bytes
workspace/fragmentation
per-token arithmetic and traffic
fallback and target verification path
```

No training, future target state, target-seeing selector, free bridge, free head, free HBM scan, unmeasured compression, or projected acceptance.

### First local Gate

The complete draft path must satisfy all of:

```text
resident bytes <= 8 GiB
per-token draft traffic/operation route <= 2.4 GB-equivalent p50 budget
no complete target-width layer scan per proposed token
causal finite-word execution exists
exact target verifier and fail-closed fallback exist
```

If this static Gate passes, run an unchanged public checkpoint locally. Measure actual accepted prefix `A`, `N/A`, draft/verify/rebuild/repair time, and exact token plus terminal target state. With raw BF16 target sweeps, promotion still requires actual `A>=339` unless exact sweep bytes are measurably reduced.

### Stop rule

Reject immediately when:

- the narrow representation is merely a trained/distilled model;
- bridge/head bytes restore a full-width scan;
- a layer uses unavailable future hidden state;
- width is selected after holdout results;
- target-scale cost exceeds the p50 budget;
- real accepted prefix does not approach the registered floor.

Do not run GitHub Actions after local validation. Commit source, config, raw result, logs, checksums, ledgers, README, and handoff, then verify the remote SHA.
