# E0 latest result — Speculative-Offload Dual-Roofline Audit

## Authoritative decision

`RETAIN_LOSSLESS_OFFLOAD_AND_OVERLAP_AS_AUXILIARY_REQUIRE_FINE_DENSE_ARITHMETIC_REDUCTION_FOR_CORE`

## Frozen target inputs

- exact target parameters: `405,849,243,648`
- favorable BF16 lossless ratio: `1.5x`
- registered PCIe ceiling: `7.4506 GiB/s`
- favorable device peak: `4.3e12 operations/s`
- GPU hot grant: `8 GiB`
- registered root free space: `97.6183 GiB`

## Derived lower bounds

- raw exact BF16 checkpoint: `755.953125 GiB`
- favorable compressed checkpoint: `503.968750 GiB`
- registered storage deficit: `406.350450 GiB`
- compressed target-sweep I/O floor: `67.641364454 s`
- dense candidate arithmetic floor: `188.767090069 ms`
- I/O/compute crossover: `358.332400151` candidate nodes
- first integer perfect-chain crossover: `359` tokens
- minimum native 4B p50 for any unreduced dense exact chain: `157.305908391 ms`
- full-model 1-bit substitute: `47.247070313 GiB`
- maximum whole-model substitute rate inside the entire 8-GiB grant: `0.169322668 bits/parameter`

## Meaning

Lossless coding and transfer/draft overlap remain useful auxiliary layers. They
can amortize exact checkpoint bytes across committed tokens. They do not reduce
the dense arithmetic per exact committed token when `N=A`, and speculative
branches worsen the arithmetic floor by `N/A`.

EXP-097A must therefore report both `N` and `A`, reduce the fine dense arithmetic
fraction, and commit the exact successor state. Reproducing the EXP-096A
coefficient words while retaining the complete fine sweep is not a core pass.

## Evidence boundary

The result is E0 model-free accounting. Native 4B p50/p95, physical target
bandwidth/compute, 405B execution, CUDA, 8-GiB allocation, and successor-state
integration remain `NOT TESTED`.
