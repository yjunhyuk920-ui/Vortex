# Next Experiment

## Closed Gate — EXP-068

EXP-068 changed execution class from static weight structure to activation-conditioned exact demand certification. It evaluated a necessary condition that was deliberately more favorable than any deployable runtime:

```text
all preceding Transformer operations and weight reads were free
the complete winning LM-head row was free
all bound and reveal-order metadata was free
each competitor token received its own independently optimal coordinate order
```

Only competitor-row LM-head entries that remained mathematically necessary under exact absolute unread bounds were charged against the whole-model dense-weight baseline.

Authoritative coverage and correctness:

```text
3 pinned models
18 model/prompt cases
6 required families
153/153 frozen source tensor hashes matched
full-model/direct-head winner mismatches: 0
bound violations: 0
control failures: 0
```

Favorable necessary lower bound:

```text
p50 weight-read fraction: 13.7696858262%   FAIL against 10%
p90 weight-read fraction: 19.2524013315%   PASS against 25%
p50 operation fraction:   13.7696858262%   FAIL against 10%
p90 operation fraction:   19.2524013315%   PASS against 25%
minimum case:              10.1376199755%
maximum case:              21.0055007890%
```

Decision:

```text
REJECT_GLOBAL_DEMAND_CERTIFICATE_AS_CORE_RETAIN_BOUND_AUDITOR_AUXILIARY
```

The final output head alone exceeds the whole-model p50 budget after every preceding layer and the winning row are granted for free. Full-network propagation, a scheduler, and a kernel can only add work. The registered exact norm/absolute-unread demand family is therefore closed as a primary core. Tile-size or ordering tuning cannot reopen it.

Authority:

```text
results/exp_068/summary.json
results/exp_068/raw/case_rows.jsonl
results/exp_068/raw/control_rows.jsonl
results/exp_068/raw/tensor_rows.jsonl
results/exp_068/evidence_manifest.json
workflow 30918865952
artifact 8896230736
artifact ZIP SHA-256 ff0f4398c0d162142d3e71d6864a3990704a14bf59e007182c9dce72c913835f
```

## EXP-069 — Causal Exact Temporal-Span Replay Gate

### Execution-class change

EXP-069 introduces a new exact information source: projection inputs and outputs that were already computed for earlier tokens in the same causal session.

For a fixed dense projection `y = W x`, cache exact pairs:

```text
(x_1, y_1), (x_2, y_2), ... where y_t = W x_t
```

If a later input is exactly in the span of cached inputs,

```text
x_t = sum_k c_k x_k
```

then exact-real linearity gives

```text
y_t = sum_k c_k y_k
```

without reading `W` again. The same idea may be expressed as an affine span of temporal deltas. This is not activation sparsity: every vector may be dense. It is not KV duplication: vectors may differ while remaining exactly dependent. It is not static low rank of `W`: the structure, if any, belongs to the causal input trajectory.

### Why this class is allowed

Potential upside:

- full dense weight reads can be replaced by cached vector combinations on dependent arrivals;
- unchanged arbitrary model weights are supported;
- no training, approximation, or future-token oracle is required;
- the cache is derived online from exact computations already performed.

Reasons for low prior probability:

- dense hidden states may add one independent direction almost every token;
- exact dependence is much stricter than numerical similarity;
- coefficient discovery and cached-output combination may erase any traffic saving;
- once the span reaches input width, cached basis outputs approach the size and work of the original matrix.

The Gate therefore stops at exact independence and favorable accounting before any replay runtime is implemented.

### Exact dyadic-rank certificate

Every captured float32 scalar is an exact dyadic rational. Decode each value into sign, integer significand, and power-of-two exponent. For odd registered primes, map the dyadic rational exactly into the finite field using modular powers and inverses of two.

For each projection and causal trace, maintain incremental ranks under at least three primes. If adding `x_t` raises rank under any prime, `x_t` is certainly independent over the rationals and cannot be reconstructed from previous exact inputs. This gives a fail-closed lower bound on mandatory full projection passes.

A non-increase under the registered primes is not automatically claimed as exact dependence. Only a separately reconstructed and verified coefficient witness may count as a replay hit.

### Population

Use unchanged pinned TinyStories-1M/3M/8M checkpoints and causal warm-decode traces from the six required families:

```text
English narrative
code
mathematics
identifier boundary
Korean
structured JSON
```

Capture the input to every registered dense projection for at least 64 exact greedy decode tokens per model/family case, or until the model terminates. Prefix/prefill vectors must be reported separately from warm decode.

### Favorable oracle accounting

For every projection call:

- a certified independent arrival charges one complete dense weight read and dense matrix-vector multiply;
- a verified dependent arrival charges no weight read, but charges coefficient discovery, cached input reads, cached output reads, and output combination operations;
- basis insertion, rank metadata, coefficients, and cached vectors are charged;
- an impossible oracle may choose the most favorable exact prior basis, but may not use future vectors;
- any unverified dependence falls back to a full exact pass.

Report:

```text
certified-independent arrival fraction
verified exact replay-hit fraction
weight-byte fraction
operation fraction
basis rank growth by token
basis/cache storage
first exact replay-hit position
```

### Controls

- a registered low-dimensional exact recurrence produces replay hits after basis formation;
- repeated and affine-dependent vectors replay exactly;
- random dyadic vectors increase rank until dimension saturation;
- a triangular causal sequence forces one new direction per token;
- modular independence witnesses agree across registered primes;
- reconstructed dependencies reproduce the source input exactly before output replay is credited;
- no future-token state, near-equality threshold, or approximate coefficient is allowed.

### Promotion Gate

```text
zero trace/reference/rank/witness/control mismatch
100% registered dense-projection and family coverage
p50 mandatory full-weight-read fraction <=10%
p90 mandatory full-weight-read fraction <=25%
p50 total operation fraction <=10%
p90 total operation fraction <=25%
verified exact replay-hit fraction sufficient in every family
projected cache/storage compatible with the 8 GiB final target
no largest-model degradation >25%
```

Passing the rank/replay Gate authorizes a bitwise floating-point replay and cache-layout Gate. It does not authorize a physical kernel or a 405B claim.

### Failure decision

```text
REJECT_CAUSAL_EXACT_TEMPORAL_SPAN_REPLAY_AS_CORE_RETAIN_DYADIC_RANK_AUDITOR_AUXILIARY
```

On failure, exact temporal linear-span reuse is closed as a primary core. It may not be rescued by numerical tolerances, longer traces selected after observation, cross-prompt future data, or uncharged coefficient/cache work.

### Stop rule

Before survival, prohibit:

```text
GPU replay kernels
model-wide cache integration
approximate subspace projection
learned temporal adapters
cross-session oracle dictionaries
405B implementation work
```

### Claim boundary

Phase A/B/C small-model causal trajectory evidence, ceiling E1. Bitwise floating-point replay equivalence, a physical cache/replay kernel, actual 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/second remain NOT TESTED.
