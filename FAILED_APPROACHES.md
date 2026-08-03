# Failed and Demoted Approaches

Permanent anti-repetition register. Revisit only with a mechanism that directly addresses the recorded failure and a stronger falsification.

## F-001 — Static low-rank/generic factorization

Failure: storage occasionally fit projections, but real decisions failed or reads remained close to full stream. Do not repeat by changing only rank/block/basis or hiding residual traffic.

## F-002 — Progressive low precision as primary path

Failure: Q2/Q3 quality failure; Q4 autonomous prefixes negligible; target verification amortization exceeded one thousand accepted tokens.

## F-003 — Independent exact-neuron selection

Failure: at most two exact tokens while traffic exceeded target.

## F-004 — Deterministic signed residual refinement

PR #31–#34 observed cancellation but required roughly 90–98% refinement and hundreds of GiB/token.

## F-005 — Prompt-derived recurrent programs

Failure: reuse near one token; exact autonomous prefixes one or two.

## F-006 — Sparse repair with impossible oracles

Failure: most tokens still repaired; projected roughly 128–169 GiB/token and 552–726 GFLOP/token.

## F-007 — Prompt suffix/nonlocal replay

Failure: future-aware reuse far below required amortization.

## F-008 — Raw exact-prefix graph

Failure: 64 records ->64 unique nodes; held-out first miss at step zero.

Classification: auxiliary exact memoization only.

## F-009 — Future-aware suffix DAG as complete runtime

Positive: 64->38 exact nodes.

Failure: future continuation required; causal held-out start coverage 0%.

Classification: auxiliary body compression.

## F-010 — Metadata size relabeled as traffic

False. Separate total representation, logical bytes, physical transactions, and latency.

## F-011 — Probe count relabeled as latency

False. Small serial probes can be cheap; hardware evidence or a valid lower bound is required.

## F-012 — Small-model evidence promoted to 405B success

Forbidden. Synthetic/small-checkpoint work does not measure target VRAM, 405B, PCIe, SSD, CUDA, TTFT, or tokens/second.

## F-013 — Global-range Serfling CPTC-v1 as primary executor

Authority: `results/exp_047/summary.json`.

Correctness passed at E1. Performance failed: 4/525 certificates, 99.238% fallback, N=1024 mean evaluated 98.294%, positive control 10.449%, Python path about 8.6–9.1x full summation.

Decision: certificate/fallback auxiliary only.

## F-014 — Range-based CPTC oracle/stratified rescue

Authority: `results/exp_047r/summary.json`.

```text
C1 exact per-state range median 100%
C1 p90 100%
C2 median/p90 100%
C2 best 99.21875%
```

Decision: reject range-based CPTC; do not continue variance-only tuning.

## F-015 — Hard Jacobi target-only block decoding

Authority: `results/exp_048/summary.json`.

```text
p50 target passes / 32 exact tokens 58
p50 fraction 181.25%
p90 193.75%
maximum matching prefix 3
```

Do not repeat by changing only fill token, block length, or iteration cap while hiding every failed target pass.

## F-016 — Sequential partial-layer self-draft with target LM head

EXP-048 B3:

```text
18 cases, 54 variants
future information 0
maximum matching prefix 1
p50 committed tokens 1
minimum fraction 1333.463%
p90 2893.843%
```

Do not continue layer/temperature/tree tuning from this failed recursive draft source.

## F-017 — Target-only continuous Picard/Anderson fixed-point generation

Authority: `results/exp_049/summary.json`.

```text
18 cases, 1,458 trajectories
reference-selected p50 prefix 4.5
maximum prefix 6
p90 fraction 168.778596%
hard Jacobi p50 after four passes 4
Anderson p50 after four passes 1
```

Adversarial hidden chains:

```text
Picard prefixes 1,2,3,4
Anderson prefixes 1,2,3,3
hidden suffix transcript indistinguishability true
```

Decision:

```text
REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
```

Forbidden continuation: solver-hyperparameter-only tuning, soft residual convergence relabeled as exact token progress, or a target-only fixed-point GPU backend.

## F-018 — Fixed target-independent external drafting as universal or practical core

Authority:

```text
results/exp_050/summary.json
workflow 30806015309
source head SHA 1388c780abea11067c66cd666ed0a313ec2f682c
artifact 8852817664
artifact ZIP SHA-256 a32ffe8dbfc201c6d70ca8dac660164d8400691ad4d8fe3593d688e7754f6159
```

Universal counterexample:

```text
fixed draft first token 7
arbitrary target first token 8
matching proposal prefix 0
exact correction committed
exact target output preserved
```

Therefore a target-independent fixed draft cannot guarantee a nonzero exact prefix for every arbitrary target.

Practical fixed TinyStories pool:

```text
3 targets
36 target/draft/prompt pairs
108 K rows
exact mismatches 0
target-future uses 0
matching prefix 0 in 72/108 rows
matching prefix 1 in 24/108
matching prefix 2 in 6/108
matching prefix 3 in 6/108
reference-selected p50 prefix 0.5
maximum prefix 3
p90 normalized fraction 163.20987654%
Korean useful acceptance false
structured JSON useful acceptance false
target medians 1.0 / 0.0 / 0.5
```

PROJECTED actual 4B draft requirement:

```text
4/405 + 1/K <=0.01185185185
K >=507 exact proposal tokens before additional overhead
```

Permanent decision:

```text
REJECT_TARGET_INDEPENDENT_EXTERNAL_DRAFT_AS_UNIVERSAL_CORE
```

The tested pool is also rejected as a restricted practical core.

Forbidden continuation:

- expanding the same failed draft pool with a proposal tree;
- selecting drafts using target reference/future tokens while calling the selector deployable;
- ignoring one full draft forward per proposed token;
- reporting exact correction tokens as accepted draft tokens;
- claiming narrative/code successes cover Korean/JSON failures;
- replacing the fixed pool with another arbitrary small model without a pre-registered stronger rationale and universal claim restriction.

Allowed reuse:

- exact block verifier;
- external-draft accounting/reference tests;
- universal first-token counterexample;
- restricted-domain draft research only with an explicit non-universal claim and a materially different evidence base.
