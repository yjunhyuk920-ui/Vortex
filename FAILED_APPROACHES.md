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

Correctness passed at E1: 525 cases, zero wrong accepts, zero fallback mismatch, zero independent-bound mismatch, 15/15 adversarial fallback.

Performance failure: 4/525 certificates, 99.238% fallback, N=1024 mean evaluated 98.294%, positive control 10.449%, Python path about 8.6–9.1x full summation, projected target fraction 1.185%.

Decision: retain certificate/fallback reference; reject one global range plus basic Serfling as primary runtime.

## F-014 — Range-based CPTC as core, including oracle-tight and stratified rescue

Authority: `results/exp_047r/summary.json`, workflow `30795946233`.

```text
C1 exact per-state range median 100%
C1 p90 100%
C2 median/p90 100%
C2 best 254/256 = 99.21875%
wrong accepts 0
bound violations 0
```

The pre-registered limits were 10%/25%. The exact realized range oracle still read the complete population.

Permanent decision:

- reject range-based CPTC as primary execution architecture;
- do not continue C3 empirical-Bernstein/variance tuning as rescue;
- retain only certificate, fault rejection, and exact fallback;
- revisit only after another mechanism changes the decision object or amortizes the operation first.

## F-015 — Hard Jacobi target-only block decoding as core

Authority: `results/exp_048/summary.json`, workflow `30798936320`.

```text
exact mismatches 0
p50 target passes / 32 tokens 58
p50 accepted tokens per pass 0.551724
p50 fraction 181.25%
p90 fraction 193.75%
maximum matching prefix 3
```

Failure: more full target streams than exact sequential generation. Do not repeat by changing only fill token, block length, or iteration cap while hiding failed passes.

Classification: exact control only.

## F-016 — Sequential partial-layer self-draft with target LM head as core

EXP-048 B3 used the same target's first 1/2/4 layers, final norm, and full LM head to generate 32 tokens sequentially. All draft and verification costs were charged.

```text
18 cases, 54 variants
exact mismatches 0
future information 0
cases with any matching proposal token 4/18
maximum matching prefix 1
p50 committed tokens 1
minimum accounted fraction 1333.463%
p90 fraction 2893.843%
```

Permanent decision:

- reject sequential early-layer self-draft as primary runtime;
- reject unrestricted layer/temperature/tree tuning as continuation;
- retain exact block verifier;
- revisit only when proposal generation avoids per-token target-head cost and supplies a stronger information source.

## F-017 — Target-only continuous Picard/Anderson fixed-point generation as core

Authority:

```text
results/exp_049/summary.json
workflow 30803672059
source head SHA 91d0caa86d784c663bc520d36d9b512f0cc526e9
artifact 8851957250
artifact ZIP SHA-256 4cd6c8c4afb833562438a97f052d45d331f3691362472fb08e594bd0c5585b9e
```

EXP-049 tested:

- hard synchronous Jacobi;
- fixed damped Picard with top-k 1/8, damping 0.5/1.0, zero/last/next-repeat initialization;
- bounded Anderson histories 2/4/8 with float64 solve, regularization, clipping, condition checks, and fail-closed Picard fallback;
- blocks 64/128/256 and target-pass checkpoints 1/2/4;
- three pinned TinyStories targets and six held-out families;
- exact-reference selection of the best pre-registered S1/S2 trajectory per case;
- two hidden triangular causal chains.

MEASURED favorable checkpoint result:

```text
18 cases
1,458 trajectory rows
exact mismatches 0
future target uses in S1/S2 0
unhandled numerical failures 0
oracle-best p50 matching prefix 4.5
oracle-best maximum prefix 6
oracle-best p90 target-equivalent fraction 168.778596%
model medians 4.5 / 5.0 / 4.0
all selected rows used 4 target passes and K=64
17/18 selected hard top-1 Picard; no Anderson selected
```

MEASURED controls:

```text
hard Jacobi p50 prefix after 4 passes 4
Anderson p50 prefix after 4 passes 1
Anderson/Jacobi improvement 0.25x
```

MEASURED adversarial result:

```text
Picard prefixes by round 1,2,3,4
Anderson prefixes by round 1,2,3,3
hidden suffix transcript indistinguishability true
one-new-exact-position-per-round barrier observed true
```

Failure:

- the strongest non-deployable reference-selected upper bound missed the 16-token early prefix Gate;
- p90 logical traffic was 168.78%, not <=10%;
- Anderson was worse than Jacobi, not >=4x better;
- continuous mixing did not reveal adversarial hidden predecessor information.

Permanent decision:

```text
REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
```

Allowed reuse:

- numerical reference/fault tests;
- Picard/Anderson positive controls;
- exact block verifier;
- triangular adversarial family for future universal claims.

Forbidden continuation:

- tuning only top-k, temperature, damping, initialization, Anderson history, block size, or iteration count;
- claiming average real-model prefixes override the arbitrary-model adversarial barrier;
- hiding the exact-reference variant selector;
- counting soft residual convergence as exact token acceptance;
- implementing a GPU backend from this negative Gate.

Revisit only if a new mechanism imports future information not present in target-only synchronous transcripts, or explicitly changes the universal/exact mission contract.
