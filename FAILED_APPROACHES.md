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

Authoritative source: `results/exp_047/summary.json`.

Correctness passed at E1: 525 cases, zero wrong accepts, zero fallback mismatch, zero independent-bound mismatch, and 15/15 adversarial fallback.

Performance failure: 4/525 certificates, 99.238% fallback, N=1024 mean evaluated 98.294%, positive control 10.449%, Python path about 8.6–9.1x full summation, projected target fraction 1.185%.

Decision: retain certificate/fallback reference; reject one global range plus basic Serfling as primary runtime.

## F-014 — Range-based CPTC as core, including oracle-tight and stratified rescue

Authoritative source:

```text
results/exp_047r/summary.json
workflow 30795946233
source head SHA 0beb068e9679c9f4d51d1b210b0eee7fbc325214
artifact SHA-256 6c9a4fdca80d29964eca02d16f8b36f5ca8e211653f6fb9ddfe548a729c6e12d
```

Strongest favorable control:

```text
C1 exact per-state min/max range
3 pinned trained dense checkpoints
18 held-out current-token states
median evaluated fraction 100%
p90 evaluated fraction 100%
wrong accepts 0
```

Deployable candidate control:

```text
C2 checkpoint-span stratified bound
median 100%
p90 100%
best case 254/256 = 99.21875%
bound violations 0
```

The pre-registered rejection thresholds were C1 median <=10% and p90 <=25%. C1 used the realized exact contribution range and still read the complete population in every case. This falsifies the claim that merely tightening sound range metadata or adding variance adaptation can close the core gap.

Permanent decision:

- reject range-based CPTC as a primary execution architecture;
- do not continue C3 empirical-Bernstein/variance tuning as a rescue of EXP-047R;
- retain only the E1 certificate, fault rejection, and exact fallback as auxiliary safety machinery;
- revisit only if a new mechanism independently changes the decision object or amortizes/avoids the full operation before the certificate is applied.

## F-015 — Hard Jacobi target-only block decoding as core

Authoritative source:

```text
results/exp_048/summary.json
workflow 30798936320
artifact SHA-256 67c1e6d8965f7535020ecd4c02bb8a2af1156a234564f3cdf74d10c882fd7eb9
```

EXP-048 B2 used 32-token blocks, fill token zero, and at most four exact target iterations per cycle. Every target pass was charged.

MEASURED:

```text
exact output mismatches 0
p50 target passes per 32 exact tokens 58
p50 accepted tokens per target pass 0.551724
p50 target-equivalent stream fraction 181.25%
p90 target-equivalent stream fraction 193.75%
maximum matching prefix 3
```

Failure: target-only hard Jacobi consumed more full target streams than exact sequential generation. Do not repeat by changing only fill token, block length, or iteration cap while hiding every failed target pass.

Classification: exact control only. Revisit only with a mechanism that changes convergence information per target pass and survives the triangular dependency audit.

## F-016 — Sequential partial-layer self-draft with target LM head as core

EXP-048 B3 used the same unmodified checkpoint's first 1, 2, or 4 layers, final normalization, and full target LM head to generate 32 causal proposal tokens sequentially. All draft layer/head/embedding-equivalent streams, one exact verification stream, rejected positions, and correction were charged.

MEASURED:

```text
3 models × 6 families = 18 cases
54 fixed variants
exact mismatches 0
future information uses 0
best cases with any matching proposal token 4/18
maximum matching proposal prefix 1
p50 committed tokens per target verification 1
minimum fully accounted fraction 1333.463%
p90 fully accounted fraction 2893.843%
```

Failure: early target layers did not predict a useful exact prefix, while the full LM head was reread for every proposed token. The B4 proposal tree is not a valid rescue because it multiplies a failed proposal source and must charge every branch/head evaluation.

Permanent decision:

- reject sequential early-layer self-drafting as the primary runtime;
- reject unrestricted layer-count, temperature, or tree-width tuning as a continuation of EXP-048;
- retain only the exact block verifier;
- revisit proposal generation only when the mechanism avoids per-token sequential target-head cost and supplies a stronger falsification.
