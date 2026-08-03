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
