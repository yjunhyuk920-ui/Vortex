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

Forbidden. Synthetic/TinyLlama work does not measure target VRAM, 405B, PCIe, SSD, CUDA, TTFT, or tokens/second.

## F-013 — Global-range Serfling CPTC-v1 as primary executor

Authoritative source:

```text
results/exp_047/summary.json
```

Frozen summary currently records workflow `30793232558` and source SHA `74ac92e9b1c8fffbc50a2322d9b36dd3c05f0d79`.

Correctness:

```text
525 cases
wrong accepts 0
fallback mismatches 0
independent-bound mismatches 0
adversarial fallback 15/15
```

Performance failure:

```text
certified 4/525
fallback 99.238%
N=64/128/256 mean evaluated 100%
N=512 mean evaluated 98.519%
N=1024 mean evaluated 98.294%
positive control 10.449%
Python optimized/reference about 8.6–9.1x
simple projected target fraction before overhead 1.185%
```

Decision:

- retain certificate/fallback code as E1 reference;
- reject one global range plus basic Serfling alpha spending as primary runtime;
- do not build a full backend from CPTC-v1.

Allowed next work:

- non-deployable oracle-tight real-state range audit;
- deployable checkpoint-derived stratified bounds;
- independently proven variance-adaptive finite-population bounds.

If oracle-tight held-out real-checkpoint contributions still require high tile fractions, reject range-only CPTC entirely rather than tune sample limits or delta.
