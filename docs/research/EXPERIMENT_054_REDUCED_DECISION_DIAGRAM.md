# EXP-054 — Exact Reduced Ordered Decision-Diagram Gate

## Status

```text
Implementation branch: research/exp-054-reduced-decision-diagram
Gate registration: COMMITTED BEFORE RUN
Phase: A/B
Evidence ceiling: E1
Real Transformer operation replacement: false
Phase D: NOT TESTED
```

## Mechanism change

EXP-053 exact AIGs evaluated nearly every reachable bit-level gate. EXP-054 compiles the same immutable bounded operators into reduced ordered multi-terminal decision diagrams:

```text
signed weights + modular score contract
        |
        v
exact residual-score Shannon compiler
        |
        +--> memoized (order position, wrapped score tuple)
        +--> low==high elimination
        +--> unique (variable, low, high) nodes
        |
        v
one exact root-to-terminal query path
```

The compiled representation contains terminals, ordered decision nodes, a root reference, and compile accounting. It contains no input-to-output truth table.

## Variable orders

Every operator is compiled twice:

```text
D1 natural order: input index 0..n-1
D2 weight-magnitude order: descending sum_c |w[c,i]|, then input index
```

The smaller completed diagram may be selected using only compiled byte/node counts. Both order compilation state visits and time are charged.

## Exact compiler state

At order position `k`, the compiler state is:

```text
(k, wrapped partial score for every class)
```

The low branch leaves scores unchanged. The high branch adds the selected input column modulo the registered accumulator width. At the leaf, strict signed top-1 with lower-class tie break returns a terminal class.

Reduction:

```text
low == high -> child
identical (variable, low, high) -> shared node
```

## Compile ceiling

Each variable order has a hard ceiling:

```text
2,000,000 recursive compile-state visits or unique nodes
```

A ceiling hit returns no partial diagram, records `ceiling_hit=true`, and requires exact fallback. It is a scientific failure row, not an infrastructure exception.

## Corpus

Use the exact EXP-053 operator generator and matrix:

```text
families: sparse_structured, low_rank_structured, dense_random, late_bit
input scaling: n=8/12/16/20, C=4, W=8/12/16/16
class sweep: n=12, C=2/8, W=12
24 operators *2 variable orders
```

All completed diagrams are exhaustively verified over their complete finite domains. The 20-bit domains contain 1,048,576 assignments each.

## Query accounting

For each exact query:

```text
path probes = number of decision nodes visited
query probe fraction = path probes / input bits
```

Report p50/p90/max path probes over the entire finite domain for each completed order and selected diagram.

This fraction is relative to reading all registered binary activation inputs. It is not a measured Transformer parameter-byte or hardware latency fraction.

## Compile and storage accounting

Record:

```text
compile state visits
memoized residual states
unique decision nodes
reachable terminals
serialized bytes
both-order compile time and visits
selected order
exact validation assignments and mismatches
p50/p90/max path probes
compile-order required queries
405B source-parameter storage projection
```

Order-search amortization:

```text
required_queries = ceil(
  total_compile_state_visits_both_orders /
  (required_target_fraction * input_bits * class_count)
)
```

This treats one direct linear decision query as `input_bits * class_count` scalar input-weight interactions. It is an E1 logical compiler-cost audit, not a hardware measurement.

Storage projection:

```text
selected_diagram_bytes / source_parameter_count * 405,000,000,000
```

## Growth audit

For each family at fixed C=4:

```text
log2(selected_nodes +1) = intercept + slope * input_bits
multiplicative growth per added bit =2^slope
```

## Pre-registered rejection Gate

Reject exact reduced decision diagrams as the primary runtime if any condition holds:

```text
exact mismatch >0
explicit truth table stored as representation
p50 query probe fraction >10%
p90 query probe fraction >25%
any compile ceiling/fallback row >0
projected diagram storage >1 TiB
adversarial growth multiplier >1.5 per added input bit
required order-search amortization >1,000,000 queries
```

Failure decision:

```text
REJECT_EXACT_REDUCED_DECISION_DIAGRAM_AS_CORE_RETAIN_BDD_REFERENCE_AUXILIARY
```

Structured success cannot erase dense-random or low-rank failures. A negative result forbids reporting only late-bit diagrams, hiding second-order compile cost, or treating exhaustive validation as the compiled representation.

## Promotion boundary

Synthetic success would still require:

```text
real small-checkpoint operation replacement
exact output agreement
p90 fully-accounted target fraction <=1.185185%
physical diagram lookup/storage accounting
non-degrading medium/large scaling
8 GiB hot-state closure
Phase-D hardware evidence
```

## Claim boundary

```text
405B execution: NOT TESTED
8 GiB VRAM: NOT TESTED
CUDA/PCIe/SSD/TTFT/tokens per second: NOT TESTED
real Transformer operation replacement: false
Phase D: NOT TESTED
```

## Commands

```bash
python -m pytest -q tests/exp_054
bash experiments/exp_054/run_current_env.sh
bash experiments/exp_054/reproduce.sh
```
