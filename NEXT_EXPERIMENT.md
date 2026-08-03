# Next Experiment

## Closed Gate — EXP-053

Authority: `results/exp_053/summary.json`; workflow `30814648709`; source head `325cc694d4b2e88e34dba5ba8e980e3970c34c66`; workflow merge `4ecca6405f549fc9a05d7ad17cfe1d7c3a9c3398`; artifact `8856213147`; ZIP SHA-256 `eb7ecf8f284cc974d62e03bee767892666160abfae79a70bb32446f0dfe95178`.

24 weight-derived circuits were exhaustively checked over 4,506,624 inputs with zero output-bit mismatch and no truth-table representation. Structural hashing left p50/p90 reachable fractions 0.84168345/0.94107229; dense-random p50 was 0.92452096. The maximum 405B source-parameter circuit projection was 255.5966 TiB. Late-bit controls simplified to zero AND nodes, but sparse controls still retained 65–78% of the exact bit-blast and projected 3.17–7.45 TiB. Growth and compile-amortization Gates passed; node, byte, storage, and random-dense Gates failed.

Decision:

```text
REJECT_BIT_EXACT_DECISION_CIRCUIT_COMPILER_AS_CORE_RETAIN_AIG_REFERENCE_AUXILIARY
```

## EXP-054 — Exact Reduced Ordered Decision-Diagram Gate

### Mechanism change

Compile the same bounded signed modular top-1 operators into a reduced ordered multi-terminal decision diagram (ROMTDD/ROBDD-like representation). Unlike EXP-053 AIGs, runtime evaluates one variable-dependent root-to-terminal path rather than every reachable gate.

The compiler may use exact residual score states and Shannon decomposition, but it may not store an explicit input-to-output truth table.

### Conditions

```text
D0 independent arithmetic reference
D1 natural input-bit variable order
D2 deterministic weight-magnitude variable order
D3 exact unique-table reduction: low==high elimination and (var,low,high) sharing
D4 sparse/low-rank controls
D5 dense-random and late-bit adversaries
D6 exact path evaluator and exhaustive finite-domain equivalence
```

For each operator, compile both registered variable orders. A fixed weight-derived selector may retain the smaller diagram, but compile visits/time for both orders are charged.

### Registered domains

Use the EXP-053 operator families and scaling matrix, with an early safety ceiling:

```text
input bits 8, 12, 16, 20
classes 2, 4, 8
accumulator widths 8, 12, 16
maximum compile states/nodes per order 2,000,000
```

A ceiling hit is a scientific failure row with exact fallback, not an infrastructure crash.

### Accounting

```text
recursive compile-state visits
memoized residual states
unique decision nodes
terminal count
serialized bytes
both-order compile time/bytes
p50/p90 root-to-terminal input probes
query probe fraction = path probes / input bits
fallback on ceiling or corruption
405B source-parameter storage projection
node growth per added input bit
```

### Early rejection Gate

```text
exact mismatch >0
explicit truth table stored as representation
p50 query probe fraction >10%
p90 query probe fraction >25%
any dense-random 20-bit case exceeds 2,000,000 compile states/nodes
projected diagram storage >1 TiB
adversarial node-growth multiplier >1.5 per added input bit
variable-order search cost cannot be amortized within 1,000,000 queries
fallback/ceiling rate >0
```

Failure decision:

```text
REJECT_EXACT_REDUCED_DECISION_DIAGRAM_AS_CORE_RETAIN_BDD_REFERENCE_AUXILIARY
```

### Promotion boundary

Synthetic success still requires real small-checkpoint operation replacement, exact output agreement, p90 fully-accounted target fraction <=1.185185%, non-degrading scale, 8 GiB hot-state closure, and Phase-D measurement.

### Evidence boundary

```text
Phase A/B; evidence ceiling E1
real Transformer operation replacement NOT TESTED
405B / 8 GiB / CUDA / PCIe / SSD / TTFT / tokens/sec NOT TESTED
```

### Next exact action

1. implement exact reduced multi-terminal decision diagrams from residual arithmetic states;
2. add natural and weight-magnitude variable orders;
3. enforce compile-state/node ceilings with exact fallback;
4. exhaustively validate all completed finite domains;
5. measure path probes, storage, growth, and order-selection cost;
6. freeze diagrams, checksums, raw rows, and decision.
