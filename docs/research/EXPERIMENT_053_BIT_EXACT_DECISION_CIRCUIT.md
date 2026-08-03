# EXP-053 — Automatic Bit-Exact Decision-Circuit Compiler Gate

## Status

```text
Implementation branch: research/exp-053-bit-exact-decision-circuit
Gate registration: COMMITTED BEFORE RUN
Phase: A/B
Evidence ceiling: E1
Real Transformer operation replacement: false
Phase D: NOT TESTED
```

## Mechanism change

EXP-052 rejected state-enumerative exact prefix/KV advice. EXP-053 does not store runtime states. It compiles immutable target weights and exact bounded arithmetic semantics into a structurally hashed AND-inverter graph:

```text
signed quantized weights + modular accumulator contract
        |
        v
bit-vector constant-add and signed-comparison compiler
        |
        v
algebraic simplification + structural hashing
        |
        v
exact AIG binary representation
        |
        v
exact circuit query OR declared exact fallback
```

Exhaustive input enumeration is used only to validate equivalence on the finite registered domains. It is not stored in the circuit and is not compiler input.

## Bounded operator contract

Input:

```text
n binary activation bits x_i in {0,1}
C output classes
signed integer weights w[c,i] and biases b[c]
W-bit two-complement modular accumulator
```

Reference scores:

```text
s_c = wrap_W(b[c] + sum_i w[c,i] * x_i)
```

Output:

```text
argmax_c signed(s_c)
```

Ties select the lower class index. The compiled circuit must reproduce this result for every one of the `2^n` inputs.

This operator is a bounded quantized linear decision kernel, not yet a full Transformer layer.

## Q0 — independent arithmetic reference

The reference uses integer arithmetic followed by explicit W-bit two-complement wrapping and strict signed top-1 selection.

## Q1 — weight-derived AIG compiler

- one input AIG literal per activation bit;
- constants use AIG false/true literals;
- each nonzero signed weight is conditionally added as a W-bit two-complement constant;
- score comparisons are strict signed comparators;
- class-index outputs use exact mux chains;
- no runtime input assignment is observed during compilation.

## Q2 — exact structural reduction

The builder performs:

```text
x &0 ->0
x &1 ->x
x &x ->x
x &~x ->0
commutative operand canonicalization
exact structural hashing of identical AND nodes
```

Metrics separate:

```text
requested_and_count
unique_and_node_count
reachable_and_node_count
raw bit-blast baseline bytes
reachable query bytes
```

The query fraction denominator is the same exact bit-blasted arithmetic before structural hashing, after constant/algebraic trivial simplification. It is not a measured Transformer FLOP or hardware-byte baseline.

## Q3/Q4 operator families

Every family is compiled for fixed-class scaling cases:

```text
(n=8, C=4, W=8)
(n=12, C=4, W=12)
(n=16, C=4, W=16)
(n=20, C=4, W=16)
```

Class sweep controls:

```text
(n=12, C=2, W=12)
(n=12, C=8, W=12)
```

Families:

### sparse_structured

Each class depends on one or two distinct activation bits. This is a favorable structural control.

### low_rank_structured

Class rows are integer multiples of one shared base vector. This tests whether shared algebraic structure survives bit blasting.

### dense_random

Every class/input weight is a nonzero deterministic random signed integer. This is the primary adversarial scaling family.

### late_bit

The exact class depends only on the final input bit, with other classes suppressed. This validates late-bit dependence and aggressive simplification.

## Q5 — exact validation and query

All registered circuits are validated over their complete finite input domains, including all `2^20 =1,048,576` assignments for 20-bit cases.

Packed AIG evaluation uses bit masks only as a validator. The saved runtime representation contains inputs, output literals, AND nodes, requested-node count, and source-parameter count; it contains no truth table.

Every mismatch records the first counterexample and fails the integrity Gate.

## Accounting

For each circuit:

```text
compile elapsed time
source scalar parameter count
nonzero weight count
requested AND count
unique AND count
reachable AND count
serialized circuit bytes
reachable query bytes
query node fraction
query byte fraction
exhaustively validated assignments
compile-equivalent query count
projected 405B circuit bytes
```

Logical compile-equivalent queries:

```text
requested_and_count / max(1, reachable_and_count)
```

Required repetitions to amortize compile construction below a fraction `f`:

```text
ceil(requested_and_count / (f * max(1, reachable_and_count)))
```

This is structural gate-operation accounting, not wall-clock target-stream evidence.

Projection:

```text
projected_405B_circuit_bytes
  = circuit_bytes / source_parameter_count * 405,000,000,000
```

The projection is deliberately conservative and labeled PROJECTED. It does not measure a 405B checkpoint.

## Growth audit

For each family at fixed `C=4`, fit:

```text
log2(reachable_AND_nodes +1) = intercept + slope * input_bits
per_added_bit_growth = 2^slope
```

Structured success cannot erase dense-random or late-bit behavior.

## Pre-registered early rejection Gate

Reject the bit-exact circuit compiler as the primary runtime if any condition holds:

```text
exact mismatch >0
truth table stored as representation
p50 query node fraction >10%
p90 query node fraction >25%
p50 query byte fraction >10%
p90 query byte fraction >25%
projected circuit bytes >1 TiB before target scale
adversarial per-added-bit growth >1.5
random-dense p50 query fraction >25%
compile-equivalent queries >1,000,000
```

Failure decision:

```text
REJECT_BIT_EXACT_DECISION_CIRCUIT_COMPILER_AS_CORE_RETAIN_AIG_REFERENCE_AUXILIARY
```

A negative result forbids relabeling the unreduced circuit as compression, using exhaustive truth tables as compiled output, or reporting structured controls without adversarial dense results.

## Promotion boundary

A successful synthetic Gate would still require:

```text
real small-checkpoint operation replacement
exact model output agreement
p90 fully-accounted target fraction <=1.185185%
compiler/storage/query costs charged
non-degrading medium/large scaling
8 GiB hot-state closure
Phase-D hardware evidence
```

## Evidence boundary

```text
405B execution: NOT TESTED
8 GiB VRAM: NOT TESTED
CUDA/PCIe/SSD/TTFT/tokens per second: NOT TESTED
real Transformer operation replacement: false
Phase D: NOT TESTED
```

## Commands

```bash
python -m pytest -q tests/exp_053
bash experiments/exp_053/run_current_env.sh
bash experiments/exp_053/reproduce.sh
```

<!-- EXP-053-AUTHORITATIVE-FINAL -->
## Final authoritative result

Authority: `results/exp_053/summary.json`; workflow `30814648709`; source head `325cc694d4b2e88e34dba5ba8e980e3970c34c66`; workflow merge `4ecca6405f549fc9a05d7ad17cfe1d7c3a9c3398`; artifact `8856213147`; ZIP SHA-256 `eb7ecf8f284cc974d62e03bee767892666160abfae79a70bb32446f0dfe95178`.

24 weight-derived circuits were exhaustively checked over 4,506,624 inputs with zero output-bit mismatch and no truth-table representation. Structural hashing left p50/p90 reachable fractions 0.84168345/0.94107229; dense-random p50 was 0.92452096. The maximum 405B source-parameter circuit projection was 255.5966 TiB. Late-bit controls simplified to zero AND nodes, but sparse controls still retained 65–78% of the exact bit-blast and projected 3.17–7.45 TiB. Growth and compile-amortization Gates passed; node, byte, storage, and random-dense Gates failed.

Decision: `REJECT_BIT_EXACT_DECISION_CIRCUIT_COMPILER_AS_CORE_RETAIN_AIG_REFERENCE_AUXILIARY`.
