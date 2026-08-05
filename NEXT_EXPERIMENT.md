# Next Experiment

## Closed Audit — EXP-071

EXP-071 audited whether two primary unconditional online matrix-vector lower-bound results can rigorously rule out the registered conventional exact VORTEX runtime model.

Source-audited population:

```text
CGL15 Theorem 3
CKL18 Theorem 1.2
CKL18 Theorem 1.3
```

Exact reduction and registered-plan integrity:

```text
1,052,740 exhaustive binary matrix/vector cases
4,164 direct float32 replay cases
reduction mismatches 0
control failures 0
9 Llama-405B tensor families
884 tensor instances
405,849,243,648 registered parameters
```

Applicability result:

```text
largest valid square subproblem n          16,384
CKL18 maximum side-information range       67,108,864 bits = 8 MiB
registered hot/side-information allowance  68,719,476,736 bits = 8 GiB
allowance / theorem range                  1,024x
CKL18-covered tensor families              0 / 9
model-wide direct-sum theorem              not established
finite constants hidden by Omega           unresolved
```

Even the deliberately non-authoritative unit-constant CGL15 indicator, illegally summed as if a direct-sum theorem existed, was only `0.0248972%` of packed Q4 cells. This is diagnostic only and is not a certified finite lower bound.

Decision:

```text
INSUFFICIENT_LOWER_BOUND_DO_NOT_CLAIM_IMPOSSIBILITY
```

This does not establish feasibility. It establishes only that the registered papers do not justify a 405B impossibility claim under the full 8 GiB jointly computed side-state regime.

Authority:

```text
results/exp_071/summary.json
results/exp_071/raw/theorem_hypotheses.jsonl
results/exp_071/raw/tensor_rows.jsonl
results/exp_071/raw/control_rows.jsonl
results/exp_071/processed/direct_sum_audit.json
workflow 30965323458
artifact 8914506737
artifact ZIP SHA-256 bc81e90e3b5a35935f893ad7396d4b41a13de46606ce14bccc53cf79e30e8ba4
```

Permanent restrictions:

- do not divide the shared 8 GiB state by tensor/layer count without a direct-sum theorem;
- do not add independent per-matrix asymptotic lower bounds without a composition proof;
- do not set hidden asymptotic constants to one and call the result certified;
- do not equate cell probes with physical GPU, PCIe, or SSD transactions;
- do not claim that all exact software runtimes are impossible from EXP-071.

## EXP-072 — Exact Nonlocal Q4 Shared Arithmetic-DAG Synthesis Gate

### Execution-class change

EXP-070 tested only contiguous short coefficient patterns. EXP-067 tested only whole-row equality/sign/proportional reuse and common-right rank. Neither test covered a general exact straight-line arithmetic program that can share **non-contiguous intermediate linear forms** across many output rows and across projections consuming the same activation.

For symbolic input coordinates `x_j`, a circuit node may form an exact integer linear form:

```text
z_k = a * z_i + b * z_j
```

where `a,b` are bounded registered small integers and the resulting symbolic coefficient vector is verified exactly. Outputs must reconstruct every requested Q4 row as an integer coefficient vector. The circuit may discover reusable forms such as:

```text
x_3 - x_19
2*x_7 + x_41
(x_3 - x_19) + (2*x_7 + x_41)
```

These forms need not correspond to contiguous blocks, identical rows, low-rank factors, or a fixed tensor decomposition.

### Why this class remains eligible

- Q4 coefficients come from a small alphabet;
- many rows consume the same activation vector within one projection or Q/K/V group;
- arbitrary linear circuits can share partial expressions that block dictionaries cannot see;
- exact symbolic reconstruction is possible without training or activation approximation;
- a positive result would expose a true smaller executable program rather than only a compressed byte layout.

Prior probability remains low because general dense matrices usually have high linear-circuit complexity. Therefore the experiment is bounded and must stop before kernel work unless the full Gate passes.

### Registered synthesis scopes

Use the unchanged frozen real-Q4 population and checksums. Evaluate only preregistered scopes:

```text
single projection tiles
complete Q/K/V groups sharing one input
complete Gate/Up groups where present
```

Registered column widths:

```text
16, 32, 64
```

Registered output-row tile heights:

```text
16, 32, 64
```

No post-result tile size, model, matrix role, or selected favorable subset may promote the candidate.

### Registered circuit search

Implement deterministic bounded search families only:

1. pair-frequency common-subexpression elimination over signed linear forms;
2. bounded beam search seeded by the highest-support exact pair forms;
3. addition-chain seeds for Q4 constants `-8..7`;
4. joint-output search for matrices sharing an identical runtime input.

Every candidate node is represented by its full exact integer coefficient vector during compilation. Hash matches must be confirmed by full-vector equality. Cycles, approximate coefficient matches, floating tolerances, and activation-derived search are forbidden.

### Accounting

Report separately:

```text
operation fraction
query-description byte fraction
static circuit-storage fraction
compiler work and peak memory
```

Charge at minimum:

- every runtime add, subtract, small-constant multiply, and output accumulation;
- every required input load not already resident under the registered baseline;
- circuit opcodes, operand IDs, constants, output maps, tile maps, and row scales;
- repeated execution of a circuit template on different activation values;
- cross-tile output accumulation;
- dense fallback for any unsynthesized row.

One plan per scope must minimize the maximum of operation, query-byte, and static-storage fractions. Per-axis cherry-picking is prohibited.

### Exactness controls

- exhaustive symbolic reconstruction for every emitted circuit;
- random activation replay in exact integer arithmetic;
- float32 replay reported separately without promoting changed reduction order to bitwise equivalence;
- Hadamard/butterfly and hand-shared linear-form positive controls;
- forced-unique and dense-random Q4 negative controls;
- one-coefficient mutation must invalidate or alter the expected circuit;
- deterministic reruns must produce identical circuit and evidence hashes.

### Promotion Gate

```text
zero checksum/reconstruction/collision/control mismatch
100% registered population coverage
p50 operation fraction <=10%
p90 operation fraction <=25%
p50 query-description fraction <=10%
p90 query-description fraction <=25%
p50 static-circuit fraction <=10%
p90 static-circuit fraction <=25%
no required matrix role or shared-input group with p90 >25%
no largest-model degradation >25%
```

Passing authorizes only a bitwise floating replay-order Gate and a physical packed-circuit kernel Gate. It does not authorize a 405B or 8 GiB claim.

### Failure decision

```text
REJECT_EXACT_NONLOCAL_Q4_ARITHMETIC_DAG_AS_CORE
RETAIN_LINEAR_CIRCUIT_SYNTHESIZER_AUXILIARY
```

On failure, do not rescue the family by hiding circuit bytes, counting compiler work as free amortization without a query threshold, reporting positive structured controls as real-model evidence, selecting only Q/K/V or only favorable tiles, or using approximate linear forms while claiming exactness.

### Stop rule

Before survival, prohibit:

```text
CUDA circuit kernels
model-wide circuit transcoding
unbounded SAT/SMT synthesis
learned or activation-specific circuits
floating-point semantic claims
405B implementation work
```

### Claim boundary

Phase A/B/C real-Q4 symbolic-circuit evidence, ceiling E1. Floating-point reduction-order equivalence, a physical circuit kernel, actual Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/second remain **NOT TESTED**.
