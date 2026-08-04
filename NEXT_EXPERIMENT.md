# Next Experiment

## Closed Gate — EXP-066

EXP-066 reused checksum-verified EXP-065 Kronecker ranks and EXP-058 full-matrix ranks, then propagated exact adjacent TT-rank lower bounds across 4,384 preregistered plans.

Authoritative result:

```text
operation p50 3.8941375969%   PASS
operation p90 6.7788461538%   PASS
storage p50  11.0523897059%   FAIL against 10%
storage p90  22.9882812500%   PASS
```

Decision:

```text
REJECT_REAL_Q4_TT_MPO_BOND_RANK_AS_CORE_RETAIN_MPO_CERTIFIER_AUXILIARY
```

The favorable storage lower bound already fails. Unresolved ranks and implementation costs can only increase it. Exact classical single-matrix TT/MPO is therefore closed as a primary core for the measured real-Q4 population. No core reconstruction, contraction kernel, broad mode-order rescue, Tensor Ring, Hierarchical Tucker, or adjacent decomposition is authorized.

## EXP-067 — Pinned Real-Q4 Joint Multi-Projection Exact-Reuse Gate

### Execution-class change

EXP-067 does not seek another representation of one matrix. It examines operations that the Transformer already evaluates together from the same input activation:

```text
attention group: q_proj, k_proj, v_proj
MLP group:       gate_proj, up_proj
```

The candidate mechanism is exact common arithmetic across operators:

- one input linear form reused by multiple output rows;
- identical, negated, or integer-proportional Q4 rows reused with a cheap scale/sign correction;
- exact common right-factor width shared by an entire projection group;
- repeated row blocks reused without changing the model or quantization.

This is an executor/compiler question, not target retraining or approximate compression.

### Prior and cheap falsification

The prior is low:

- EXP-058 proved individual matrices full rank;
- EXP-065/066 found no useful exact single-matrix classical tensor structure;
- independently learned projections are expected to contain few exact repeated linear forms.

The cheapest decisive test is therefore to measure exact reusable arithmetic directly on the pinned Q4 weights before implementing any joint kernel.

### Population

Use unchanged revisions and the frozen deterministic row-wise Q4 quantizer:

```text
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Every analyzed tensor checksum must match the frozen EXP-057/058/065 Q4 evidence. Required groups are every complete attention `Q/K/V` group and every complete MLP `Gate/Up` group present in the models.

### Exact canonical row classes

For every Q4 integer row `w`, construct fail-closed canonical identities:

1. exact equality: `w_a = w_b`;
2. sign equality: `w_a = -w_b`;
3. primitive integer proportionality:
   - divide by the row gcd;
   - normalize the first nonzero sign;
   - retain the exact integer multiplier;
4. exact repeated contiguous row blocks at preregistered block sizes.

Zero rows are reported separately and may not be silently folded into proportional classes.

One canonical dot product may be charged once, followed by one cheap sign/scale operation per dependent row. Hash matches must be byte-verified before accounting.

### Common-right-factor lower bound

For a projection group with matrices `W_i`, any exact shared input transform

```text
z = Bx
W_i x = A_i z
```

requires

```text
rank(B) >= rank(vertical_stack(W_i)).
```

Certify the stacked Q4 rank using existing modular witnesses. This closes exact shared low-width input transforms when the stacked rank equals the input width.

### Favorable accounting

Report both structural coverage and best-case operation/storage fractions:

```text
baseline dot products = total output rows
unique canonical dot products = canonical row-class count
reuse operation fraction =
  (unique full dot products + dependent sign/scale corrections)
  / baseline full dot products
```

Also charge canonical maps, multipliers, group metadata, and query traffic. Compilation time is reported separately. Dense GEMV kernel efficiencies are not credited without a physical kernel.

### Controls

- synthetic duplicate/sign/proportional groups achieve the registered reuse exactly;
- a one-nibble mutation breaks the corresponding class;
- random dense groups show negligible exact reuse;
- hash collisions are byte-verified and fail closed;
- stacked-rank witnesses validate under at least two primes;
- Q4 checksums match frozen evidence;
- no approximation, activation oracle, training, or changed quantization.

### Promotion Gate

```text
zero checksum/certificate/control mismatch
100% complete registered projection-group coverage
p50 exact joint operation fraction <=10%
p90 exact joint operation fraction <=25%
p50 exact joint storage fraction <=10%
p90 exact joint storage fraction <=25%
random-control reusable-row fraction <=1%
no largest-model degradation >25%
```

Passing only a few layers or one operator family is insufficient. Promotion authorizes an exact reconstruction/replay test, not a physical kernel claim.

### Failure decision

```text
REJECT_REAL_Q4_EXACT_JOINT_ROW_REUSE_AS_CORE_RETAIN_GROUP_CERTIFIER_AUXILIARY
```

On failure, exact equality/sign/proportional common-subexpression reuse and exact shared low-width input factors are closed as the primary core for the measured population. The next candidate must move to certificate-guided demand-driven execution or another new information source.

### Stop rule

Before the Gate passes, prohibit:

```text
joint CUDA kernels
model-wide integration
approximate row clustering
learned cross-projection adapters
unbounded transform searches
arbitrary linear-circuit synthesis
```

### Claim boundary

Phase A/B/C weight observation, evidence ceiling E1. 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, tokens/second, real joint-kernel speedup, and end-to-end model-output preservation remain NOT TESTED.
