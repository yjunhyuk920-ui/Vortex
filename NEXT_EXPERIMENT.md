# Next Experiment

## Research-efficiency classification

EXP-066 is authorized as a **bounded cheap-kill certificate Gate**, not as an open-ended Tensor-Train implementation program.

Reason for allowing it:

- it was preregistered before the efficiency-policy change;
- it strictly generalizes the one-cut Kronecker rank test;
- exact unfolding-rank certificates can reject it before factor reconstruction, runtime integration, or kernel work;
- the certifier can reuse the frozen real-Q4 population and modular-witness infrastructure.

The Gate must terminate at lower-bound certification and fully favorable accounting unless the promotion thresholds pass.

Before survival, the following are prohibited:

```text
exact MPO core reconstruction
physical MPO kernels
model-wide MPO runtime integration
broad rescue searches over unregistered mode orders or factor schedules
adjacent classical single-matrix tensor decompositions opened only because TT/MPO failed
```

## Closed Gate — EXP-065

All 144 dense projections selected a full-rank four-row Kronecker rearrangement. Favorable lower-bound operations exceeded dense execution by more than 2x and static storage did not shrink.

```text
REJECT_REAL_Q4_KRONECKER_RANK_AS_CORE_RETAIN_TENSOR_CERTIFIER_AUXILIARY
```

## EXP-066 — Pinned Real-Q4 Exact Tensor-Train / MPO Bond-Rank Gate

### E0 efficiency triage

Optimistic upside if exact bond ranks are genuinely small:

- static core storage and per-query weight reads could, in principle, fall by an order of magnitude or more;
- chained contractions could replace the dense matrix-vector product;
- the mechanism is automatically derived from unchanged weights.

Reasons for low prior probability:

- EXP-058 proved ordinary exact low rank absent;
- EXP-065 found full-rank Kronecker rearrangements with more than 2x favorable operation cost and no storage reduction;
- the same real-Q4 population has repeatedly behaved like general dense matrices under exact classical structure tests.

Cheapest decisive falsification:

> certify every necessary prefix/suffix unfolding rank and derive favorable lower bounds on exact MPO storage and contraction work before reconstructing any core.

Authorized implementation stage:

> rank/certificate generation, witness verification, controls, accounting, frozen evidence, and repository validation only.

### Mechanism

Factor matrix dimensions into ordered radix sequences:

```text
m = product_k m_k
n = product_k n_k
```

Pad the shorter sequence only with unit modes, pair `(m_k,n_k)`, and reshape the Q4 matrix into an interleaved Matrix-Product-Operator tensor with physical mode sizes `d_k=m_k*n_k`. For every cut `k`, certify the exact rank of the prefix/suffix unfolding:

```text
R_k = rank(unfold(W, product_{i<=k} d_i, product_{i>k} d_i))
```

These are necessary TT/MPO bond ranks. With `R_0=R_L=1`, exact core storage is lower-bounded by:

```text
sum_k R_{k-1} * m_k * n_k * R_k
```

All admissible radix schedules and deterministic mode-order variants defined before execution must be evaluated. Every selected cut receives independently verified witnesses under at least two primes.

### Population

Use the unchanged TinyStories-1M/3M/8M revisions and frozen EXP-057 Q4 checksums. Analyze all 153 two-dimensional tensors and report promotion statistics over all 144 dense projections.

### Accounting

Charge the bond-rank storage lower bound, mode-order metadata, per-row scales and biases, input reads, MPO contractions, every intermediate tensor read/write, output reductions, compilation and certificate work. Use favorable 4-bit core storage so rejection remains conservative.

### Controls

- exact rank-1 and low-bond MPO tensors certify correctly;
- a one-nibble mutation raises at least one bond rank;
- dense-random and forced-unique tensors produce high bond ranks;
- interleaved reshape/order round trips are exact;
- every selected bond witness verifies under two primes;
- no approximation, training, activation table or changed quantization.

### Promotion Gate

```text
zero checksum/certificate/control mismatch
all 144 dense projections covered
p50 lower-bound operation fraction <=10%
p90 lower-bound operation fraction <=25%
p50 lower-bound storage fraction <=10%
p90 lower-bound storage fraction <=25%
dense-random adversary p50 <=25%
projected static storage <=1 TiB
no largest-model degradation >25%
exact integer MPO reconstruction before operation-replacement promotion
```

Passing rank or storage alone is insufficient. Promotion requires operations and storage to survive together at population level and a credible route to actual Transformer operation replacement.

Failure decision:

```text
REJECT_REAL_Q4_TT_MPO_BOND_RANK_AS_CORE_RETAIN_MPO_CERTIFIER_AUXILIARY
```

### Mandatory stop rule after failure

If the operation or storage lower-bound Gate fails:

1. retain only the generic MPO unfolding/rank certifier as auxiliary infrastructure;
2. do not implement exact cores, contractions, kernels, or runtime integration;
3. close exact classical single-matrix tensor factorization as a primary core direction for the measured real-Q4 population;
4. prohibit rescue through mode-order expansion, rank tuning, Tensor Ring, Hierarchical Tucker, Butterfly-like relabeling, or another adjacent decomposition unless a new measured fact or asymptotic mechanism explicitly reopens the family;
5. select the next core candidate through `docs/RESEARCH_EFFICIENCY_CONTRACT.md` rather than by decomposition adjacency.

### Post-EXP-066 pivot rule

If EXP-066 fails, the next primary Gate must change execution class and show a credible order-of-magnitude upside before implementation.

Priority classes for E0 triage are:

- **joint multi-projection common-arithmetic compilation** — analyze Q/K/V together and Gate/Up together, seeking reusable exact arithmetic across operators rather than another isolated matrix representation;
- **certificate-guided demand-driven or lazy execution** — determine whether only a small dependency subgraph is needed to settle the final decision, with exact fail-closed expansion when the certificate is insufficient;
- **proposal plus substantially cheaper exact verification** — only if a theorem or executable bound shows verification avoids nearly all target work;
- another genuinely new information source or execution representation with a route toward the approximately 1.185% final target fraction.

These are not presumed solutions. Each must pass target-upside, novelty, scaling, full-cost, universality, correctness, and cheap-falsification triage before receiving an experiment number.

### Claim boundary

Phase C weight observation and exact unfolding-rank certification only. Exact MPO cores, Q4 model-output preservation, a physical MPO kernel, real Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT and tokens/sec remain NOT TESTED.
