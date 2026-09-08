# Direct global finite-word producer frontier — 2026-09-09

## Decision

```text
REJECT_P1_RECONSTRUCTIVE_GLOBAL_CODE_AS_CORE
REDUCE_P2_EXACT_SUM_PLUS_ROUNDING_WITNESS_TO_THE_DIRECT_MATVEC_PROBLEM
ESTABLISH_GLOBAL_NONLINEAR_SIMULTANEOUS_FULL_MV_ROUTE_SPAN_GATE
KEEP_THAT_GLOBAL_GATE_FAR_TOO_WEAK_FOR_TARGET_IMPOSSIBILITY
CONSTRUCT_EXACT_ARBITRARY_GF2_GAUSS_JORDAN_PRODUCER
REJECT_THAT_EXPLICIT_PRODUCER_AT_THE_90_PERCENT_OPERATION_AND_METADATA_GATE
KEEP_DIRECT_GLOBAL_NATIVE_FINITE_WORD_PRODUCER_OPEN
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
```

The fixed VORTEX mission is unchanged.  This round starts from remote scientific
head `6021ff11ca313f304511ceecaa38ac75af22bf8c`, freezes three new direct-producer
principles in `PREREGISTRATION.md`, commits that preregistration separately at
`a2d269c0aeee089be9859337813dfa195fa5b862`, and then executes all three.

The round does **not** produce the arbitrary-native 405B executor.  Its main
constructive advance is narrower but real: it obtains an explicit finite
producer for every binary matrix, then identifies exactly why this producer
misses the required cost, while also removing two broader escape assumptions:
global reconstructive coding and exact-sum/rounding separation.

Canonical E0 result:

```text
results/e0_direct_global_producer_gate_v5/summary.json
SHA-256 a907cb1dbd40b9a7bf71b88a159cf0358c4d3b0656470e71e26f37e2c9185ffe
```

The earlier v1--v4 outputs are retained as strengthening/debug history rather
than overwritten.

## 1. Frozen target and accounting population

The E0 binary reduction uses all 883 registered non-embedding dense matrices of
the 405B shape plan:

```text
binary source coefficients               403,747,897,344
global hot-advice grant                       68,719,476,736 bits = 8 GiB
registered dense matrices                               883
sum output rows                              19,997,952
sum input/right dimensions                   19,111,936
64-bit binary source+advice cells          7,382,302,720
```

The binary reduction is a favorable necessary-condition model for a universal
arbitrary-checkpoint producer.  Passing it would not establish Q4/BF16/FP32
native arithmetic.  Failing a route that explicitly reconstructs all source
bits or executes a declared binary producer is decisive only for that route.

The historical `8/675` traffic fraction is used only as a familiar E0 logical
comparison.  It is not treated as the final p50/p95 latency theorem.

## 2. P1 — globally coded native reconstruction

### 2.1 Explicit finite candidate

P1 does not divide the 8 GiB advice among matrices.  Concatenate every source
word globally and regard the source as symbols over a fixed `GF(2^64)` defined
by one public irreducible polynomial.  A fully explicit systematic MDS-style
erasure code can be instantiated with a deterministic Cauchy parity matrix;
the registered source-plus-parity symbol count is far below `2^64`, so distinct
field points can be assigned finitely:

```text
Compile(X):
    split canonical source X into 64-bit symbols x_0,...,x_(k-1)
    store X unchanged in paid cold storage
    choose fixed distinct field points a_i and b_l
    for each parity l:
        parity_l = XOR_i x_i / (a_i - b_l)
    store at most 8 GiB of parity symbols

Query/reconstruct:
    read a fixed information set of raw/parity symbols
    solve the finite Cauchy erasure subsystem for missing source symbols
    recover every required original source symbol exactly
    execute the original native dense kernel/reduction order
```

This is deliberately overpowered and expensive, but all subroutines are finite.
Every square Cauchy submatrix is nonsingular, so any erasure set up to the
parity count is recoverable. Naive construction has a valid upper bound
`O(k*r)` field operations; naive erasure solving has a finite cubic upper bound
in the erased-symbol count. The global generator is procedural, so no free
per-matrix advice split appears.

### 2.2 Exactness, state and termination

For every source bit string `X`, exact erasure decoding gives

```text
Decode(Encode(X), information_set) = X.
```

The downstream executor then receives the identical checkpoint words and runs
the identical reference reduction order.  Therefore, if state/RNG are equal at
step `t`, logits and required state at step `t+1` are equal and the same sampler
consumes the same RNG words.  Equal initialization gives the usual induction.

All loops range over finite source/parity dimensions, so the constructor,
decoder and reference step terminate.

### 2.3 Paid bound and decisive rejection

Even granting **all** 8 GiB as information-bearing parity, exact reconstruction
of `N` arbitrary bits from an `R`-bit advice string needs at least `N-R`
additional source bits for some source.  This is simple injectivity: otherwise
`(advice, read bits)` has fewer than `2^N` descriptions.

Registered best-case source-read removal:

| Source alphabet | Source size | Maximum removed by 8 GiB | Minimum raw read |
|---|---:|---:|---:|
| binary | 47.0024 GiB | 17.020392% | 82.979608% |
| Q4 | 188.0098 GiB | 4.255098% | 95.744902% |
| BF16 | 752.0391 GiB | 1.063775% | 98.936225% |

P1 also retains **100% of the native dense arithmetic** after reconstruction.
Parity reads, interpolation, RAM/SSD traffic, PCIe/HBM movement, workspaces and
metadata are positive extra terms.  Hence P1 fails both the >=90% source-effect
route and the arithmetic route before target hardware is relevant.

The strongest adversary is simply an incompressible arbitrary checkpoint.  No
codec implementation or checkpoint distribution can repair this universal
reconstruct-all interface.

## 3. P2 — exact dyadic sum plus native-order rounding witness

P2 asks whether the expensive part of native exactness can be factored into:

```text
cheap exact mathematical sum + small rounding-order witness.
```

### 3.1 Literal finite candidate

A literal version is fully constructive for every finite-word input:

```text
for each original leaf product:
    decode its exact finite value / exceptional class
    append its exact dyadic contribution to an arbitrary-precision accumulator
in parallel:
    execute the declared reference finite-word reduction tree
return (exact_sum, reference_rounded_word)
```

Exceptional NaN/Inf/signed-zero cases branch to their finite ABI rules.  The
constructor stores the original words plus the fixed reduction-tree metadata.
Runtime uses finite loops and finite-precision/reference operations; it
terminates.  Because it literally returns the reference rounded word, exposed
state and RNG induction are immediate.

But this literal candidate performs every product and the entire original
reduction, so it is already above 100% dense work before the exact accumulator.
The only credible P2 route was therefore to replace the reference-order witness
with a much smaller statistic.

### 3.2 Zero-exact-sum rounding gadget

That route is not easier than direct MatVec.  For one binary bit `b`, use eight
balanced-tree leaves

```text
[ 2^24, b, -2^24, 0, -b, 0, 0, 0 ].
```

Their exact integer/dyadic sum is always zero.  Under explicit IEEE FP32
round-to-nearest-even pairwise balanced addition:

```text
fl(2^24 + b) = 2^24
fl(-2^24 + 0) = -2^24
fl(-b + 0) = -b
```

so the upper tree returns exactly `-b`.

For arbitrary binary row `a` and query `v`, concatenate one aligned gadget per
coordinate with `b_j=a_j & v_j`, padding the number of gadgets to a power of two.
The upper tree sums the small integer gadget outputs exactly, yielding

```text
exact mathematical sum = 0
balanced FP32 result    = -popcount(a & v).
```

Thus its parity contains `a·v mod 2` although the exact sum contains **zero
information** about that bit.

The implementation exhaustively checked every row/query pair through width 5;
all exact sums were zero and every rounded count/parity matched.

### 3.3 Consequence

A universal `RoundWitness` that repairs exact mathematical sums to this declared
reference ABI can be forced to compute arbitrary binary matrix-vector/count
information.  Therefore P2 does not create a simpler independent producer.  It
reduces the missing work back to P3.

This theorem is explicitly scoped to the registered balanced IEEE-FP32 RNE
tree.  It is **not** a claim about every CUDA GEMM reduction schedule.  No target
native ABI was executed here.

The candidate's paid storage/read/arithmetic upper bound is at least the original
checkpoint plus one full leaf-product/reduction traversal; exact-sum state and
witness state are additional.  It therefore fails the >=90% route as a complete
candidate.

## 4. P3 — query-direct globally nonlinear word router

P3 is the highest-upside class because it does not reconstruct the checkpoint
and does not assume associativity.  Stored 64-bit cells may be arbitrary
checkpoint functions, mix all matrices, and every later address may depend on
the current query and previous returned values.

### 4.1 New global vector-route span theorem

Let the binary source be the direct sum of matrix blocks

```text
X = direct_sum_i F2^(m_i*n_i).
```

A simultaneous query supplies one nonzero `v_i in F2^(n_i)` for every block and
returns every full vector `M_i v_i`.  Consider an exact deterministic router
with `S` cells of `w` bits and at most `t` probes.

Apply the prior largest-fiber recursion to the **joint output vector**.  After
fixing the first `t-1` returned words, a final route retains at least

```text
2^(N-(t-1)w)
```

sources.  Every scalar answer for every query tuple in that route factors
through one final `w`-bit word.  If `L` is the span of all those scalar
coefficient masks, the same projection-fiber argument gives

```text
dim L <= t*w.                                      (1)
```

For block `i`, let `U_i` be the span of all right factors appearing in the route
and `k_i=dim U_i`.  The full-Mv output contains coefficient masks

```text
e_row tensor u,   u in U_i,
```

whose block-local span has dimension exactly `m_i*k_i`.  Different blocks are
disjoint, so (1) becomes

```text
sum_i m_i*k_i <= t*w.                              (2)
```

This is the key global advance: **no advice is divided per matrix**.  An
arbitrary globally mixed nonlinear cell is charged once through the common
route span.

One route contains at most

```text
product_i (2^k_i - 1) < 2^(sum_i k_i)
```

simultaneous query tuples.  There are at most `S^t` routes.  With every query
requiring `k_i>=1`, maximize `sum k_i` subject to (2) by greedily allocating
extra dimensions to the smallest `m_i`; every dimension has unit benefit and
cost `m_i`, so this relaxation is exact.

### 4.2 Registered global result

Using the complete binary source plus all 8 GiB as a favorable pool of 64-bit
cells:

```text
S                                           7,382,302,720 cells
ceil(log2 S)                                           33 bits
matrix blocks                                           883
sum output rows                                  19,997,952
sum right dimensions                              19,111,936
nonzero tuple count                  > 2^19,111,935
```

Even one simultaneous full-Mv query forces

```text
t*64 >= 19,997,952
t >= 312,468 words.
```

The proof-safe route-cover calculation gives:

```text
t = 578,618: favorable coverage exponent <= 19,111,911  REJECTED
t = 578,619: favorable coverage exponent <= 19,111,944  count-feasible only
```

So every such global router needs at least `578,619` word probes in the strong
simultaneously-independent service model.  This is `37,031,616` bits = about
`4.4145 MiB`.

But the favorable registered target line permits `299,072,516` 64-bit words.
The lower bound is only

```text
0.193471% of that allowance
```

or about `516.87x` too weak to reject it.  **This is not a constructor and not
an impossibility result.**  Complete simultaneous tuple causal reachability and
native arithmetic are also not proved.

This canonical substitution also does **not** ban an expanded checkpoint-
dependent cold sidecar larger than `source + 8 GiB`.  The theorem itself accepts
an arbitrary finite `S`; a larger sidecar increases `S` and weakens the route-
count consequence while its storage/build/read costs must be charged elsewhere.
Therefore the `578,619` number is not a lower bound for unlimited expanded
storage and is never used as one.

### 4.3 Actual arbitrary-GF(2) producer constructed

The round did not stop at the lower bound.  A finite direct producer for every
binary matrix was implemented:

```text
Compile(W):
    run deterministic Gauss-Jordan row elimination
    record every elementary row XOR
    store reduced matrix R = E W and the operation stream

Query(v):
    compute z = R v exactly over GF(2)
    replay the recorded elementary row XORs in reverse on z
    return y = E^(-1) z = W v
```

The representation is the reduced matrix plus the complete row-op stream.  The
address rule is sequential over the reduced rows and then the reversed program;
there is no selector oracle.  Each XOR is self-inverse, so exactness follows
algebraically.  The compiler examines a finite `rows*columns` pivot grid and the
runtime visits a finite program, so both terminate.

Random rectangular and square matrices in the focused test matched direct GF(2)
reference outputs for every registered small query.

### 4.4 Why the explicit producer fails

For the fixed unit-lower-triangular all-one family, the declared pivot order
performs exactly

```text
n(n-1)/2
```

row XORs and reduces to identity.  Runtime therefore costs

```text
n + n(n-1)/2
---------------- = 1/2 + 1/(2n)
       n^2
```

of the direct scalar bit-operation count.

At `n=16,384`:

```text
runtime operations          134,225,920
dense bit products          268,435,456
fraction                    16,385/32,768 = 50.0030518%
row-op metadata lower       3,757,867,008 bits = 447.97 MiB
```

The metadata lower is already about 14x the raw 32-MiB binary matrix, before
the reduced matrix itself, alignment and interpreter state.  This exact
producer therefore fails the >=90% operation-removal Gate and has adverse
program traffic/storage.

This adversary rejects **this Gauss-Jordan producer only**.  The same structured
matrix has other cheap algorithms, so the result is not misreported as a
general linear-circuit lower bound.  More sophisticated static linear circuits
remain governed by the existing F-050 boundary rather than being reopened here.

## 5. Candidate comparison after execution

| Principle | Arbitrary source constructor | Exact query/state path | >=90% route | Result |
|---|---|---|---|---|
| P1 global reconstruction | yes, finite systematic erasure code | reconstruct exact checkpoint then reference step | impossible for declared interface: Q4 read removal <=4.255%, arithmetic 0% removed | REJECT |
| P2 exact sum + witness | literal finite producer exists; optimized witness was the hoped source | exact sum + reference-order witness | literal path >100%; gadget makes optimized witness contain arbitrary Mv | REDUCE TO P3 / REJECT AS DISTINCT CORE |
| P3 direct global router | general nonlinear class specified; exact arbitrary-GF2 elimination producer implemented | binary exact; native lift still missing | elimination only ~50% removal; general route theorem does not reject target | GENERAL CLASS OPEN, CONCRETE PRODUCER REJECTED |

P3 remains the strongest research class because it alone can in principle avoid
both reconstructing all weights and replaying every native contribution.  No
explicit globally nonlinear P3 encoder/address/decoder meeting the target was
found in this round.

## 5.1 Complete paid-cost ledger for the executed candidates

| Cost term | P1 global reconstruction | P2 exact sum + witness | P3 Gauss-Jordan control |
|---|---|---|---|
| Constructor | Cauchy parity: naive `O(k*r)` GF(2^64) ops | finite ABI decode/tree metadata; literal path needs no magical preprocessing | Gauss-Jordan: at most finite pivot grid and row XORs; each compile row XOR touches the stored row |
| Persistent source | original checkpoint retained | original checkpoint retained | reduced binary matrix `R` retained |
| Auxiliary storage | <=8 GiB parity in the favorable screen; full use leaves no physical GPU headroom | exact-sum/reduction metadata plus any witness representation | row-op program; 447.97 MiB lower for one 16,384 adversary matrix |
| CPU RAM / SSD | raw checkpoint + parity/interpolation buffers; if parity is cold its reads count | raw checkpoint + exact accumulator/witness buffers | reduced matrices + row program; cold program fetch counts |
| PCIe / HBM | at least unreplaced source information plus native kernel stream; parity movement extra | all original leaf data plus witness/exact-sum movement | every consumed reduced/program word; no free instruction fetch |
| GPU allocation/workspace | 8-GiB advice grant is an E0 oracle, not a feasible total allocation; KV/workspace still positive | exact accumulator + native reduction workspace + KV/state | output/reduced-vector/program working state plus KV/native lift if attempted |
| Arithmetic | 100% original dense arithmetic + decode | 100% leaf products/reference tree + exact-sum overhead in literal implementation | `Rv` plus reverse row XORs; 50.0030518% scalar-bit fraction on frozen n=16,384 adversary |
| Address/metadata | field-point rule small, but erasure locations/decoder work paid | reference-tree and exceptional-value control paid | two row indices per op; metadata explicitly charged |
| Verification/repair | none needed for exact reconstruction; any checksum is extra | literal reference witness is exact but is the work being removed; no free repair | algebraically exact GF(2); native lift/verification absent rather than free |
| KV/RNG/state | identical only after full native reference step | identical only for the literal full-reference path; optimized witness has no whole-model proof | not supplied: binary effect only, therefore O2/O3 remain OPEN |
| >=90% route | FAIL: Q4 read removal <=4.255%, arithmetic removal 0% | FAIL: literal work >=100%; cheap witness reduced back to MatVec | FAIL for implemented producer: ~50% scalar work plus adverse metadata |

Every omitted real hardware cost is nonnegative.  No overlap or peak-bandwidth
assumption is used to convert these failed logical candidates into a pass.

## 6. Full O1--O6 accounting

### O1 — OPEN

The Gauss-Jordan compiler is uniform for arbitrary GF(2) matrices, but there is
no uniform producer for arbitrary native checkpoint words with target cost.
P1 is arbitrary-native but reconstructive and fails cost; P2 is arbitrary-native
only in its literal full-work form.

### O2 — OPEN

No full arbitrary-checkpoint load/prefill/decode causal executor uses the new
producer.  The prior causal bridge remains a reachability/input source, not the
missing whole-body executor.

### O3 — OPEN

P1 would inherit native state/RNG equality only after full reconstruction and
reference execution.  P2's gadget is scoped to a declared balanced FP32 tree.
P3 exactness is GF(2) only.  No arbitrary Q4/BF16/FP32 finite-word logit/KV/RNG
induction has been established.

### O4 — OPEN

The rejected candidates have explicit finite cost bounds, but no surviving
whole-model sufficient upper bound exists.  Required cost terms remain:

```text
constructor + transformed/original storage
RAM + SSD + PCIe + HBM
GPU peak allocation/workspace
addresses + metadata + program fetch
arithmetic/reductions + native lift
KV/state + verification/repair/fallback
initialization + short-session amortization
```

### O5 — OPEN

No actual 405B, CUDA, <=8 GiB GPU, PCIe/SSD/HBM schedule, same-machine native
4B Q4 p50/p95 or TTFT run occurred.

### O6 — PARTIAL

Preregistration, executable gate, exact result JSON/checksum, tests and this
report are independently inspectable.  Whole-theory reproducibility remains
open because O1--O5 are open.

## 7. Literature check

The current literature check did not yield the missing arbitrary-dense zero-error
upper construction:

- Anand, van den Brand, McCarty, *The Structural Complexity of Matrix-Vector
  Multiplication* (2025), `https://arxiv.org/abs/2502.21240`, obtains
  subquadratic preprocessed queries for bounded-VC/near-structured matrices; it
  does not give an arbitrary-dense producer.
- Ko, *Lower Bounds for Linear Operators* (2025),
  `https://arxiv.org/abs/2509.02730`, strengthens lower bounds even against
  nonlinear preprocessing for random operators; it is not an upper
  construction and its quantitative statement is not promoted into a VORTEX
  impossibility theorem.
- Hirahara--Shimizu's 2025 OMV error-correction reduction is already audited by
  `E0_AVERAGE_ORACLE_AMPLIFIER_FRONTIER.md`: it still needs a charged fast
  approximate oracle.  Direct exact-row lottery restores a rank-covering source
  scan and is not reopened here.
- M4RI/Four-Russians packed GF(2) computation reduces machine operations but its
  source/table traffic remains the previously closed full-scan/table family.

Absence of a construction in this search is not an impossibility theorem.

## 8. Validation and immutable result history

Focused new suite at the final source state:

```text
10/10 PASS
```

It includes arbitrary rectangular/square GF(2) producer equality, the complete
rounding gadget control through width 5, P1 exact accounting, P3 registered
global route calculations, and a tiny exhaustive check that the route-capacity
greedy allocation equals the true optimum.

Canonical result:

```text
results/e0_direct_global_producer_gate_v5/summary.json
SHA-256 a907cb1dbd40b9a7bf71b88a159cf0358c4d3b0656470e71e26f37e2c9185ffe
```

Earlier strengthening artifacts are retained:

```text
v1 7aaf68b8d68e06c04511ecbf1693b7fec0e37a547186e709307d665a047b8146
v2 06d630e92d69fdc3c842f8050886644be6a77762362c3757f61a516ad8b6f03f
v3 e9882c0a6d86de66e795fc52fcee1c1ab1e9d423a21c2d24e8eb8a828bcca06d
v4 9febeb38dd1161bb3b9028c23817705300ffa129c8e130f062848886ee75ad95
```

The completed 15/15, 28/28, 14/14 and Boolean-lift exhaustive suites were **not
rerun**, per the continuation instruction.  No old native-global 3,510-file
replay was rerun.

## 9. Next exact construction

The remaining object is now even narrower:

```text
Compile(all arbitrary native checkpoint words) -> global representation G
Address(G, current causal state/right factors) -> subdense bounded reads
Decode(read words) -> exact native ordered dense effects
```

It may not reconstruct most original weights (P1), outsource rounding to an
easier exact-sum witness (P2), or merely store/replay a quadratic linear program
(the explicit P3 control).  A credible next mechanism must make the *direct
query* nonlinear and implicit while providing an actual finite cell compiler and
decoder, not just passing the weak global route lower bound.
