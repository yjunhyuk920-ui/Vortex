# E0 Finite-Semiring Preprocessing Frontier

## Verdict

```text
REJECT_WILLIAMS_FINITE_SEMIRING_GRAPH_AS_REFERENCE_EXACT_2_5_PERCENT_CORE
KEEP_THE_PUBLISHED_FINITE_SEMIRING_ALGORITHM_VALID_IN_ITS_MODEL
KEEP_GENERAL_NONLINEAR_NUMERICAL_RANK_ONE_GAP_OPEN
KEEP_UNIVERSAL_2.5% NOT ESTABLISHED
KEEP_NO_SURVIVING_CANDIDATE
```

This audit tests one of the strongest favorable interpretations of Ryan
Williams' preprocessed finite-semiring matrix-vector algorithm. The algorithm
is real and globally checkpoint-dependent. It does not fit the registered
VORTEX contract once the lookup graph's physical answer payload, persistent
catalog, native rounding semantics, and 32-token boundary are separated.

This is a rejection of this constructor, not a proof that every possible
globally nonlinear bounded-word data structure fails.

## Elementary explanation

Imagine that a question contains a short strip of `b` symbols. The algorithm
prepares a card for every strip that might ever be asked.

```text
possible symbols per position = K
positions on one strip        = b
number of cards               = K^b
```

When a real question arrives, the algorithm does not reread every matrix
entry. It chooses the already prepared card matching each strip of the input
and follows that card's links. Larger strips mean fewer links must be read.

The catch is that larger strips create exponentially more cards:

```text
read saving grows like b
card catalog grows like K^b
```

For a 16,384-wide layer, the published theorem's parameterization grants at
most `b=14`. A semantic 40-fold read reduction would need `b>=40`. One may
physically build cards with `b>14`, but that is outside the advertised theorem
parameter and makes the catalog still larger. It is not a free extension.

There is a second problem. The paper assumes an associative finite semiring,
where regrouping additions never changes the answer. Native BF16 and FP32
rounded additions can change their answer when regrouped. Therefore the
paper's correctness proof cannot simply be relabeled as bit-exact Transformer
arithmetic.

## 1. Published theorem and the exact graph shape

For a finite semiring with `K` elements, Williams' Theorem 1.2 preprocesses an
`n x n` matrix and reports

```text
preprocessing: O(n^(2 + epsilon*log2(K)))
query:         O(n^2 / (epsilon*log(n))^2) steps
machine:       pointer machine or log-n word RAM
block:         b = ceil(epsilon*log2(n)), 0 < epsilon < 1
```

The construction splits the input and output into

```text
g = ceil(n/b)
```

groups. Every input group contains a first-layer node for each of its `K^b`
possible input patterns. Every first-layer node stores one output-pattern
neighbor for every output group. A query chooses one first-layer node per
input group and reads all of those neighbor lists.

The paper counts adjacency-list operations. VORTEX must additionally count
the physical information returned by each neighbor lookup.

## 2. Favorable physical payload equation

Give the constructor every favorable representation choice:

- the output group is known from list position;
- a neighbor stores only its `b`-symbol output pattern;
- every symbol uses exactly `log2(K)` bits;
- object headers, global addresses, offsets, counters, cache lines, pages,
  query inputs, outputs, and semiring operations cost zero.

One selected neighbor still has to identify one of `K^b` patterns, so it
needs at least

```text
b*log2(K) bits.
```

The direct graph therefore has

```text
query payload
  = g^2 * b*log2(K)
  approximately n^2*log2(K)/b bits,

persistent edge payload
  = g^2 * K^b * b*log2(K)
  approximately n^2*log2(K)*K^b/b bits.
```

Relative to the semantic raw matrix payload, the ideal continuous ratios are

```text
query fraction   = 1/b
sidecar fraction = K^b/b.
```

This is the central tradeoff. The logarithmic query saving is bought with an
exponential checkpoint-dependent catalog.

These equations are deliberately lower than a real implementation. They are
a rejection screen, not a deployable upper bound and not a general lower
bound for unrelated data structures. A new cross-edge compressor, shared-list
encoding, or implicit catalog would need its own exact decoder, worst-case
representation, operation, and probe proof; this audit does not reject such a
different construction merely by charging the published explicit graph.

## 3. Finite registered substitution

The largest registered square has `n=16,384`, so `log2(n)=14`. Grant the
largest `b=14` allowed by the theorem's displayed `epsilon<1`
parameterization, even where the omitted finite-`n` work would make this
choice less favorable.

For one `16,384 x 16,384` Boolean matrix, the explicit ceiling-aware graph
already has

```text
groups                         1,171
patterns per group             16,384
selected neighbor values       1,371,241
minimum query payload          19,197,374 bits
minimum query/raw ratio        7.151579%
minimum persistent edge bits   314,529,775,616 bits
minimum persistent edge bytes  39,316,221,952 bytes
minimum persistent edge size   36.616085 GiB
```

Thus even this single one-bit square has a favorable edge payload more than
four times the complete global 8 GiB advice grant. This omits the original
matrix and every practical graph cost.

### 3.1 Model-wide continuous lower-payload screen

Apply the favorable `1/b` and `K^b/b` ratios continuously to all
`405,849,243,648` registered parameters and compare query bytes with the
reported `551.22 GB` DFloat checkpoint denominator.

| Symbol interpretation | One-query payload / DFloat | Minimum `b` for DFloat `1/40` | Ideal sidecar | Result |
|---|---:|---:|---:|---|
| one bit | `0.657388%` | `4` | `55,292.57 GiB` | query fits; sidecar is `6,911.57x` 8 GiB |
| Q4, 4-bit symbol | `2.629552%` | `15` | `9.727e17 GiB` | query and sidecar fail at `b=14` |
| BF16, 16-bit symbol | `10.518207%` | `59` | `1.456e69 GiB` | query and sidecar fail |

The `b>=40` statement applies to a query charged against its own semantic raw
matrix payload. The DFloat denominator changes the required block size by
alphabet: `4`, `15`, and `59` for one-bit, Q4, and BF16 symbols respectively.
Keeping these denominators separate prevents a false universal claim.

The one-bit row is not a Transformer solution. It computes only one-bit
semiring arithmetic. Bit-slicing a native numerical computation needs all
relevant planes plus exact carries, products, rounding, and accumulation
order; naming the one-bit row does not supply those operations.

## 4. Native rounded arithmetic is not the theorem's semiring

A semiring requires addition to be associative. Native rounded addition is
not associative.

For BF16 round-to-nearest-ties-to-even, let

```text
a = 1
b = c = 2^-8.
```

Then

```text
(a plus_bf16 b) plus_bf16 c = 1
a plus_bf16 (b plus_bf16 c) = 1 + 2^-7.
```

For FP32 accumulation, let

```text
a = 2^24
b = 1
c = -2^24.
```

Then

```text
(a plus_fp32 b) plus_fp32 c = 0
a plus_fp32 (b plus_fp32 c) = 1.
```

Williams' graph groups and combines partial contributions using semiring
associativity and distributivity. The two exact counterexamples show that its
proof does not preserve either naive BF16-per-add or FP32-per-add reference
semantics under regrouping.

A future constructor could encode a fixed reference accumulation order or a
larger finite-state transition monoid. That would be a different constructor
whose transition representation, multiplication, query probes, and state must
all be charged. This audit does not rule it out by name.

## 5. Full MatVec is not the remaining scalar interface

The published constructor returns the complete vector `W*u`. The open VORTEX
interface is the scalar decision quantity `r^T W u`. Running the published
algorithm and then taking the final dot product over-computes the required
answer. This audit grants the complete output, its movement, and the final dot
product for free; Q4/BF16 and the persistent catalog still fail.

Conversely, the full-vector theorem cannot be quoted as a direct scalar
rank-one data structure. A new scalar-specific construction could have a
different tradeoff and remains inside the open ticket.

## 6. The 32-token boundary

The registered gate grants `1/40` of the DFloat checkpoint for the entire
32-token certified block, not for each vector multiplication. Its weight
allowance per certified token is

```text
(1/40)/32 = 1/1280 = 0.078125%.
```

The paper proves a bound for one input vector and gives no union-of-probes or
shared-work theorem for 32 causal input vectors. If its direct constructor is
executed 32 times without unproved cross-query reuse, the favorable payloads
are

| Symbol interpretation | Direct 32-query payload / DFloat | Multiple of block budget |
|---|---:|---:|
| one bit | `21.036415%` | `8.414566x` |
| Q4 | `84.145660%` | `33.658264x` |
| BF16 | `336.582640%` | `134.633056x` |

These three rows describe direct no-reuse execution. They are not a lower
bound against a new multiquery representation. The sound conclusion is that
the publication supplies no 32-token theorem; one may not assume the missing
reuse for free.

## 7. Why the constructor fails before hardware

The constructor fails independent E0 gates:

1. At `b=14`, even the ideal Q4 and BF16 single-query payloads exceed the
   complete 2.5% block allowance.
2. The favorable one-bit query payload fits, but its model-wide persistent
   edge payload is `6,911.57x` the global 8 GiB advice grant and has the wrong
   arithmetic semantics.
3. Native rounded BF16 and FP32 addition violate the semiring premise used to
   justify regrouping.
4. The publication returns full MatVec, not the scalar rank-one interface.
5. It provides no 32-causal-query shared-probe guarantee.
6. All practical address, counter, output, operation, cache-line, page, and
   shared-link costs were omitted favorably.

No SSD, PCIe, HBM, CUDA, latency, or model-forward measurement can repair
these representation and semantics failures. Hardware promotion is therefore
closed for this route.

## 8. Exact claim boundary

This audit establishes only

```text
the published finite-semiring lookup graph
does not establish the registered universal reference-exact 2.5% claim.
```

It does not establish

```text
every nonlinear numerical rank-one data structure is impossible.
```

The active ticket still requires one of two genuinely new deliverables:

1. a scalar-specific, globally coupled, bounded-word exact constructor with a
   complete 405B representation/query/32-token/native-rounding equation; or
2. a direct lower bound or finite adversarial certificate covering arbitrary
   adaptive global advice under the 8 GiB grant.

Until one exists, the correct project state is

```text
PRIMARY RESEARCH AND PROTOTYPE TRACK / REVISE
UNIVERSAL 2.5%: NOT ESTABLISHED
GENERAL IMPOSSIBILITY: NOT PROVED
CORE CANDIDATE: NONE
```

## 9. Reproduction

```powershell
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_finite_semiring_preprocessing_frontier.py `
  --output-dir results\e0_finite_semiring_preprocessing_frontier

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_finite_semiring_preprocessing_frontier -v
```

The derivation uses deterministic integer/rational arithmetic plus explicit
IEEE rounding controls. It performs zero model forwards and zero hardware
actions.

Observed validation was 9/9 focused tests, 36/36 related finite-word and
nonlinear frontier tests, and 544/544 repository tests when the repository
import path was declared for subprocesses. The first unconfigured full-suite
launch passed 543 tests and exposed one existing EXP-072A child-process
`vortex_runtime` import-path failure; that exact test passed after the path was
declared. The standard validation runner completed, and an independent output
directory reproduced `summary.json` byte-for-byte.

Canonical summary SHA-256:

```text
8188c0fde1f7289daf4052461292b775889229963132bbeed7c45bc6553e09f1
```

## Primary source

- Ryan Williams, *Matrix-Vector Multiplication in Sub-Quadratic Time (Some
  Preprocessing Required)*:
  https://people.csail.mit.edu/rrw/mat-vec3.pdf
