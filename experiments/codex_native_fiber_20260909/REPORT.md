# Native rank is not native bijectivity

2026-09-09. Bounded O3 continuation of the rank-normal program-carrier candidate
registered at `52facca` and subsequently executed by the other writer at
`c9ce7ff`. This package began at `207da38` and preserves both earlier results.

The exact native map of a mathematically full-rank dense matrix need not be
injective. We construct a finite BF16-weight family whose declared FP32 native
map sends at least `256^(n-1)` different BF16 vectors to one output. Consequently,
arbitrary bijective input and output encodings cannot turn this map into a
full-information coordinate copy. This is a new local obstruction to the native
lift of rank-normalization, independent of the earlier cost rejection of the
literal transformed Hadamard. It is not a general impossibility theorem.

No target-feasible arbitrary dense producer is constructed. No >=90% whole-model
removal, native HF continuation proof, or target memory/latency upper bound is
established. O1-O5 OPEN, O6 PARTIAL, THEORY_STATUS=NOT_ESTABLISHED,
HARDWARE_STATUS=NOT_TESTED, CORE_ADMISSION=false. Current 405B hardware is
unavailable per the user. Three qualifying new principles were not delivered.

## 1. A finite invariant of exact encodings

Let `F:D->Y` be a finite function. Let `E:D->D'` and `H:Y->Y'` be bijections.
For any y, the fiber of `H F E^-1` over H(y) is exactly `E(F^-1({y}))`.
Thus every fiber cardinality is preserved. In particular, a noninjective F
cannot become an injective full-information copy, permutation or embedding by
these encodings, regardless of their construction time, circuit complexity,
nonlinearity or storage. The domains may have different word formats; the
claim is about injectivity, not an assumption that BF16 and FP32 domains match.

If an augmented representation `(H(F(x)),s(x))` is required to be injective on
a fiber with at least M inputs, s needs at least M distinguishable values.
A fixed-capacity side state therefore needs at least `ceil(log2 M)` bits on
that fiber. This does NOT say the reference runtime needs to retain this state:
ordinary lossy native execution may intentionally forget the information.
It is a condition on an additional reversible lift, not a KV-memory lower bound.

The argument does not exclude rank-deficient copies merely because collisions
exist. Their complete fiber profile would need its own comparison. It does not
exclude global cross-wire encodings retaining other live inputs, nonbijective
state representations with an exact continuation relation, caches/side state,
or arbitrary nonlinear direct producers. No per-matrix advice split is used.

## 2. Full-rank, all-nonzero native dense witness

For any integer `2 <= n <= 16384`, construct

```
W = I + 11^T                 diagonal 2, every off-diagonal entry 1
d = ceil(log2 n)
delta = 2^(-24-d-8-2)
x_0 = 1
x_j = a_j*delta, j>0         a_j in {0,...,255}
```

Every weight and input is a finite exactly representable BF16 value: a_j uses
at most eight significant binary digits, and delta's exponent is between -35
and -48. No NaN, signed-negative zero, subnormal, overflow or flush-to-zero
behavior is needed. There are exactly `256^(n-1)` distinct input vectors.

W has full algebraic rank over the reals. If `Wx=0` and `s=sum x_j`, then
`x=-s*1`, hence `s=-n*s`, giving s=0 and x=0. Equivalently det(W)=n+1.
At even n, reducing its entries modulo two also gives a full-rank GF(2) matrix;
that reduction changes the numeric weights and is not itself a native lift.

The declared native ABI rounds each product separately to FP32 RNE, then
uses the fixed balanced FP32 RNE reduction tree, padding with +0 to a
power-of-two width. FP32 output words are observed. A later BF16 RNE store
does not change the final constants 1 and 2. This is an explicit software ABI,
not an assertion about an unspecified HF/CUDA kernel.

### Node-by-node native proof

Each row has exactly one large leaf, at input coordinate zero. Its value is
2 for row zero and 1 for every other row. All other leaves are nonnegative
integer multiples of delta. Their total coefficient is at most `255*n`, and
`255*n <= 4,177,920 < 2^24` in the registered range.

Every small product is therefore exact FP32, including the diagonal factor
two. Any subtree without the large leaf sums an integer multiple of delta
with coefficient below 2^24; every intermediate sum in such a subtree is exact
and normal or +0. Its value is bounded by

```
255*n*delta <= (255/1024)*2^-24 < 2^-24.
```

Here `2^-24` is half the upward FP32 ULP at 1. At 2 the half ULP is larger.
When a node containing the large leaf meets a tiny-only subtree, strict
round-to-nearest therefore returns the same large value. Induction up the
tree proves that the native output is always `[2,1,...,1]`. This proof accounts
for each original rounding event, rather than inferring it from the exact sum.

Thus F_W has a fiber of size at least `2^(8(n-1))`. It cannot be transformed
into an injective full-rank coordinate copy by independent bijective encodings.
At n=16384 the fiber contains at least `2^131064` inputs. If an augmented
reversible lift retains the distinctions, the side-state information term is
at least 131064 bits, or 16383 bytes. This small local term is neither an
8-GiB failure nor a whole-model state lower bound and cannot be summed across
operators without a separate simultaneous-state proof.

## 3. Later derived binary-source corollary

This corollary was derived after the frozen I+11^T experiment and was not
substituted into its controls or result counts. Define binary B with row zero
all ones; for i>0, row i is all ones except B_ii=0. Every row has first entry
one. Subtracting row zero from each other row gives -e_i, so det(B)=(-1)^(n-1).
It has full rank over both the reals and GF(2), with only n-1 zero coefficients.

On the same BF16 input box, every row's large leaf is one and every tiny sum
is at most `255*(n-1)*delta`. The preceding native induction gives the constant
all-one output vector. Thus even keeping the source weights literally binary
does not permit a universal lift from full-rank GF(2) copy form to an injective
native copy form. No new exhaustive binary experiment is claimed for this
proof-only corollary. Its algebra and rounding argument are explicit.

## 4. What actually ran, and the paid bounds

[run.py](run.py) constructs the family algorithmically and evaluates the
registered cases. For n=3 it exhausts all 65536 vectors and checks 196608 FP32
output words against the fixed constants. All match. A fixed n=2 control
with x=[1,1] returns [3,3], distinguishing the evaluator from a constant-output
stub. The 1024 representative input-word checks confirm BF16 representability
for every a at the four registered widths.

The full reference processes every coefficient: n^2 products and
`n*(next_power_of_two(n)-1)` additions per vector, plus generated-weight/input
accesses, list construction and output conversion. Materializing the BF16
checkpoint costs n^2 writes and 2*n^2 payload bytes; reading and validating an
existing checkpoint against this special family costs n^2 coefficient reads.
The rule does not compress arbitrary checkpoints or bypass those validation costs.
At n=16384 these reference terms are 268435456 products, 268419072 additions,
and 536870912 BF16 matrix bytes, before inputs, outputs, workspaces and metadata.

The two n=16384 controls did NOT run that full dense reference or allocate
the full matrix. They used a proved positive-zero pruning rule for only the
registered one/two-nonzero-input vectors. A subtree of +0 leaves is +0; adding
+0 to a positive finite value preserves its word. The routine follows the
same remaining logical tree nodes. The large controls evaluate two leaf
products per row and 14 remaining addition nodes per row, with generated
coefficient access, hash/set/dictionary operations, and O(n) output storage
also paid. Sixteen small controls (n=2,3,16,64) compare this routine directly
against the full tree, with zero differences. The two large outputs match.

The family generator/reference/proof checker are finite CPU procedures. The
full reference uses O(n) temporary numeric words for one row plus O(n) outputs;
the Python objects, allocator, instructions and interpreter are extra physical
costs, not their packed numeric size. The pruned helper uses O(k log n) work
per row for k positive leaves (at most two here), O(k) live node maps and O(n)
outputs. Enumeration costs 256^(n-1) reference calls if extended naively;
only n=3 was enumerated. That exponential verifier is not an admitted compiler,
decoder, unbounded-preprocessing permission, or a speedup for arbitrary inputs.

CPU/RAM/SSD/PCIe/HBM/GPU, initialization, metadata, RNG/KV/cache,
verification/fallback and allocator costs of a real integration remain paid
and unresolved. The reference performs no native sampler or cache transition.
An unchanged Python RNG state is only a local no-RNG-consumption check. No
HF reachable-activation trace, actual 405B, CUDA, <=8GiB GPU allocation,
native4BQ4 latency or TTFT was tested.

## 5. Candidate comparison and remaining construction

The three program-carrier principles at `52facca` remain the relevant comparison,
not three renamed ideas introduced by this audit. Their explicit router and
Patricia realizations failed their paid program-traffic bounds; the literal
rank-normal/nonlinear realization restored dense work. This audit adds a
separate native obstruction to the isolated full-rank copy claim. The broader
globally co-designed encoded graph remains OPEN and cannot assume native
bijectivity from algebraic rank.

Primary-source screening found no basis to promote the following known methods:

- [Williams' finite-semiring algorithm](https://people.csail.mit.edu/rrw/mat-vec3.pdf)
  uses exact associative/distributive operations and a compiled lookup graph.
  The repository already accounts for its physical payload and missing native
  rounding lift in `E0_FINITE_SEMIRING_PREPROCESSING_FRONTIER.md`.
- [Grammar-compressed matrix multiplication](https://arxiv.org/abs/2203.14540)
  bounds query work by the compiled grammar size. That does not establish a
  sufficiently small grammar and original rounded reduction for every checkpoint.
- [0-1 multiplication via clustering](https://link.springer.com/article/10.1007/s00224-026-10286-7)
  exposes Hamming-cluster radii in its exact entry/full-product costs. No uniformly
  small radius or native ordered lift follows for arbitrary dense checkpoints.

These are scope decisions, not fresh experiments or universal impossibility
proofs. No more favorable hardware, free decoder, selected checkpoint/input
distribution, per-matrix advice split or sparse-delta premise was introduced.

The next decisive core object is still a finite paid arbitrary-native dense
representation and query/state program. For an encoded-state route it must
specify a native noninjective map and its continuation relation, or explicitly
account for side state, without evaluating the removed dense function inside
the encoder/decoder. Rerunning this collision box cannot supply that object.

## 6. Reproduce

With Python 3.10+ and standard library only, from this directory:

```
python run.py --output-dir <new-directory>
python verify_bytes.py
```

The generator writes explicit LF bytes. Local Git attributes retain LF; the
manifest and actual index/remote bytes are checked independently. Canonical
[summary](results/summary.json) SHA-256:
`8b20bfcaca5d88b044b0bbaefd0d4b8b35f2a3801e073b2b768b956bc0f140e8`.
See [validation](VALIDATION.json) and [independent review](INDEPENDENT_REVIEW.md).
Luna/low source extraction and Sol/high proof review were requested explicitly;
the spawn result disclosed task IDs but did not independently verify the actual
model selection. Agreement is not proof; the finite-map and native-node
arguments above are the basis of the conclusion.
