# A finite guarded native transition algebra, not a fast effect producer

## Result

For a declared scalar binary RNE accumulator, an ordered sequence of exact dyadic
addends on one guarded normal-binade grid has a constructive summary consisting
of two integer offsets and one interval per incoming parity. The guard is built
with the summary. No 2^32-entry accumulator table or input enumeration is needed.

This closes a bounded representation lemma in O2/O3. It does **not** construct the
missing cheap heterogeneous Transformer effect source. Direct construction still
reads every coefficient and forms/classifies every query-dependent product. Its
work is linear with positive extra costs. CORE_ADMISSION=false; no 10x route,
model experiment, hardware test, HF equivalence, or O1–O6 closure is claimed.

## 1. Declared arithmetic and exact domain

Precision p has RNE ties-to-even, gradual underflow, and the usual finite exponent
range. A scalar FMA means RN(a + w*x) with the product exact until that rounding.
Actual GEMM/ATen/tensor-core schedules, FTZ/DAZ, intermediate precision, NaN payloads,
reduction trees and register frontiers are not identified with this ABI.

For exponent e, set h=2^(e-p+1), L=2^(p-1), U=2^p. Positive incoming lattice
indices are k in [L,U] if U*h is finite and [L,U-1] at the maximum exponent.
The negative chart uses the reflected interval. Including U at ordinary exponents
is intentional: this upper endpoint may be reached while a later operation still
uses the same valid grid window.

For e above the minimum normal exponent, the exact pre-round sum z/h must lie in

    [L-1/4, U+1/2].

At the minimum normal exponent, replace L-1/4 by L-1/2, because the predecessor
is a subnormal at spacing h rather than h/2. At the maximum exponent, replace the
upper bound by U-1/2 and make it OPEN: that midpoint overflows under RNE. Negative
windows are reflected, including endpoint openness.

These windows are sufficient, not claimed to be the largest possible accepted
set. In particular the upper half-cell above U is included only while uniform
h-grid rounding agrees with native rounding. The wider native cell above U is
not silently treated as an h-grid cell.

Incoming zero/subnormal, infinity, NaN, and an escaped normal chart are not covered
by the normal summary. The first-escape executor calls the declared scalar reference
on finite products and finite/infinite accumulators. NaN payload behavior is
explicitly unsupported by the reference; an actual adapter must call its original
instruction there. Negative-zero products carry a separate sign bit. Fractions
alone identify +0 by convention and cannot represent a -0 product.

## 2. Leaf construction and proof

Write the exact scaled product t=(w*x)/h=q+r, where q=floor(t) and 0<=r<1.
For incoming parity s=k mod 2,

    d_s = q                        if r<1/2
    d_s = q+1                      if r>1/2
    d_s = q+((s+q) mod 2)           if r=1/2.

The last line rounds the integer k+q to an even result on a tie. It works for
negative t too; truncation toward zero would be wrong. Whenever the exact-sum
window holds, the native result is (k+d_s)*h.

The window and state-index bounds produce a leaf integer interval. For an exact
lower bound alpha on k, use ceil(alpha), adding one when alpha is an excluded
integer endpoint; for an upper bound use floor(beta), subtracting one when beta
is excluded and integral. Intersect with the state bounds, then tighten both ends
to parity s. Empty intervals have canonical offset zero.

The window proof is local: ordinary interior h-grid rounding is native rounding;
at the lower endpoint its predecessor is h/2 away (h at emin), which gives the
quarter-/half-cell cutoff. At U the previous gap is h, and the half-cell ending
at U+h/2 still rounds to the even endpoint U. At emax that endpoint is not finite,
and the RNE overflow midpoint must be excluded.

## 3. Ordered composition, with paid validity

Let A(k)=k+d_s on interval I_s and B(k)=k+e_s on J_s. Put

    r_s = (s+d_s) mod 2
    (B after A)(k) = k+d_s+e_(r_s)
    domain_s = I_s intersect (J_(r_s)-d_s).

Parity-tightening the intersection constructs the exact conjunction of all leaf
window guards. If the intersection is empty, store offset zero. This uses a fixed
number of integer additions, comparisons and parity operations. It is composition
of state functions, not reassociation of the numeric sum. Induction proves both
output preservation and equivalence to the complete conjunction of the declared
intermediate guards. Associativity follows from function composition and interval
preimages; no desired incoming accumulator is used in construction.

For a nonempty parity domain the offset is the difference of two indices inside
one state interval, so |d_s|<=2^(p-1). Six signed 32-bit fields hold the two offsets
and four bounds for FP32. Exponent/sign/format/empty flags are extra, so 24 bytes
is only a payload bound, not an object-size measurement. The Python reference uses
Fraction and Python objects and does not claim that packed footprint.

## 4. Constructive first escape and state preservation

The stream executor keeps an initial lattice index and its running guarded
summary. For each exact product it constructs the leaf and composes it. If the
new guard contains the initial index, it extends the summary. Otherwise it decodes
the already accepted prefix, executes that one escaped original scalar instruction,
and starts a fresh grid when the resulting accumulator is normal. Non-normal
states execute the scalar instruction directly. Negative-zero product metadata is
passed on both scalar paths. It never needs a free membership oracle or an unseen
future accumulator. It reads the stream once; a failed leaf causes no full-prefix
replay. Each escape is charged, and there can be N escapes on N terms.

An induction on terms proves that the represented scalar accumulator equals the
reference after every term, including crossings, cancellation, underflow and
finite-product overflow in the supported scalar reference. This does not discharge
whole-kernel state: r independent accumulators need r such represented words, all
cross-lane/horizontal reductions must retain the actual original schedule, and all
observable output/KV/layout/RNG state still requires the unchanged outer computation.

## 5. Actual checks and evidence boundary

Preregistration preceded tests. Validation 01 passed. Before validation 02, the
bounded empty-domain refinement was separately registered; original source and
record are retained. Both runs use the same frozen population and pass:

- 34,026 single-leaf cases, 7,418 accepted
- 77,274 ordered three-addend cases, 41,626 accepted
- 8,748 ordered composition/associativity cases
- 38,420 escape and signed-zero cases
- 66 FP32/BF16 boundary and fused-vs-separate witnesses
- 0 mismatches

The tiny reference independently rounds by nearest neighbors in its finite format.
The FP32 reference binary-searches ordered finite encodings; it never creates a
2^32 table. Tests are reference/property evidence, not a proof for an HF kernel.
Wall times in the records describe these Python tests only.

Two exact FMA witnesses prevent an invalid product-plus-add lift:

    w=x=2^-75, a=2^-149:
    fused FP32 = 0x00000002; separately rounded product/add = 0x00000001

    w=x=2^64, a=-2^127:
    fused FP32 = 0x7f000000; separately rounded product = +infinity

Finite BF16 operands produce an exact product with <=16 significant bits, but its
exponent may underflow or overflow FP32. Product precision alone is insufficient.

## 6. Full direct cost and the missing producer

Let N be a scalar stream length, M the number of output streams, r the actual live
accumulator count, C the number of first escapes and S the number of non-normal
scalar steps. With one-pass evaluation C+S<=MN.

Construction/query work: MN coefficient references, MN dynamic input uses, MN exact
products, up to MN leaf classifications, MN constant-size compositions, MN guard
checks, C+S scalar fallback operations, and O(C+S+r) state decodes. The constructor
is query-dependent, not model-once. The original weights remain necessary. A
streaming packed representation uses O(r) summary state plus products/workspace;
materializing all leaves would add O(MN) records and is unnecessary. Actual Python
memory is not measured as a target allocation.

For BF16 originals the direct coefficient payload remains 2MN bytes. Across the
registered 405,849,243,648 parameters it is 811,698,487,296 bytes before all other
reads and traffic. A favorable 10x traffic gate would require <=81,169,848,729.6
bytes for that same contribution. This construction stays at 100%, and usually
adds arithmetic instead of removing it. Products, label addressing and every
input-dependent guard are paid even when the eventual summary is only 24 bytes.

What can be reduced: a summary can replace many repeated applications of the SAME
already-built additive stream to different initial accumulators; it can reduce
sequential dependency depth via function composition. Neither gives different
output rows their heterogeneous products, nor yields additional causal tokens.
At least ten genuinely needed applications of the identical stream would be
necessary even for a favorable tenfold amortization of its construction. No such
reuse is supplied by ordinary dense decode, and query-independent preprocessing
cannot cache summaries for arbitrary activation vectors without solving a new
producer problem.

A direct non-enumerative sharing attempt groups equal coefficient words within
each input column, computes each distinct w*x_j once, and broadcasts its exact
classification. This is finite and avoids enumerating possible x. If column j has
D_j distinct words, product count becomes sum_j D_j, but MN coefficient-position
labels are still consumed, and MN ordered guard/summary actions remain. For
arbitrary matrices D_j may be min(M,65,536); all-input acceleration does not follow.
This is a charged shared product dictionary, already within the scope of prior
local-reuse exclusions, not a new global nonlinear information source.

If all products are integer multiples of one power-of-two quantum h, and
every original partial sum is a representable integer multiple of h, rounding
vanishes. One sufficient finite condition is h>=2^-149, every prefix integer has
absolute value <=2^24, and the resulting values stay below FP32 overflow. Integer
prefix extrema can certify it without invoking FP32 accumulation, but computing
those extrema still consumes every product. This does not create a free selector.

The next essential gap is specific: a non-enumerative encoded-checkpoint query
algorithm that emits each heterogeneous ordered summary (including extrema/guards
and actual multi-register semantics) without reading/classifying the dominant MN
coefficient-product positions, with total representation/address/decoder/repair
cost <=one tenth of the original contribution. No such algorithm is constructed.
A smaller map does not imply a cheaper map producer.

## 7. Quantifiers, startup and final-target accounting

Correctness is all-input within the declared scalar ABI and guarded domains; the
escape algorithm covers its explicit finite-product domain. That is separate from
any speed claim over a distribution. A distributional p50/p95 claim needs a frozen
common workload and charged crossing/escape tails. A worst-case one-pass bound
contains all terms; a short-session cold cost includes weight load and metadata.
No amortization horizon is assumed, and initial/prefill costs are not hidden.

Even a future tenfold source would not by itself meet the final goal. A sufficient
latency upper bound must add source decoding, CPU/SSD/PCIe/HBM traffic, native
nonlinear layers, attention and required KV, sampler/RNG, allocation/workspace,
repair and startup/TTFT, then establish the requested quantile ratio against the
same-machine 4B-Q4 baseline using actual measurements or a valid coupled bound.
The baseline quantiles and complete target allocation are not measured here.

THEORY_STATUS=NOT_ESTABLISHED (whole mission)
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
README_CURRENT is owned by the integrating parent, not changed by this package.

## References and reproduction

The transition derivation here is direct integer/Fraction reasoning. It is not an
unprecedented-algorithm claim. Relevant primary context:
- IEEE-style FMA and rounding overview: https://docs.nvidia.com/cuda/floating-point/index.html
- Exact sequential-result optimistic parallelization precedent: https://ic.ese.upenn.edu/pdf/fpaccum_arith2007.pdf
- BF16 product/accumulator context: https://arxiv.org/abs/1904.06376

Run `python experiments/binade_transition_20260930/validate.py --output NEW_PATH.json`.
The validator refuses to overwrite an existing result. `results/validation_01.json`
and `results/validation_02.json` plus logs retain actual counts and runtime.
