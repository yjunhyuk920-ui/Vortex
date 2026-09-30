# Executable complete scalar native charts

## Result and unchanged rejection

The prior crossing note's DERIVED-only scalar chart constructor is now executable
in `constructor.py`. It builds the complete input-state map of an ordered finite
addend stream without enumerating initial FP32 words. Normal-binade crossings,
subnormals, both signed zeros, cancellation, overflow and initial infinities are
handled explicitly. The prior sealed files are preserved; this continuation has
its own preregistration, sources, results and checksum manifest.

This closes a bounded native scalar construction gap. It does not produce the
heterogeneous products of a dense Transformer cheaply. Every weight/input pair is
read and multiplied, and chart construction adds substantial work. No >=10x route,
core admission, actual GEMM/HF replacement, GPU experiment or target performance
claim follows.

## 1. Exact theorem and supported interface

Fix a binary format with gradual underflow and RNE ties-to-even. Inputs are every
valid non-NaN format word, including both signed zeros and infinities. Each addend
is a finite exact dyadic with a separate sign for exact zero. Invalid format words,
NaNs, non-dyadic Fractions and nonfinite FMA operands are rejected explicitly.

For a finite accumulator, one scalar step rounds the exact value a+c once. A
nonzero value that rounds to zero keeps its sign. An exact zero result is -0 only
when both the incoming accumulator and exact zero addend are negative zeros;
nonzero cancellation produces +0. An infinite accumulator stays that infinity
under a finite exact addend. This is an explicitly specified closed scalar domain.

For BF16 operands the FMA wrapper computes c=exact(w*x), preserves the product's
zero sign, and charges both operand accesses and the product. There is no premature
FP32 product rounding. The compiled map returns the final reference WORD for every
admitted initial word, not merely an approximately equal numerical value.

An actual ATen/GEMM/tensor-core implementation is not automatically this scalar
ABI. Reduction trees, multiple registers, FTZ/DAZ, nonfinite product operands and
NaN payload behavior need their actual declared semantics. Those are not supplied
by the present construction.

## 2. Initial state representation and disjoint regions

The initial ordered input domain uses one integer key per legal word. Negative
words are ordered from -infinity through -0; +0 through +infinity follow. Each
signed normal binade and signed subnormal interval has an affine map from that
key to its initial lattice index k. Zeros and infinities are singletons. FP32
initialization creates 514 intervals, using exponent loops rather than word
enumeration.

Each nonconstant chart stores its initial key interval, its key-to-k offset and
the previously checked PeriodicMap. Constant charts store an exact native word.

Native rounding is partitioned by exact SUM, with no overlapping binades:

- Positive normal regions [2^e,2^(e+1)), with the final upper boundary replaced by
  the open overflow midpoint at emax; negative regions are reflected
- Nonzero subnormal-output regions (hmin/2,2^emin) and (-2^emin,-hmin/2)
- Positive zero basin (0,hmin/2] and negative zero basin [-hmin/2,0)
- The exact zero singleton, separately from the two underflow basins
- Inclusive overflow tails [overflow_midpoint,+infinity) and its reflection

Here hmin=2^-149 for FP32. There are 515 disjoint regions. The zero basins include
the half-minimum-subnormal ties because zero is even. Exact cancellation and
-0+-0 cannot be conflated. Grid-region pieces exclude zero output, so no
nonconstant chart has an untagged zero. Existing constant zeros use their original
word and the addend zero sign for the next step.

A normal region's exact sum can round to the next binade endpoint. That endpoint
is a valid output; the next step selects a region from its own exact sum. The
partition does not assume the output stays in the exact-sum binade.

## 3. Constructive intersection and proof induction

For a nonconstant chart F(k), intersect its integer input interval with one region
of F(k)+c. The implemented inverse supplies the first k with F(k)>=theta or
F(k)>theta. Thus:

- Lower closed bound l: k>=inverse(l-c, non-strict)
- Lower open bound l: k>=inverse(l-c, strict)
- Upper closed bound u: k<inverse(u-c, strict)
- Upper open bound u: k<inverse(u-c, non-strict)

Clip these bounds to the chart's input interval. Empty intersections are dropped.
On a grid region, append its exact uniform-grid rounding using PeriodicMap.then.
On zero/overflow regions, emit the corresponding constant word. Exact nonzero
cancellation emits +0; existing zero constants already take the sign-aware scalar
path. Existing infinities remain constant. No membership or inverse oracle is used.

If a nonconstant piece has equal endpoint values, monotonicity proves it is
constant throughout; canonicalize it to the exact finite word. Merge adjacent
pieces only when their constant words agree or their complete map/offset records
are identical. Equality of a few samples never licenses merging.

Proof induction:

1. Initial charts are exact identities on a disjoint complete partition
2. A constant chart is composed with the declared scalar instruction exactly
3. A nonconstant chart's inverse intersections partition precisely its admissible
   initial integers; the native operation equals the selected uniform-grid or
   constant rule on each intersection
4. The checked itinerary update preserves that ordered uniform-grid composition
5. Constant canonicalization and exact record merging preserve every word

Therefore each appended step preserves the complete scalar successor map for all
admitted initial words. The tests below check this induction at every prefix in
an exhaustive tiny domain, but the theorem does not depend on sampled inputs.

## 4. Cuts, finite algorithm and termination

The native scalar step is monotone on ordered non-NaN words, including signed-zero
and infinity conventions. A region transition therefore introduces at most one
new input cut globally. The exact-zero singleton has two adjacent transitions,
already included in the 515-region partition count.

Consequently, after j products,

    K_j <= 514 + 514*j.

The earlier 1024*j bound remains conservative. This is a sufficient upper bound,
not a claim that all these charts occur or that all smaller representations fail.

The implementation sweeps ordered charts and advances one ordered region cursor.
A region is revisited only as the containing region of a later chart. At one step,
region-intersection work is bounded by K_j+515; the code checks that count. Every
inverse evaluates at most three exact quotient expressions. Construction loops
are finite over the input stream, current charts and 515 regions. No initial-word
loop, exhaustive program synthesis, period enumeration or hidden solver exists.

Ignoring precision-dependent constants only after exposing them, the sufficient
bounds are O(514*N^2) multiword actions and O(514*N) chart records. Querying an
already constructed map uses binary search among its charts plus one periodic-map
evaluation or exact constant return. The Python reference converts an exactly
representable result to a word via a bounded format-code binary search and checks
that conversion did not round its value; this additional work is not free.

## 5. Physical arithmetic, storage, startup and query costs

Constructor counters include chart visits, region advances/intersections, inverse
calls and their actual level-quotient terms, map updates, constant scalar steps,
scalar rounding calls, endpoint evaluations, merges and invariant checks. Query
counters are separate fields. Reference-check instrumentation is counted and is
not described as a fast production backend.

Generic dyadic inputs may have large numerator/denominator widths; those widths
are recorded. Operation cost depends on that supplied bit size and current chart
integer widths, not merely on N. BF16 products have the fixed bounds used in the
prior crossing proof: at most 16 significant bits, magnitude <2^256 and minimum
nonzero product 2^-266. No FP32 cast is inserted to reduce this range.

An explicit conservative bit-cost bound is O(514*N^2*b^2), using elementary
integer/Fraction arithmetic, where b bounds all operands and temporaries; this
is not a unit-cost unbounded-word RAM claim. All denominators are powers of two,
so a direct signed-integer/exponent implementation can use shifts, comparisons
and additions instead of general rational division. At N=16,384 with BF16
products, stored virtual levels fit the earlier seven-64-bit-limb allowance,
while pre-round thresholds may combine a magnitude below 2^271 with a 2^-266
quantum: allow nine 64-bit limbs for such temporaries. Old/new chart lists,
region records, scratch arithmetic and conversion checks are additional to chart
payload. The Python Fraction implementation is the reference; no packed backend
or its service rate is measured.

FP32 chart periods can reach 2^254, so a cut can require 254 bits. The earlier
conservative 256-byte packed-chart allowance at N=16,384 is a derived payload
layout, not the actual Python object size. Indexing, allocator and temporary
objects are extra. Python process RAM was not measured as a target allocation.

With the sharpened bound and N=16,384:

    chart count <= 8,421,890 per row
    chart payload at 256 bytes <= 2,156,003,840 bytes per row
    16,384-row square payload <= 35,323,966,914,560 bytes
    chart visits <= 68,992,122,880 per row

These loose upper bounds describe the explicit constructor's sufficient resource
inventory. They are not a universal lower bound. More decisively for this
algorithm, its coefficient traversal is always complete by construction.

For M output streams with N BF16 terms, the wrapper performs MN coefficient
accesses, MN dynamic-input accesses and MN exact products before/while constructing
its MN-dependent maps. Original coefficient payload is still 2MN bytes. Across
405,849,243,648 registered parameters this is 811,698,487,296 bytes per complete
contribution sweep, before charts, inputs, CPU/SSD/PCIe/HBM movement and state.
The constructor has no 10x work/traffic path.

Initialization includes regions, initial charts and original weight loading.
Changing the activation vector changes the product stream and requires new maps;
these costs cannot be called model-once preparation. The map pays off only for
repeated queries of the identical already-constructed addend stream, for which no
causal dense-Transformer reuse guarantee is provided. Warm query lookup time alone
would omit the dominant constructor. No short-session or TTFT amortization is
assumed. Retaining maps for arbitrary future inputs would need a new paid source.

## 6. Actual registered checks

Validation 01 passed. Before validation 02, API guards and supplied-bit accounting
were separately registered; first sources/results remain unchanged.

Both runs have:

- All 114 non-NaN tiny-format input words, including both infinities and zeros
- 1,000 complete three-addend sequences, 342,000 prefix-word comparisons
- 114 identity-word checks
- 64 deterministic FP32 initial boundary words over 11 streams, 3,520 identity/
  prefix comparisons
- 192 BF16 exact-product wrapper comparisons, plus explicit fused underflow,
  fused overflow/cancellation and negative-zero witnesses
- Explicit rejection of NaNs and nonfinite BF16 operands
- Zero mismatches

Validation 02 additionally checks three invalid-word rejections and one
non-dyadic rejection. Positive-input thresholds and populations were unchanged.

Tiny maps peaked between 18 and 35 charts. The selected FP32 constructors peaked
at 1,019 charts. Across their 44 appended addends they paid 23,358 chart visits,
24,499 region intersections, 47,515 inverse calls containing 105,036 quotient
terms, 19,829 map updates and 10,147 scalar rounding calls. Their assertions also
checked 28,941 chart records and 84,808 representation fields. These aggregate
counts span eleven independent maps; summed final-chart fields are not one map's
resident storage. Detailed per-stream/per-step counters are in the JSON results.

The selected native streams use already supplied synthetic exact addends. Their
coefficient/product counters are therefore zero, not evidence that products can
be obtained free. The separate BF16 wrapper records one coefficient read, one input
read and one exact product per pair. In an actual matrix application all such
products are required and charged.

Test durations are Python validation runtimes, not executor latency benchmarks.
The FP32 population is selected; the tiny population is exhaustive. Neither is an
actual HF model, hardware ABI or target performance measurement.

## 7. Remaining primitive and full-state boundary

The missing scalar map producer is now an executable finite construction, but it
is not the cheap heterogeneous effect producer required by the mission. The
unresolved subroutine is an encoded-checkpoint query algorithm that bypasses most
of the coefficient/product positions while generating their exact effects and
paying all addresses, decoding, construction and repair.

Even in the no-crossing exact-integer case, initial a=2^23, binary coefficients and
inputs, and N<=2^23-1 make each row's final offset exactly sum_j wij*xj. Its low bit
is binary MatVec. The new chart representation does not answer that source problem
without the original products. This is a reduction identifying an obligation,
not a universal impossibility proof or a reachable-HF-input claim.

A kernel with several independent scalar accumulator streams can use several such
maps only after its actual instruction schedule is fixed. Cross-lane operations,
other arithmetic and final reductions must execute in the original semantics;
the constructor does not discover that ABI. Original logits, required KV/layout,
RNG and the rest of the Transformer state are not generated or compressed here.

All-input scalar correctness is separate from distributional performance. No
workload distribution, measured 4B-Q4 baseline, sufficient same-machine latency
ratio, complete <=8 GiB allocation or TTFT bound is supplied. An eventual >=10x
source still would not by itself establish the final target.

THEORY_STATUS=NOT_ESTABLISHED (whole mission)
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN

## Reproduction

`python experiments/binade_transition_20260930/native_charts/validate_native.py --output NEW_PATH.json`

The validator refuses to overwrite an existing record. Main dependencies are the
sealed parent `reference.py`, `crossing.py` and the independent tiny nearest-neighbor
reference in `validate.py`; dependency hashes are included in this package's
SHA256SUMS. No new external package, checkpoint, GPU or service is required.
