# Crossing extension: a non-enumerative piecewise dyadic map

## Status and distinction

IMPLEMENTED/CHECKED: a constant-level representation, exact update, and constructive
threshold inverse for a **fixed itinerary of uniform dyadic RNE grids**. This is
`crossing.py`; the frozen checks are `validate_crossing.py`.

DERIVED, NOT IMPLEMENTED: the full native scalar-stream chart partition described
below, including its signed-zero tags and overflow constant tails. Its finite
algorithm and upper bound are supplied, but the executable native chart constructor
and independent all-path check remain an exact bounded next subroutine. No full HF,
actual GEMM ABI, target hardware, or cheap coefficient producer is claimed.

The fixed-binade package remains preserved and unchanged in scientific scope.

## 1. Chart lemma

Let the initial input be a=k*h0, k integer, with h0 a positive power of two. Consider
an ordered fixed itinerary

    F0(k)=k*h0
    Fj(k)=Q_hj(F_(j-1)(k)+cj),

where every hj is a positive power of two, each cj is an exact dyadic, and Q_h
rounds to the nearest multiple of h with even-integer ties. The actual native
rounding grid may depend on k; this lemma applies after a chart has fixed that
itinerary. It does not assume all real inputs use one itinerary.

Set H=max(h0,h1,...,hN), T=2H and P=T/h0. P is an integer power of two. Then:

1. F is monotone
2. F(k+P)=F(k)+T for every integer k
3. On residues 0<=k<P, F takes at most THREE distinct values

Proof of (2): T/hj is an even integer. Therefore Q_hj(z+T)=Q_hj(z)+T,
including midpoint ties; translation by only H would not suffice in general.
Induct through the ordered stages.

For (3), consider the first stage whose output grid is H. Before that stage the
prefix is monotone and T-equivariant, so over one half-open input period its
values lie between G(0) and G(0)+T, endpoints included. Rounding to H leaves at
most three values separated by H. Subsequent finer-grid stages transform those
at most three values independently and cannot create additional levels. If H=h0
and no H-grid stage exists, the initial period has only two integer input residues,
which proves the same bound. Later new larger grids restart this argument.

Three really is needed: with h0=1 and a single Q_2(k), residues k=0,1,2,3 give
0,0,2,4. The third value equals the next period's first value. Counting only two
levels from a half-open interval would be incorrect because its image need not be
half-open.

Thus store P,T, at most two nonzero cutpoints b1,b2, and at most three levels vi:

    k=qP+r, 0<=r<P
    F(k)=qT+vi when bi<=r<b_(i+1).

No enumeration of P residues is needed.

## 2. Exact monotone inverse: no free cut finder

For a representation with cuts bi and levels vi, the least integer k satisfying
F(k)>=theta is the minimum, over its at most three residue intervals, of

    P*ceil((theta-vi)/T)+bi.

For F(k)>theta use floor((theta-vi)/T)+1 instead of ceil. Each residue interval is
constant within a period; these are its first satisfying periods, and taking the
minimum gives the global first satisfying integer. Monotonicity proves all later
indices satisfy the predicate. This is an explicit constant-count algorithm over
bounded multiword integers, not an assumed inverse and not a scan of a period.

## 3. Constructive update

To append Q_h(F+c):

- If h<=H, T/h is even, so apply Q_h(vi+c) to each stored level, retain cuts, and
  merge adjacent equal levels
- If h>H, set T'=2h and P'=T'/h0. Let y0=Q_h(F(0)+c). On 0<=k<P', possible new
  values are y0, y0+h and y0+2h
- For each target y=y0+h,y0+2h, let theta=y-h/2-c. The new value reaches y when
  F(k)>=theta if y/h is even, and when F(k)>theta if y/h is odd. Use the inverse
  above, clip to the new period, and omit absent levels/zero-length intervals

This gives the exact new cuts and levels with a fixed number of arithmetic
operations. The strict comparison at odd target indices is essential. All products
and all updates are still executed. This is a scalar map representation result.

## 4. From itinerary charts to native scalar rounding: derived algorithm

For finite exact addends c1,...,cN, each native scalar operation a -> RN(a+cj) is
monotone. Hence every prefix is monotone. A preimage of a threshold or a singleton
rounding boundary is an interval of initial accumulators.

Partition initial FP32 finite values into signed normal-binade grids, positive and
negative subnormal grids, and the two zeros. There are 512 such initial pieces
when zero tags are separated; optionally include two infinity constants for 514.

Partition the exact sum z at each step into:

- Positive/negative normal magnitude-binade intervals, with dyadic grid selected
  from that exact-sum binade; on a closed exact-sum binade, rounding to that grid
  agrees with native RNE, including its upper endpoint
- Positive and negative subnormal intervals with grid 2^-149
- The exact zero singleton, with its required signed-zero rule
- Positive and negative overflow tails beginning at the correct inclusive RNE
  midpoint, which become constant infinities

The disjoint positive normal intervals are [2^e,2^(e+1)) for e<emax,
and [2^emax,overflow_midpoint) at emax. Negative intervals are their reflected
(-upper,-lower] counterparts. Positive/negative subnormal intervals are
(0,2^emin) and (-2^emin,0); zero is a separate singleton. Overflow tails include
their midpoint. These are exact-SUM regions, not output-binade classifications:
a value in one region can round to its next-binade endpoint, and the next step
selects a new region from its own exact sum. Thus that endpoint never creates two
competing input-chart itineraries. The fixed-binade Grid's optional inclusion of
U is not reused as an overlapping partition here.

There are 513 such simple exact-sum regions;
allowing separate strict/non-strict preimages and special zero boundaries, B=1024
is a conservative bound on newly introduced cuts per step. A sharper B is not
needed for this result.

At step j, for each current input chart and next exact-sum region, intersect its
integer input interval with the constraints on F_(j-1)(k)+cj. Compute lower and
upper preimages using the explicit inverse above; apply the uniform-grid update
only on nonempty intersections. Overflow pieces become constant infinities.

Signed-zero bookkeeping is not optional. On a negative/positive exact-sum region,
a rounded zero has that sign. Exact cancellation of nonzero opposite terms gives
+0 in RNE. When both terms are zero, -0 is preserved only when both zero operands
are negative; the exact-product sign must be retained. Separate the zero singleton
preimage and attach one zero tag to each resulting chart. Finite products plus
infinite accumulator preserve that infinity; NaN/payload behavior and nonfinite
product operands are outside this derivation and require their declared original
instruction behavior.

Because a monotone prefix crosses each region boundary at most once over all
initial inputs, a step adds at most B cuts globally, rather than multiplying every
existing chart by B. After N steps there are at most 514+BN charts.

A constructive ordered sweep computes intersections in O(K+B) chart actions at a
step with K charts: visit the input charts in order and advance through the ordered
output-region boundaries using the two endpoint values and the explicit inverse.
Each chart and each region boundary is advanced a finite number of times. This
avoids testing every chart against every region.

The resulting sufficient constructor bound is O(BN^2) bounded multiword actions,
O(BN) stored charts, plus all N product formations. Querying an already constructed
map takes O(log(BN)) chart search and one constant-level map evaluation. These are
paid, query-dependent maps; preprocessing them is not a model-once operation.

This section is a finite derived construction, not an executed all-native chart
implementation. The checked fixed-itinerary code does not yet emit or test all
these native intersections and zero/overflow tags.

## 5. Physical words and representative upper bound

For FP32, h0 can be 2^-149 and H can be 2^104. Thus P can equal 2^254. A nonzero
cutpoint may need 254 bits, and P itself has bit length 255 if stored as an integer.
Its power-of-two exponent can instead be stored directly. Treating each cutpoint
as a free 64-bit address would be wrong.

For finite BF16 operands, |cj|<2^256. Every virtual level after a rounded stage is
a multiple of 2^-149. Its magnitude in those units is bounded by a constant times
(N+2)*2^405; including sign, seven 64-bit limbs are sufficient at N=16,384. Two
cutpoints need up to eight 64-bit limbs total. A conservative 256-byte chart record
can hold three seven-limb levels, two four-limb cuts, 32-bit initial interval bounds,
exponents and flags at that N. This is a derived fixed-layout payload allowance,
not the measured Python size, and allocator/index metadata is additional.

With B=1024 and N=16,384 the loose sufficient bounds are:

    charts per row <= 16,777,730
    chart payload at 256 bytes <= 4,295,098,880 bytes per row
    16,384-row square payload <= 70,370,900,049,920 bytes
    chart-update sum <= 137,438,986,240 per row

These are conservative upper bounds, not necessary sizes or a universal lower
bound. Their looseness cannot by itself prove all compact maps infeasible.
Nevertheless this explicit constructor has no 10x traffic route: it still reads
and classifies every original product. It adds large costs before even one output
is used. O(BN) rather than a literal 2^32 table is a representation improvement,
not an execution success.

## 6. Checks and retained witness correction

Both crossing validation runs passed the frozen population:

- 3,375 length-three dyadic itineraries
- 118,125 exact output comparisons
- 33,750 strict/non-strict threshold inverse checks
- 118,125 translation-equivariance checks
- 17 wide-period checks, including P=2^254 without enumerating it
- Maximum observed levels: three; zero mismatches

The first run's separately displayed witness happened to use two levels even
though its exhaustive population observed three. It was a valid map, but not the
three-level illustration described in its source comment. The source and result
are retained in `results/source_crossing_01` and `results/crossing_01.json`. The
second run changes only that displayed witness to Q_2(k), adds an explicit
three-level assertion, and repeats the same population. No failed theorem or
changed acceptance threshold is hidden.

These are checks of the fixed-itinerary algebra only. Native all-chart generation,
actual instruction ABI, GEMM reduction layout, HF outputs/state, GPU, and latency
remain NOT TESTED.

## 7. Exact source gap, beyond compact maps

The effect producer remains hard even when there are no crossings or rounding
errors at all. Take h=1, initial accumulator a=2^23, N<=2^23-1, and coefficient and
input values wij,xj in {0,1}. Every prefix is an exactly representable FP32 integer
inside [2^23,2^24), and the final guarded offset of row i is exactly

    delta_i = sum_j wij*xj.

Its low bit is the GF(2) matrix-vector product. Thus a universal non-enumerative
producer for these exact summaries already includes arbitrary preprocessed binary
MatVec, before native carries, half-ulp ties, crossings or state are added.

This reduction is not an impossibility theorem, a quantitative lower bound, or a
claim that these primitive inputs are reachable in every released Transformer.
An algorithm relying on a smaller reachable activation population must prove that
restriction and its causal construction rather than silently changing all-input
primitive quantifiers.

The remaining missing algorithm is still the encoded-checkpoint query that obtains
those heterogeneous offsets and ordered guard effects below the coefficient-work
budget. Compact chart closure supplies neither it nor a favorable workload
quantile. Do not proceed to a model/backend run without its explicit paid >=10x
path and the separate final 4B-Q4/VRAM/TTFT closure.
