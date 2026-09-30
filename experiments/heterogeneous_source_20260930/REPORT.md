# A paid heterogeneous suffix source, and its unresolved coverage lemma

## Result first

A **finite scalar algorithm** can omit an accumulator prefix without generating
its exact effect: construct a sound interval for that prefix, run the original
ordered suffix on both extremal states, and accept the common **word** if they
coalesce. Each output stream can coalesce to a different word. A failed check
executes the entire original stream; that work is charged.

This is a genuine alternative to first producing every prefix product, or to
requiring many different output rows to share one answer. It supplies a
conditional >=10x work/traffic route with explicit costs. It does **not** supply
universal cheap coverage. A direct analytical screen shows that ordinary additive
suffixes cannot collapse a broad prefix interval. A zero-product suffix reduces
successful prefix certification to computing the skipped exact result itself.

The strongest next lemma is therefore a **paid causal narrow-enclosure source**,
with a certified coverage/cost bound on the unchanged mission. It is not supplied
by this package. No model, GPU, numerical experiment or core promotion occurred.

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
THREE_QUALIFYING_NEW_PRINCIPLES=false
FULL_MISSION_O1_O6=OPEN
HANDOFF_STATUS=IN_PROGRESS

## 1. Fixed theorem and current obligation

The final theorem remains the one in [the registration](PREREGISTRATION.md) and
[the constructive contract](../../docs/CONSTRUCTIVE_THEORY_CONTRACT.md): arbitrary
public unmodified dense HF checkpoints, all legal inputs/contexts and original
RNG/required state, batch one, one GPU with complete peak <=8 GiB, same-machine
native 4B Q4 p50 <=1.2x and p95 <=1.5x, and the existing TTFT requirement. No
model narrowing, approximate state, free preprocessing, hidden remote compute or
uncharged fallback is introduced.

The immediate problem is O2/O4: the
[implemented native chart constructor](../binade_transition_20260930/native_charts/REPORT.md)
now gives exact finite scalar maps but consumes every coefficient/input pair.
This report changes the **proof unit**, from constructing an omitted prefix's
effect to proving that a suffix erases the prefix's uncertainty. The whole-model
O1-O6 ledger remains open, as detailed in section 10.

## 2. Three different execution directions, honestly classified

This comparison is not a claim that three new qualifying cores were invented.
The previous grammar mechanism is retained only as the existing baseline; it is
not re-run or renamed. The three directions reverse different premises:

| Direction | Reversed premise | Finite execution and exactness | Charged >=10x condition | Decisive gap or check |
|---|---|---|---|---|
| Checkpoint syntax sharing | Each row is independently generated | Evaluate the ordered grammar of scalar transition leaves; compose only identical reusable subexpressions, preserving original order | Query products D=sum_j distinct_i(w_ij), grammar actions, child/label reads, map composition and metadata together must fit 0.1 of the original contribution | D can be MN; even shared leaf products leave ordered heterogeneous actions. Current grammar costs fail this route; no rerun |
| Native suffix synchronization | The skipped prefix's exact state must be produced | Bound the prefix without query-time prefix reads; run a fixed original suffix at the two bounds; equal endpoint WORDS force the actual result; otherwise full exact fallback | Suffix coefficient/input reads, two endpoint instruction streams, enclosure work and all failed full streams must together fit 0.1 | Explicit construction below. Broad bounds provably cannot synchronize unless the suffix can supply enough rounding displacement; narrow cheap bounds and coverage remain missing |
| Native inverse-threshold interrogation | Build the entire forward map, then evaluate it | Binary-search the final ordered word. Pull each proposed threshold backward through the exact scalar FMA preimage; compare its first preimage with the known initial word | A checkpoint index must answer ordered inverse-threshold compositions at less than 0.1 of forward source cost, including construction, probes and search | A direct finite implementation reads all MN products and performs up to 32MN inverse updates; compact inverse formula is not a cheap heterogeneous inverse producer |

All three are causal and can be exact within their declared scalar ABI. They
have different information flows: share expressions, erase a dependence, or
interrogate output inequalities. None currently closes the fixed mission.
The suffix route is developed because it has an explicit finite non-enumerative
source and a transparent conditional saving; no small probability of failure,
fast selector or future answer is assumed.

The inverse baseline needs no perfect output selector: a threshold is merely a
binary-search trial. Its exact preimage is constructed in section 8. The
nonlocal index that would make those trials cheap is an open lemma, not a
primitive silently included in the algorithm.

## 3. Declared scalar ABI and source interface

Consider one independent stream of N original instructions, initially a_0=+0:

    a_j = RN32(a_(j-1) + w_j*x_j),  j=1,...,N.

The product is exact until this one FMA rounding. Coefficients and current inputs
are finite BF16 words; FP32 uses RNE, gradual underflow, signed zeros and ordinary
signed infinities. No NaNs are created from finite operands except through an
operation outside this declared scalar recurrence; after an overflow, adding a
finite exact product leaves that infinity unchanged. The sign of an exact zero
product is retained. This matches the scalar domain of the native-charts work,
not every actual tensor-core or ATen schedule.

Choose a fixed suffix length 1<=b<=N at construction time and put k=N-b. No
input-dependent pivot search is hidden. The prototype stores a prefix norm for
this one cut. A different menu of cuts must pay every extra record and failed
attempt; choosing a better cut after examining output is not allowed.

The source accepts the original W, current x, fixed cut and declared schedule.
It returns the exact final scalar word. It either produces it from two suffix
executions or calls the unchanged entire scalar stream. It does not delete any
observable prefix state: skipping is legal only when that state is internal and
is used solely by this accumulator's later instructions. Exposed intermediate
roots or cross-stream uses must still be generated.

## 4. Constructing sound prefix bounds without prefix reads at query time

### 4.1 Checkpoint-time construction

Scan the original coefficient stream once, retaining the original weights and
recording:

    R = sum_(j<=k) |w_j|, exactly.

Also record a finite-coefficient flag. Nonfinite coefficients are handled by the
original ABI path, not a guessed NaN payload. For N<=16,384, every finite BF16
number is an integer multiple of 2^-133 and is below 2^128 in magnitude, so R is
represented by a nonnegative integer of at most 275 bits at that scale. Five
64-bit limbs suffice for this norm payload. A conservative record can reserve
48 bytes including flags/alignment; this is a proposed packed representation,
not a measured object footprint. Shared N/k and indexing metadata are extra.

Construction reads all MN coefficient words once and performs all norm decoding,
multiword additions and writes. It is not free model-once work: charge loading,
construction time, original plus auxiliary storage, and any finite amortization
horizon. For actual kernels with several independent registers, a separate
record/cut is required for each stream unless another exact sharing is proved.

### 4.2 Query-time construction

Scan the current vector once to check finiteness and compute

    X = max_j |x_j|.

This N-word scan may be shared across compatible rows. It is neither omitted nor
multiplied into MN physical reads without a declared layout. If the inputs are
nonfinite, or the coefficient flag fails, take the original path.

Set u=2^-24 and eta=2^-150. Assume k*u<1 and form the exact rational

    C = R*X,
    B = (C + k*eta)/(1-k*u).

If this cannot give a finite enclosing FP32 radius, take the original path.
Otherwise let r be B rounded **upward**, and use the ordered native interval
[-r,+r], with [-0,+0] when r=0. The upward conversion is a representation
operation, not the original FMA rounding mode. It must not round B downward.

The algorithm uses no query-time skipped-prefix weight read or product. It pays
one norm read, the shared input maximum, exact bound arithmetic and conversion.
It deliberately uses a coarse bound; no actual prefix answer is supplied.

### 4.3 Proof, including the overflow guard

For a finite exact pre-round value z, absent overflow,

    |RN32(z)-z| <= u*|z| + eta.

For any finite prefix j<=k, write e_t for its rounding errors and
E_j=sum_(t<=j)|e_t|. The exact products satisfy sum_(t<=j)|w_t*x_t|<=C. Hence

    E_j <= j*u*(C+E_j) + j*eta,
    E_j <= (j*u*C+j*eta)/(1-j*u),
    |a_j| <= (C+j*eta)/(1-j*u) <= B.

This does not assume away a first overflow. If a first overflow occurred at
step j, the finite preceding errors give

    |a_(j-1)+w_j*x_j|
      <= (C+(j-1)*eta)/(1-(j-1)*u) <= B.

Require B below the RNE overflow midpoint. That makes the hypothesized first
overflow impossible. Requiring the upward FP32 radius r to be finite is a
simple stronger implementation guard. If the guard fails, no finite-bound
claim is made and the full original stream runs.

For a known nonzero finite initial accumulator, include |a_0| in C. This is an
extension of the same proof, not an assumption that all real kernels start at
zero or use a single register.

### 4.4 Finite bounded-word arithmetic procedure

The bound does not require arbitrary-precision arithmetic at unit cost. One
explicit procedure at N<=16,384 is:

1. Load R's five limbs. Decode current BF16 maximum X into its <=8-bit
   significand and exponent
2. Multiply those five limbs by the small significand, and shift into the exact
   common unit 2^-266. Form C+k*eta in at most nine 64-bit limbs
3. Multiply the numerator by 2^24 using a bounded limb shift. Divide by the
   positive integer D=2^24-k with ordinary base-2^32 long division, at most
   eighteen iterations. Each uses one bounded 64-bit division and remainder;
   both are charged if the machine does not provide them together
4. The integer quotient plus remainder represents B at unit 2^-266. Determine
   its bit length, retain the leading FP32 significand (or the subnormal grid),
   and round upward using the discarded-bit/remainder sticky bit. Handle the
   significand carry and reject an infinite enclosing radius

All arrays have fixed bounds here. The division, limb scans/shifts, branches,
input scan and record address are paid. This is a finite reference algorithm,
not an optimized implementation or a measured service-time claim. At other
lengths/precisions, the limb count and denominator/overflow guard must be
re-derived; the fixed nine-limb inventory is not universal by declaration.

## 5. Suffix execution, signed zeros, infinity, and fallback

Initialize l=-r and h=+r as native words. For each j=k+1,...,N:

1. Read original w_j and current x_j once into temporary operands
2. Execute the original scalar FMA on l and on h, keeping its exact zero-product
   sign, rounding, underflow and overflow semantics
3. Preserve both native endpoint words

Return l only if its entire native word equals h's. Otherwise evaluate the
entire original stream from a_0 and return its word. This fallback may reread
the suffix and does not pretend the previously simulated endpoints were the
actual prefix. The simple stated schedule charges that reread and recomputation.

**Correctness.** For fixed finite operands, the scalar FMA is monotone in its
accumulator on the ordered non-NaN words, including -0 before +0. Thus each
suffix update preserves l_j <= a_j <= h_j. If the endpoint WORDS coincide,
there is only one intervening word, proving the exact actual final word. If
they do not, the original stream is exact by construction.

Numerical equality of +0 and -0 is insufficient. Keeping both signed-zero
endpoints and comparing bits handles the distinction. A shared +infinity or
-infinity is a valid scalar coalescence result in this finite-operand domain;
opposite infinities are not. Endpoints initially at both infinities cannot
coalesce under finite additions, so the failed finite-bound branch need not
wastefully try them. NaN inputs/payloads, FTZ/DAZ, alternative FMA semantics and
FP status flags remain outside this scalar proof and use the original declared
ABI unless separately proved.

No additional randomness is consumed. A complete model lift must preserve the
original sampler and every required RNG/state root; the absence of randomness
inside this scalar subroutine does not establish that model-wide lift.

An optional implementation can stop double endpoint execution once the words
coalesce, then execute the remaining suffix once. The conservative budget below
still charges twice the entire suffix, so it does not rely on early coalescence.

## 6. What the paid 10x condition actually says

For M equal-length scalar streams, let f be the fraction requiring full fallback
on the current query, counting failed finite-bound guards as failures. Let
Q_bound be every integer/address/input-check cost, and let mu be per-stream
metadata bytes. The stated one-cut schedule has sufficient logical inventories:

    coefficient words <= M*b + f*M*N
    current-input reads <= N + M*b + f*M*N
    native FMAs <= 2*M*b + f*M*N
    metadata payload = mu*M
    enclosure/dispatch/word-comparison work = Q_bound
    output writes = M native words

Inputs can be reread according to actual caches and layouts; these counts do
not certify physical transactions. Additional cache-line, page, allocator,
workspace, transfer and synchronization traffic is charged by the actual
schedule. The formulas conservatively allow a suffix trial even for streams
whose bound guard would fail before it.

If t=b/N, the favorable coefficient-payload fraction is

    t + f + mu/(2*N),

before the separate input/output/address traffic; the native-FMA fraction is

    2*t + f,

before the distinct integer/bound/dispatch costs. One must budget all terms.
With N=16,384, b=256 and mu=48, the displayed coefficient fraction is
0.01708984375+f and the displayed FMA fraction is 0.03125+f. Before other
costs, the displayed conservative FMA budget meets 0.1 only when f<=0.06875,
and the displayed coefficient budget meets 0.1 only when f<=0.08291015625.
These are sufficient allowances for the stated conservative inventories, not
necessary fallback-rate conditions for every actual guarded schedule: streams
that fail the initial bound guard can avoid the suffix trial and use less work.
The lower allowance governs this conservative joint screen; neither bound
licenses ignoring Q_bound or calling a probe/FMA ratio a latency ratio.

A **conditional source-time upper bound** can use guaranteed machine services:

    U_source <= U_input_scan + U_norm_reads + U_bound_arithmetic
              + U_suffix_operand_reads + 2*M*b*tau_FMA
              + U_dispatch/output + U_all_failed_original_streams.

Use a serial sum unless a valid overlap schedule is proved. Guaranteed service
upper bounds, not peak bandwidth, are required. A sufficient 10x core gate needs
U_source <=0.1*L_original_contribution against a justified original-service
lower bound or a coupled same-machine proof. No such constants or coverage are
measured here. The counts identify a possible route; they do not establish it.

All original weight storage remains. The auxiliary norm metadata also remains.
For illustration only, distributing the registered 405,849,243,648 parameters
into streams of exactly 16,384 terms would yield 24,771,072 records, whose
48-byte payload is 1,189,011,456 bytes. That is **not** the actual tensor/register
schedule, a full allocation inventory, or evidence of <=8 GiB fit. Multiple
native accumulator streams can multiply the record count. KV, work buffers,
original/transformed hot data and allocator costs are still open.

Construction and cold load are paid at startup. If an amortization horizon H is
claimed, include T_build/H and state H explicitly, while separately satisfying
the original TTFT and short-session requirements. No H is assumed here.

## 7. Cheap decisive analysis: why broad additive intervals do not collapse

### 7.1 A general anti-coalescence screen for this interval method

Suppose the suffix starts at the **actual endpoint words** -r,+r, r>0. Let

    C_s >= sum_(j=k+1..N) |w_j*x_j|,
    A = (r+C_s+b*eta)/(1-b*u),
    E = (b*u*(r+C_s)+b*eta)/(1-b*u).

Assume b>=1, b*u<1 and A is below the overflow midpoint, which excludes a first
overflow by the proof above. Either endpoint differs from its corresponding
exact unrounded trajectory by at most E. The two exact trajectories differ by
2r, because they add precisely the same suffix products. Consequently the
native final endpoints differ numerically by at least

    2r - 2E.

If E<r, coalescence is impossible. Equivalently, a sufficient rejection condition
for this scalar interval attempt is

    C_s < r*(1/(b*u)-2) - eta/u.

For b=256 in FP32 this becomes

    C_s < 65,534*r - 2^-126,

together with the stated no-overflow guard. An empty suffix b=0 is the identity
and is outside the divided formula; it can coalesce only an already-singleton
interval. A stored suffix absolute-weight sum and the already computed X can
supply C_s without any suffix products, but its
construction/read/storage is an extra charge. This screen is a theorem about
this enclosure and suffix, not every exact algorithm or every tighter bound.

The result matters: RNE additions mostly translate a state interval. They do not
provide the strong deterministic contraction of an arbitrary monotone dynamical
system. Very large dominating suffix terms can round away a narrow prefix; an
ordinary suffix cannot remove a broad symmetric norm enclosure merely because
it has many instructions. Invoking a generic synchronization or mixing theorem
would omit its missing premise.

### 7.2 Constructive positive case, without calling it a model result

Nonzero native coalescence is real. For k<=16,384, take prefix coefficients
2^-20 and current inputs 1. Its bound radius from section 4 is below 1/16. Put a
suffix contribution 2^21 next. The lower neighboring FP32 gap at 2^21 is 1/8,
so every exact value in 2^21+[-r,+r] rounds to 2^21. Endpoints therefore coalesce.
Remaining sufficiently small positive 2^-20 terms are individually absorbed at
that magnitude. Different rows can use different BF16 power-of-two dominating
terms, yielding different exact final words.

This is an analytical existence witness for the mechanism, not a frozen
real-checkpoint experiment, representative coverage estimate or speed result.
A planted dominant term cannot establish arbitrary-model admission.

### 7.3 Exact-integer failure and the irreducible source obligation

The registered symmetric norm source fails on w_j=x_j=1 with N well below 2^24:
its prefix radius is broad, and a short ordinary suffix cannot coalesce it.
The screen above proves this without running a model. This rejects a universal
coverage claim for the stated source, not all conceivable narrower enclosures.

A sharper obligation survives any interval refinement. Allow binary prefix
weights and inputs, k<2^24, suffix coefficients +1 and suffix input words +0.
Every prefix partial sum is an exactly representable integer, and each suffix
product is the word +0. On the FP32 accumulator words from +0 upward, the whole
suffix is the identity. For a positive integer prefix sum, it cannot coalesce two
distinct enclosing native words: successful certification requires the exact
singleton equal to

    sum_(j<=k) w_j*x_j.

There is one signed-zero exception. A +0 suffix sends both -0 and +0 to +0, so
when the prefix sum is zero the interval [-0,+0] may coalesce without being a
wordwise singleton. It still specifies the unique numeric binary sum, zero;
no nonzero finite word is erased by adding +0. Thus, on this family, a universally
successful suffix source is already an exact numeric binary matrix-vector
effect producer. This is an exact reduction identifying the
missing work, **not** an all-algorithm traffic lower bound or proof of mission
impossibility. These are legal inputs of the declared scalar primitive. No claim
is made that a particular pinned HF model reaches all of them; avoiding them in
whole-model research requires a proved causal reachability restriction consistent
with the unchanged mission, not an assumed one.

### 7.4 Exact relationship to previous failures

- The output-envelope screen required many **different rows** to have a common
  output word and observed none on its frozen corpus. Here only two possible
  states of the **same row** must agree; different rows may return different
  words. That frozen rejection does not transfer automatically
- F-065's small native-reuse observations and F-036's absolute-unread decision
  certificates do not by themselves measure this source. The new dependency is
  prefix erasure by a fixed later native action, not an assumed exact reuse hit
  or a final token-margin proof
- Nevertheless, replacing R by an unspecified tight interval would repeat the
  old missing-information error. Prefix reads, a pilot/residual calculation,
  range lookup, verification and misses must all be paid. A new name does not
  evade that obligation
- The TOP/singleton demand executor previously generated no cheap backward
  source. Two-endpoint coalescence is an explicit stronger operation when it
  happens, but its coverage is not established by that old executor's correctness

No recorded failure scope is broadened or overwritten. No failed family is
reopened through a parameter sweep or model experiment.

## 8. Exact inverse interrogation, with its source cost exposed

Let S be the ordered non-NaN FP32 word set, including both zeros and infinities.
For a fixed finite product c, define f_c(a)=RN32(a+c). For a threshold word y,
construct

    g_c(y) = min {a in S : f_c(a)>=y},

with a sentinel when the set is empty. RNE's exact halfway boundaries, tie
inclusion and zero sign determine the threshold. It can be obtained by exact
rounding-cell arithmetic and a bounded search over the 32-bit word order,
without enumerating S. A naive binary search costs at most 32 scalar checks per
single-step inverse; the existing native chart machinery supplies analytic
piecewise inverses with its multiword arithmetic charges. Neither is a free
oracle. Signed zeros use the declared word order, not ordinary real equality.

For monotone f and h, pullbacks compose in reverse order:

    min {a : h(f(a))>=y} = g_f(g_h(y)),

with the appropriate end sentinel. Starting at a trial output threshold y,
apply these exact preimages for j=N down to 1 and compare against a_0. A binary
search over final word thresholds then finds the unique original output word.
At most 32 final-threshold trials suffice for a <2^32-word domain.

An honest direct implementation can precompute all MN exact products once,
paying all coefficient/input reads and product storage, then perform up to
32MN inverse-step updates (and the cost of each inverse). Recomputing products
avoids that storage but repeats the reads/work. Building complete forward maps
first simply returns to the already charged chart construction. A compact
single-step inverse reduces representation overhead, not heterogeneous source
cost. No 10x route is established here.

The remaining inverse-source lemma would be a checkpoint-derived, bounded-word,
non-enumerative index for these ordered input-dependent preimage compositions,
including its constructor, all physical probes and decoding. It cannot be
replaced by output truth tables, free query computation, a semiring assertion
about ordinary FP32 addition, or an unspecified fast decoder. Those are already
covered by F-052/F-060/F-062 and related ledgers.

## 9. Exact missing lemma and what would qualify next

The strongest constructive follow-on from the suffix route is the following
**unproved sufficient interface**, not a supplied algorithm:

Given the unchanged checkpoint, original kernel schedule and current causal
input/state, build/use a finite checkpoint-derived representation E. Produce
for each skippable scalar prefix an exact native-word interval I containing its
actual accumulator, and a finite schedule of native suffix actions. Charge
constructor, original and transformed storage, all input/coefficient/cold-index
reads, addresses, decoding, interval work, endpoint actions and failed streams.
Prove both:

1. Endpoint equality or the explicit original fallback preserves every live
   output/required state root and RNG transition for all legal continuations
2. The **whole-query** sufficient upper bound, including the actual failed-stream
   distribution and startup, meets the core and then final target inequalities

The prefix enclosure must be obtained before skipped prefix products. A singleton
from evaluating those products is a correct but unhelpful implementation. The
binary zero-suffix reduction shows why a coverage guarantee cannot simply be
postulated: it contains a genuine heterogeneous MatVec source problem.

A valid next proposal must supply a new information relation or constructor
which explains those bounds. It could instead abandon interval synchronization
and construct a different paid native effect source; this report does not impose
intervals or a particular grammar on all architectures. Enlarging the toy,
adding cuts, tuning thresholds or testing favored model traces without such a
mechanism is not admitted core work.

Correctness with full fallback is worst-case and exact within this scalar ABI.
Performance is not. A common workload's p50/p95 bounds require its actual causal
coverage and full-query costs, with correlations across rows/layers retained.
Average fallback fraction, per-row success rate, a favorable suffix-length ratio
or separate per-layer quantiles do not establish token quantiles. The target
comparison further needs a certified baseline lower bound or a coupled ratio;
upper bounds for both runtimes alone are insufficient.

## 10. Native/state lift and O1-O6 outcome

A real projection may have several live accumulators per output, lane exchanges,
partial reductions, mixed precision, tensor-core accumulation, FTZ/DAZ and a
final storage conversion. Its exact instruction/dependency schedule must first
be identified. An independent scalar suffix can be replaced only at its actual
register boundary, with original cross-lane/horizontal reductions then executed
in their original order. State carried into another instruction cannot be
silently discarded. Whole-kernel event/error semantics must also match whatever
the frozen interface exposes.

This package does not generate logits/KV/layouts, sampler state, cache metadata
or complete Transformer successor state. Matching a scalar final word is not
an inductive full-state correspondence. Neither native chart correctness nor
this suffix proof imports a universal HF/CUDA ABI.

| Obligation | This package establishes | Whole mission status / decisive remaining item |
|---|---|---|
| O1 | Finite norm constructor and fixed-cut scalar query, with explicit exact fallback | OPEN: universal cheap automatic source and actual schedules |
| O2 | Causal scalar suffix/endpoint/fallback schedule; no future output | OPEN: complete loading, prefill, decode and cheap full-model transition |
| O3 | Symbolic scalar endpoint proof, exact zero/infinity treatment in declared finite-operand ABI | OPEN: actual kernel semantics and full successor-state/RNG induction |
| O4 | Charged constructor/query/metadata inventory and finite multiword bound procedure | OPEN: narrow-source coverage and sufficient full machine-service upper bounds |
| O5 | Conditional work/traffic thresholds only | OPEN: 405B complete <=8 GiB, TTFT, same-population p50/p95 target |
| O6 | Reviewable text proofs, local links and content hashes | OPEN: independent full-theorem proof/algorithm artifact; no numerical validation claimed |

No whole-mission obligation is closed by calling the scalar fallback correct.
The additional work is an exact finite source construction plus a derived
anti-coalescence bound that narrows its missing coverage premise. It does not
increase established mission feasibility by the count of new lemmas.

## 11. Primary context and evidence boundaries

The use of extremal trajectories as a coalescence certificate has classical
precedent in monotone coupling from the past. Propp and Wilson's 1996 paper
studies perfect sampling for Markov chains; it does not claim rapid coalescence
for native arithmetic. Only the monotone extremal-state proof pattern is used
here. No stochastic mixing, extra RNG consumption or perfect-sampling runtime
is imported: [primary paper](https://www.stat.berkeley.edu/~aldous/206-RWG/RWGpapers/propp_wilson.pdf).

NVIDIA's primary documentation explains native rounding/FMA distinctions and
why changing operation grouping changes results. It supports preserving the
declared arithmetic contract, not the suffix coverage claim:
[official floating-point guide](https://docs.nvidia.com/cuda/archive/12.6.3/pdf/Floating_Point_on_NVIDIA_GPU.pdf).

The mathematical construction and inequalities in sections 4-9 are direct
reasoning in this report, not empirical outcomes or quotations from those
sources. Related source/claim boundaries are preserved in
[the rejection ledger](../../FAILED_APPROACHES_RECENT.md) and
[the scalar native charts](../binade_transition_20260930/native_charts/REPORT.md).
No novelty priority is claimed.

Only document/link/status/hash checks are run for this package. Existing
numerical evidence is read, not rerun or modified. No source program, native
backend, model test, target hardware, quantile measurement, publication or new
remote commit is produced. Parent integration owns any root-ledger and README
updates and any separately authorized repository handoff.
