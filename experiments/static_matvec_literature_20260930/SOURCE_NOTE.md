# Exact static-matrix / dynamic-vector upper bounds: packed arithmetic is not a missing cold source

Date: 2026-09-30 UTC. Read-only literature investigation plus this separate note.
Local base: `4ac2dd1` on `research/cloud-continuation-20260930`.
The remote origin is `https://github.com/yjunhyuk920-ui/Vortex.git`; this worker
has not independently verified the current remote branch or PR. Publication,
root-ledger integration and remote verification are outside this assignment.

## Result first

**No new constructive >=10x native heterogeneous source is established.** An
additional relevant primary result is Alman--Yu's packed small-matrix arithmetic.
It can expose several bit columns of one already-available query vector to fast
matrix multiplication, so future-token availability is not its missing premise.
Its direct application nevertheless consumes every matrix bit, and returns an
algebraic sum rather than the specified ordered native floating-point result.
Neither its asymptotic statement nor the existence of a short arithmetic circuit
provides a finite 64-bit, cold-backed, native-exact target upper bound.

This is an exact theorem shortfall, not an impossibility theorem. In particular,
this note does not claim every preprocessed matrix must be fully read at query
time, does not set hidden constants to one, and does not infer a lower bound
from an upper-bound exponent. No model run, large experiment, kernel build,
latency claim, root-ledger edit, commit or publication is made.

## 1. Intended final theorem and scope

The unchanged VORTEX theorem needs a finite automatic constructor for every
admitted unmodified dense checkpoint, a causal native-output/RNG/state-exact
executor, all preparation/storage/movement/work costs, <=8 GiB total GPU state,
and the same-machine native-4B-Q4 latency/TTFT comparison. This source audit
addresses only a possible component of O2/O3/O4. It closes none of full O1--O6.
The immediately missing component remains a cheap exact heterogeneous producer
before the dominant native projection has already been evaluated.

The [native chart constructor](../binade_transition_20260930/native_charts/REPORT.md)
constructs scalar maps but still requests all dynamic coefficient products.
The [grammar accounting](../tree_grammar_source_accounting_20260930/REPORT.md)
permits distinct source count D=MN. The
[suffix synchronization construction](../heterogeneous_source_20260930/REPORT.md)
has a conditional exact path but lacks broad cheap-bound/coverage closure.
These facts are not changed by an algebraic MatVec theorem.

## 2. Three mechanisms compared without relabeling them as new cores

| Mechanism | What is computed/shared | Source/native gap | Classification |
|---|---|---|---|
| Query-bit-column packed arithmetic | Turn one current integer vector into a Boolean matrix; use bounded-word integer/field matrix multiplication | Every matrix bit is still consumed; native rounding order is absent; finite primitive costs must be instantiated | Additional primary theorem audited here; not a qualifying source |
| Query-pattern preprocessing | Store checkpoint-dependent answers/graph edges selected by current vector pieces | The explicit catalog's storage/preparation and physical answer payload remain; native operations are not a semiring | Existing Williams/Four-Russians family, not reopened |
| Static shared algebraic circuit | Compile repeated forms and execute fewer scalar gates | Circuit descriptions/constants and all reads count; universal small native-order circuit with finite synthesis/resource bound is not supplied | Existing static-DAG/grammar/FMM families, not reopened |

These are a comparison of available mechanisms, **not three newly invented
qualifying execution principles**. RSR, low-VC row differences, Boolean zero
rectangles and broadword full scans retain their existing recorded scopes.

## 3. Confirmed primary source and narrowly stated theorem

[Alman and Yu, *Faster Update Time for Turnstile Streaming Algorithms*,
arXiv:1911.01351v1, November 2019](https://arxiv.org/pdf/1911.01351), Lemma 2 and
Sections 4.2/5: for a binary w-by-w matrix and a vector of w-bit integers, their
algebraic MatVec uses O(w) words and O(w^(2 omega/3 + epsilon)) standard word-RAM
time. A stronger word-RAM-MM model gives O(w^(omega/2 + epsilon)). The reduction
writes vector bits as columns, multiplies over a field whose size exceeds w,
and reconstructs integer results. The stronger model provides a small-matrix
multiplication word operation. Their recursive theorem uses a finite tensor
rank decomposition, but its asymptotic formulation does not specify a concrete
64-bit instruction schedule or resource constant for VORTEX.

The following counts, finite packing construction and native witnesses are
this note's direct derivations, not performance results claimed by that paper.
No novelty priority is claimed for packing or for those elementary deductions.

## 4. Exact finite consequence for one cold binary square

Take n=16,384 and hardware word width h=64, distinct from the growing theoretical
parameter w. Partition a binary n-by-n A into 64-by-64 tiles. There are exactly
256^2=65,536 tiles. Pack each tile in 64 source words, without compression or
matrix-dependent omission. The layout is finite and its preparation reads all
n^2 bits and writes all n^2 packed bits, plus layout work.

For one unsigned 64-bit input-vector chunk, form its 64-by-64 bit matrix B.
For each A tile, compute C=A_tile B over F_67. Every mathematical entry of C
lies in [0,64], so its representative in F_67 is the exact integer count.
Reconstruct the 64 local integer outputs and add them into the corresponding
global output rows. All bit columns are known now: no future token or oracle
is used. Unsigned global outputs need at most 78 bits (two 64-bit limbs), not
one word. Signed inputs require a charged signed representation/reconstruction.

The explicit packed-source implementation loads:

    65,536 tiles * 64 source words = 4,194,304 words
    4,194,304 * 8 bytes            = 33,554,432 bytes = 32 MiB.

That is **100% of this binary matrix's packed source payload**, before input,
output, conversion, field arithmetic, transform storage, instructions or
workspace. This is a count for the specified scan algorithm, not a lower bound
for arbitrary preprocessed data structures. If some tiles are genuinely hot,
subtract their actual resident bytes; no entire-model residency follows from
one 32-MiB square fitting individually.

Let K_64 denote a completely specified finite kernel's cost for this one tile,
including its operand conversions and field operations, let E encode the input,
and let J reconstruct/combine outputs. The serial count is

    T_query = E + 65,536*K_64 + J + source/input/output movement.

The theorem does not numerically instantiate K_64. Treating its big-O constant,
packing cost or tensor-decomposition threshold as one cannot do so. Choosing
an ordinary finite algorithm certainly instantiates K_64, but does not supply
the desired savings. Increasing n creates more such tiles; it does not increase
the physical 64-bit word width. Granting the word-RAM-MM instruction without a
supported implementation and service cost would change the machine model.

For b_A stored binary coefficient planes and L input limbs, a direct repeated
version has b_A*L times as many arithmetic tile calls. Its favorable
plane-resident-within-tile schedule reads b_A*n^2 bits once, not once per limb;
charging that favorable reuse still does not make any plane disappear. The
query input and all reconstruction limbs, scratch and transfer costs remain.
Precomputing field linear forms trades online arithmetic for stored forms and
form traffic, exactly the distinction already required by the rectangular FMM
audit. It is not a new cold-source theorem.

## 5. A finite packing primitive with visible widths

The primitive below illustrates why a word-RAM exponent is not an instruction
count. It is a complete exact subroutine, not a proposed VORTEX core.

Let A be r-by-t with entries in [0,U_A] and v have t entries in [0,U_v]. Put

    g = ceil(log2(t*U_A*U_v + 1)),
    P = sum_(i=0..r-1,j=0..t-1) A[i,j] * 2^(g*(2*t*i+j)),
    Q = sum_(j=0..t-1) v[j] * 2^(g*(t-1-j)).

The base-2^g digit of P*Q at index 2*t*i+t-1 is exactly (Av)[i]. Each
convolution digit is bounded by t*U_A*U_v < 2^g, so no carries cross digits.
The row ranges [2ti,2ti+2t-2] do not overlap. A sufficient full-product width
is 2*r*t*g bits. Under 2*r*t*g <= h, ordinary h-bit multiplication is enough;
otherwise the full multiword multiplication and packing must be paid.
Degenerate all-zero bounds can be handled directly rather than taking g=0.

Construction: read r*t original entries, shift/insert each into P, store P.
Query: load P; read t inputs; at most two shift/OR operations per input to form
Q; one full-width multiplication; at most two shift/mask operations per output
and r output writes. Initialization, addresses, loops/unrolling, code bytes and
any multiplication/packing spills are additional explicit implementation costs.
This gives a finite sufficient primitive; none of the preparation is free.

For binary operands, packing buys parallel bit arithmetic. It does not omit
the matrix bits embedded in P. For generic residues in F_67 with t=2,
g=ceil(log2(2*66^2+1))=14; the conservative 64-bit full-product condition permits
r=1 and two products. Larger matrices use multiple words/recursive schemes,
whose counts must be provided. This illustrative condition is sufficient, not
an optimal-packing lower bound, and says nothing against other packed layouts.

## 6. Exact dyadic sums are not prescribed native accumulation

A particularly favorable source upgrade would return the exact dyadic value
Wx. Even that is insufficient by itself for the ordered FP32 recurrence.
Take the BF16-exact row W=(1,1,1), initial FP32 +0, sequential RNE accumulation:

    x = (2^24, 1, -2^24)
    exact sum = 1; sequential FP32 output = +0

    x' = (2^24, -2^24, 1)
    exact sum = 1; sequential FP32 output = 1.

At the first positive midpoint, RN32(2^24+1)=2^24. In the second sequence,
the large terms cancel first and the last 1 survives. These remain different
after BF16 output conversion. The same witness applies with separate products
or FMA because every multiplication by 1 is exact. Thus no decoder receiving
only the exact row sum can recover this native result for all legal BF16
vectors. A decoder also using x, W or an order-sensitive summary may do so,
but its algorithm and cost are the missing obligation; this witness does not
rule it out. Actual ATen/tensor-core reduction trees require their own ABI,
not an assumption that this serial stream is their universal schedule.

There is also an explicit finite-width cost to a literal algebraic BF16 lift.
Every finite BF16 value is an integer multiple of 2^-133 and has magnitude
less than 2^128. Therefore multiplying W and x by 2^133 gives signed integers
of magnitude below 2^261. A uniform two's-complement representation uses 262
bits per operand; a sum of 2^14 products has magnitude below 2^536 and fits a
537-bit signed representation. Materializing 262 coefficient bit planes takes
262/16=16.375 times the raw BF16 coefficient bits. This is only a concrete
literal lift's cost, not a universal lower bound: exponent-sensitive and other
representations can be smaller. None turns exact dyadic accumulation into the
prescribed intermediate-rounding result without a separate native-order proof.

## 7. Why previously audited theorems are not escape hatches

[Williams, *Matrix-Vector Multiplication in Sub-Quadratic Time*](https://people.csail.mit.edu/rrw/mat-vec3.pdf)
provides genuine finite-semiring preprocessing. Its direct query-pattern graph
has already been physically charged in the
[finite-semiring audit](../../docs/research/E0_FINITE_SEMIRING_PREPROCESSING_FRONTIER.md).
Invoking it after bit decomposition does not erase any required planes, carries,
rounding stages or the graph's preprocessing/storage.

[Larsen and Williams, *Faster Online Matrix-Vector Multiplication*](https://arxiv.org/pdf/1605.01695)
addresses Boolean-semiring output and gives distinct RAM/cell-probe guarantees.
Boolean nonemptiness is not a signed numerical or native-order output. Free
cell-probe computation does not supply a paid RAM implementation. These are the
existing F-061/F-062 issues, not new rejections.

[RSR-core](https://arxiv.org/html/2603.27462v1) remains the already-audited
binary/ternary pattern-sharing implementation; [low-VC/Pollard MatVec](https://arxiv.org/html/2502.21240v3)
remains the structure-dependent row-difference route. Their required premises
are not supplied for arbitrary original BF16 weights. See the existing
[native-shortcut audit](../../docs/research/E0_NATIVE_EXACT_SHORTCUT_FRONTIER.md)
and [rejection ledger](../../FAILED_APPROACHES_RECENT.md).

The existing EXP-100A catalog tests did not test every future finite-field
packed kernel. Therefore their failure must not be promoted to a theorem
excluding Alman--Yu or all FMM. Conversely, changing to a finite-field kernel
while retaining full cold scans and losing native semantics does not defeat
the recorded exclusion premises or admit an expensive experiment.

## 8. Exact remaining theorem, and stopping condition for this source audit

A useful next theorem must specify an actual checkpoint-derived representation
E(W), finite constructor, causal query program Q and physical schedule such that

    Q(E(W), x, native ABI) = Native(W, x)

at the complete observer cut, with enough native state for all continuations.
It must prove an upper bound for construction, original/transformed storage,
hot residency, dynamic source probes/bytes, address work, arithmetic, native
reconstruction/correction and fallback. A credible >=10x bound must come from
that complete schedule and a stated comparator, before hardware promotion.
At 16,384 width, unspecified asymptotic constants are not such an upper bound.
The full-model baseline/quantile/RNG/KV and <=8-GiB obligations remain separate.

No inspected primary source in this bounded search supplies that theorem.
This does not imply it cannot exist. The direct packed implementation has been
fully classified, so enlarging its arithmetic benchmark or model corpus would
not address the missing source/native obligations and is not recommended.

THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED;
CORE_ADMISSION=false; FULL_MISSION_O1_O6=OPEN.
README was reviewed; it still accurately says the cheap heterogeneous source
is missing. No README/root-ledger change is made by this worker. Parent
integration owns any separately authorized persistence and status updates.
