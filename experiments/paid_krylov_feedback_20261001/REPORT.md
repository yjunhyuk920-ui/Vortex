# A paid global source from reachable Krylov state and nonlinear feedback

## Result first

A concrete, non-enumerative source **does** eliminate repeated dense source work
for the exact finite native primitive specified below. It handles arbitrary
binary A,b,c1,c2,co, not a supplied inverse-paired chart. The original transition
has an active nonlinear gate. A paid compiler derives its reachable basis from
the original source and emits a 52-byte hot record; the causal runtime never
reads A and maintains an exact original-state continuation relation.

This is a **bounded synthetic construction**, not a released Transformer result
or a generic BF16 MatVec algorithm. Its original native program explicitly
reduces small integer dots to parity. That algebraic closure and the small
nonlinear feedback/observer boundary are essential. Ordinary HF rounding,
normalization, gates, attention, and full KV behavior have not been transformed
into this family. No target-cost, GPU, latency or new core claim follows.

The useful advance over the previous chart controls is specific: the chart is
computed by polynomial linear algebra on arbitrary supplied A; it need not
appear syntactically in the original program and never enumerates states. The
remaining cross-layer/temporal relation is also specific: a cheap exact
innovation/observer source in the transported coordinates. Small reachable
state dimension or small innovation rank alone does not supply it.

THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED;
CORE_ADMISSION=false; FULL_MISSION_O1_O6=OPEN; HANDOFF_STATUS=IN_PROGRESS.

## 1. The original observer and native state contract

All vector algebra in this section is over F2. For arbitrary A in F2^(n x n),
b,c1,c2,co in F2^n, 1<=n<=64, and a legal external input bit a, define

    g(s) = (c1^T s) AND (c2^T s)
    s_next = A s XOR b*(a XOR g(s))
    y = co^T s_next.

The original state starts at s=0, with any supplied uint32 RNG state. Its output
is the pair of BF16 words [y,1-y], plus a sampled token. It advances the original
RNG exactly once by

    R_next = (1664525*R + 1013904223) mod 2^32,
    token = [R_next < (0x40000000 if y=0 else 0xc0000000)].

This is the ORIGINAL declared sampler, not an HF softmax/categorical sampler.
The ABI exposes these outputs and permits future external bits or feedback of
the previous sampled token. The full logical state is (s,R). A readback operation
returns that state; its implementation and cost are explicit below. No hidden
logits, FP flags, aliased tensors, external effects or shape controls are omitted
from an otherwise larger program: this small program is the entire test ABI.

### The literal native implementation

The unchanged original matrix is stored as NONZERO BF16 W_ij=1+A_ij. Each row
uses sequential FP32 separate multiplication/addition from +0, BF16 store, then
integer subtract-and-parity:

    parity_i = (BF16(sum_j W_ij * s_j) - popcount(s)) mod 2.

The three observer rows are stored as 1+c_j and use the same procedure. Every
product and partial sum is a nonnegative integer <=2n<=128. Every such value is
exact in FP32 and BF16, so prescribed reduction order, BF16 storage and the
parity reconstruction are exact. The b injection and logical state are binary.
The C reference actually executes these floating operations and asserts that
its BF16 store changes no accumulator. Compiler flag -ffp-contract=off pins the
separate-operation reference. FMA would also be exact on this restricted domain,
but no other numerical ABI or arbitrary BF16 weights are asserted.

The source file has an eight-byte header and 2(n^2+4n) bytes of native arrays.
These source values are generated synthetic inputs, not HF checkpoint weights.

## 2. Finite automatic source construction

Read and validate all original W,b,c1,c2,co values, preserving the source file.
Decode A by W-1 and pack binary columns/rows into bounded words. Starting with
v=b, repeatedly form v=A^j b. Maintain an incremental exact elimination basis:

1. Reduce v against the stored pivot vectors, carrying each pivot's coordinates
   in the original, already selected Krylov columns
2. If its remainder is nonzero, append v as the next column and store that
   remainder with its coordinate label; then compute the next A*v
3. At the first zero remainder, stop. If r columns were appended, retain their
   unique dependency coefficients f such that A^r b=K f

Here K=[b,Ab,...,A^(r-1)b], with independent columns, and 0<=r<=n. This is at most
n+1 reductions and n dense packed matrix-vector multiplications. Zero b gives
r=0 immediately. No assumption of full rank, a cyclic A, a supplied chart,
training or favorable random retries is needed.

Define the r-dimensional companion map

    J z = ((z << 1) masked to r bits) XOR f*z_(r-1).

For r>0, directly from the constructed columns,

    A K = K J,                 b = K e0.

Transport all three original observer rows once:

    d1=c1^T K, d2=c2^T K, do=co^T K.

The serializer writes n,r,f,d1,d2,do,mask to a fixed 52-byte hot record. A separate
12+8r-byte decoder record stores K's columns. Original weights persist separately.
Every checkpoint-dependent bit in this runtime is in these paid records; the
program and field convention are fixed. There is no query-dependent advice,
variable-address answer table or encoded source hidden in generated code.

The implementation also directly rechecks AK=KJ, b=Ke0 and observer identities.
Those dense checks are reported separately and charged, rather than described
as free verification.

## 3. Causal hot-state update and exact continuation proof

Initialize z=0 and copy the supplied RNG. A step uses only the hot record, current
z,R and the actual current a:

    p1=parity(d1 AND z); p2=parity(d2 AND z)
    z_next=J z XOR e0*(a XOR (p1 AND p2))
    y=parity(do AND z_next)
    execute the exact original RNG draw, threshold and output writes.

For r=0 retain z=0, y=0 and still execute the original RNG/output procedure;
the input cannot change original s because b=0.

Let Gamma(z,R)=(Kz,R). The invariant holds initially. If s=Kz, then

    A s XOR b*(a XOR g(s))
      = K[Jz XOR e0*(a XOR ((d1*z) AND (d2*z)))].

Thus Gamma maps the new encoded state to the exact original successor. The
transported do gives exactly the original y. Logit words, ordered RNG draw count,
RNG word and token then agree. Induction covers every legal external bit sequence
and every original RNG seed, including a policy using each executor's own prior
tokens. No future output or precomputed input sequence enters the update.

This is exact continuation, not merely sample agreement or lossy minimization:
K is injective on encoded states and actually reconstructs the whole original
logical state. If external input bits are unconstrained, every Krylov-reachable
linear state is still reachable despite the nonlinear gate: choose a equal to
the desired effective control bit XOR g(s). Runtime needs no such choice or
reachability selector; it is just a proof that the nonlinear port has not
artificially shrunk the full externally controllable state set.

### Readback and imports are different operations

Readback computes Kz by visiting the r stored basis columns and XORing the
selected ones, then returns R. For n<=64 that is r eight-byte column reads and
O(r) word work, not a free observation. The recorded tests perform readback for
verification after every step, separately from the source-only hot query.
For general n, decoder traffic is r*ceil(n/64) words.

The allowed initializer is s=0. An arbitrary externally supplied original state
would require a paid solve Kz=s, with a domain test. States outside im K are not
supported by this initializer. There is no claim that all HF cache imports can
be represented by this r-dimensional construction.

## 4. Complete paid inventory and the actual saving mechanism

For general F2 algebra, an elementary scalar implementation uses O(r*n^2)
operations for the powers, O(n*r^2) for elimination, and O(n*r) to transport the
three observers, plus source reads, writes, allocation and direct verification.
At r<=n this is polynomial O(n^3), not full-state enumeration. When n<=64, each
vector fits a 64-bit word, and the provided implementation uses packed AND,
popcount, XOR and shifts. This changes the unit, not the underlying source
accounting; Python object, hashing/dispatch and allocation costs are not declared
one-cycle machine instructions.

The machine schedule is explicit and serial: load/validate source, build packed
A, construct K and f, transport observers, run direct checks, write/reload the
records, initialize state, and execute sequential query steps. No overlap or
free preprocessing is assumed. File-system physical I/O service, code loading,
process/runtime overhead and integrity policies add their actual costs to the
inventories below. These are not measured memory-service or latency bounds.
[COST_MODEL.md](COST_MODEL.md) supplies a separate conservative fixed-array
word-instruction and data-buffer upper bound, including verification and
readback. It does not price Python or external services as one-cycle operations.

### Retained storage and workspace

- Original source: 8+2(n^2+4n) bytes, kept as an archival source
- Hot encoding: 52 serialized bytes, rounded/aligned allocation extra
- Mutable logical state: r bits plus uint32 RNG, stored as 12 serialized bytes
- Decode encoding: 12+8r bytes for n<=64; cold unless readback is requested
- Constructor: original input buffer, packed A (8n bytes), K (8r bytes), pivot
  vectors and coordinate labels (at most 16n bytes), validity/pivot indexes,
  loop state and serialized output buffers. The Python implementation retains
  objects as well; these payload counts are not process RSS or GPU allocations
- Query: the fixed hot record and state, bounded word scratch, four output bytes
  for the two BF16 logits and a token. Trace/logging and independent verification
  are separate actual test costs, never assigned to a fictitious free tier

### Online work for the supported primitive

For r>0, the displayed recurrence uses three 64-bit parities, a shift/mask,
conditional f XOR, input/feedback XORs, the one nonlinear AND, and the original
RNG/threshold/logit writes. A direct fixed-word implementation needs at most
32 arithmetic/bit/select operations for these displayed operations (including
popcount as a word primitive); dispatch, loads/stores and physical service are
additional. It reads no original A or c rows. No 64-bit unit-cost claim is made
for arbitrary-length vectors or big integers.

The literal original native body executes n^2+3n floating products and the same
number of additions per step, plus bit extraction, BF16 conversion, parity,
injection, original RNG and output work. A much better simple packed-F2 baseline
would use n dense row-parity probes for A plus the three observer probes and
control work. That favorable baseline must not be confused with native dense
floating execution: this source family is specially losslessly bit-compressible.
The Krylov recurrence removes even those n per-step A-row probes, but a latency
ratio is not derived by equating float operations with word operations.

For a finite T-step session, total time is exactly accounted as

    construction + load + initialization
      + sum_t(query arithmetic + hot/state service + output/RNG)
      + sum_readbacks(decoder service + reconstruction)
      + any allowed input-state solve, logging or verification.

Cold/short-session TTFT includes the entire constructor unless the actual reuse
policy pays it elsewhere. The asymptotic field-operation mechanism is
O(n^3)+T*O(n) versus T*O(n^2) in a scalar field model, so a finite sufficiently
long T can amortize construction. The ratio is not automatically good at a
specified short session, and it is not a same-machine 4B latency comparison.
There is no unbounded/amortization-free precomputation claim.

### Actual unfiltered random 64-bit record

n=64,r=63: original source 8,712 bytes; hot 52 bytes; decoder 516 bytes.
Construction actually reads all 4,352 BF16 source values, performs 63 packed
matrix calls / 4,032 A-row reads, 4,096 pivot-slot checks, 1,028 pivot-pair reads,
2,056 pivot XORs, and 189 observer-column reads. It additionally performs 4,032
A-row reads in direct matrix-identity verification. Packing, branches, output
writes, decoder checks, serialization, Python overhead and I/O are additional;
these event counts are deliberately not mislabeled a complete operation total.

Each hot query uses zero original-matrix reads. A requested raw-state readback
uses 63 basis words (504 payload bytes). Reading back every token therefore
removes the small observer boundary advantage against the favorable packed
matrix baseline; it cannot be silently omitted from an ABI that requires it.
No elapsed-time, peak-RSS/GPU or 405B projection was measured.

## 5. What relation would genuinely extend this construction?

A useful broader source would provide, on an invariant reachable domain, an
exact finite-word decomposition of the ORIGINAL whole transition of the form

    s_next = A s + B phi(C s,a),
    observation = psi(D s,a,phi(C s,a)),

over a specified bounded-word algebra, together with a paid constructor and
small evaluable phi/psi. It must include all state, original RNG/effects and
required readbacks. The symbols here are a falsifiable structural target, not
a new assumed oracle. The implemented family has one B column, two feedback
rows and one original observer row; phi is one AND and an input XOR.

For k input/innovation columns, an explicit multi-chain reachable basis can be
constructed by extending each new B column until dependence on the previously
closed chains. There are at most k chain ends. The transported A has shifts
inside chains and at most k dense feedback columns. It costs O(k*r) scalar field
work plus observer/innovation work, instead of r^2, but only when all transported
innovation features and original observations are actually cheap. This extension
is derived here, not implemented or measured. Narrow rank alone is insufficient.

Three indispensable requirements are now separated:

1. **Native law:** the field/word identities must hold for the prescribed original
   native instructions on every legal reachable state. Our exact small-integer
   parity circuit supplies that law only for its own ABI. A plain BF16 sum already
   breaks exact integer additivity: BF16(256+1)=256, not 257. Rounding/gates cannot
   be silently commuted through a real/field basis change
2. **Paid innovation source:** phi(CKz,a) and the required observers must be
   obtainable before the dense work they eliminate, with sufficiently small
   transported feature payload, evaluation and write cost. An arbitrary lookup
   or original full forward used for phi returns the missing dense computation
3. **State boundary:** the allowed initial-state constructor, KV/control/RNG
   evolution and every required readback must stay within the proved relation.
   Taking only a sampled token as observer is invalid when logits or raw state
   are required; many independent observed rows restore dense transport cost

A full append-only cache illustrates why mere low innovation rank is weak:
writing only a new cache block makes the next-state difference low-dimensional
relative to the entire accumulated cache, but computing that block can still
be the complete original dense forward. This construction does not evade that
cost simply by calling the new block an innovation.

### A cheap falsifier for one proposed linear-coordinate feedback representation

For a fixed F2 state representation and a proposed rank-k nonlinear port B,
all second additive differences of the transition must lie in im B:

    F(s) XOR F(s+u) XOR F(s+v) XOR F(s+u+v) in im B.

The A term cancels. Therefore k+1 independent such output vectors, with all four
states legal in the same declared domain, refute that particular rank-k port.
An invertible LINEAR change of state preserves their span dimension. Arbitrary
nonlinear coordinates do not, and independently sampled legal states need not
form a legal quadruple. This is a scoped falsifier for this source family, not
a Transformer lower bound or a substitute for the constructive source above.
No new native-HF or large synthetic test is authorized merely by this criterion.

## 6. Evidence and relevance to the address-fiber result

The four unfiltered registered sources give Krylov ranks 29,63,1,0. Independent
native C and serialized-source candidate agree on all 4,096 causal steps, including
original decoded state, logit bits, token, actual input and RNG. The three nonzero
cases detect a corrupted feedback coefficient at steps 29,63,1 respectively.
The zero-input case is correctly inert rather than forced to produce a witness.

On random64, the nonlinear gate is one on 130/512 external-input steps and
117/512 own-output steps. Removing that gate changes 510/512 and 507/512 states
on those respective same-input trajectories. It is genuinely active feedback;
this result is still synthetic and says nothing about trained models.

This source does not contradict the nonlinear-cell address-fiber theorem. Its
runtime query is an actual input bit plus a causally maintained z, not an
independently supplied arbitrary matrix/query pair. The paid checkpoint-dependent
basis establishes a reachable-state relation and transports the fixed observers.
It answers no arbitrary original MatVec query. Asking for an arbitrary new state,
new observer or raw state materialization incurs the source/solve/decode costs.
The original BF16 source also remains stored, with its full construction charge.

Prior nonlinear coordinates recognized explicit inverse-paired reversible IR or
enumerated states. Prior boundary convolution assumed a structured original
matrix. Here A is arbitrary within the stated binary family, and temporal reuse
plus low-dimensional nonlinear feedback supplies the exact source. This remains
established control/Krylov algebra, not a novelty claim or generic fast simulation
of arbitrary nonlinear programs.

## 7. Validation, source and next obligation

[PLAN.md](PLAN.md) was saved before numerical results. [verify.py](verify.py)
contains the compiler and candidate; [native_reference.c](native_reference.c)
is the separate floating reference. [results/summary.json](results/summary.json)
and per-step traces preserve the evidence. Generated binary sources/hot records
and executable are reproducible and excluded from publication; no upstream
checkpoint or raw replay archive is uploaded.

Primary background: Emilio Frazzoli's MIT 6.241J
[Lecture 20, canonical forms](https://ocw.mit.edu/courses/6-241j-dynamic-systems-and-control-spring-2011/43a02bc538e449c1e5eae487139b5030_MIT6_241JS11_lec20.pdf),
page 7, describes the reachable similarity/companion construction. The discrete
finite-field nonlinear feedback, exact native primitive, paid source/state/observer
boundary and tests here are self-contained adaptations, not claims attributed
to that continuous-time lecture.

An independent reviewer checked the algebra and reproduced the 32 scientific
files, then added small exhaustive and rank-64/native-boundary checks. Its
[original audit](INDEPENDENT_REVIEW.md), [receipt](independent_review.json),
[portable script](review_independent.py) and [counter correction history](REVISION_LOG.md)
are preserved. This is independent manual/programmatic review, not a proof
assistant or target-hardware verification.

Full-mission O1-O6 stay OPEN. Bounded O2/O3 receive an actual state/observer
constructor and exact induction; bounded O4 receives explicit paid schedules
and records. Neither generic HF detection/lifting nor target O5 is established.

The next worthwhile work is to find an actual original-native cheap innovation
and transported-observer relation, or a materially different source principle.
Do not grow the random matrix dimensions, repeat parity controls, replace parity
by a claimed HF operation, or substitute a cheap state label for its missing
native update. No larger backend/model/GPU experiment follows from this pass.

Local start: da2fa6c46163e89b5349395c0916b6bd26ead67e on
research/cloud-continuation-20260930. Parent subsequently verified remote
publication dbbe6cb3bcccca57b0454f21b91a84c692bd97e6 while local HEAD and dirty replay
were deliberately left intact. Root integration/publication is parent-owned;
this worker makes no remote write or root-ledger edit. README was reviewed;
its no-admitted-core/no-target-success statements remain accurate, and a new
bounded result link should be added only by the integrating writer.
