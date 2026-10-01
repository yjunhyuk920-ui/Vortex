# Query-adaptive nonlinear cells: a finite address-fiber obstruction

## Result first

The existing F-089 interface

    every 25 x 108 binary matrix -> 50 arbitrary 64-bit cells
    every rank-one parity query -> at most two adaptive cell reads

is impossible under its independently legal source/query domain, **even when
all cell contents are nonlinear and globally mix this entire matrix**. In fact,
a self-contained bound below requires **at least five reads**. Five is merely
the first value this necessary condition does not reject, not a construction.

This closes one previously count-feasible finite interface, not arbitrary
checkpoint encodings, global 8 GiB advice, reachable Transformer states, or the
fixed mission. No cheap source or executor is constructed. The lower bound is
worst-case; it supplies no p50/p95 distributional claim.

The proof is an elementary asymmetric-communication/large-fiber argument with
an independently derived rank-one intersection upper bound. The technique is
not claimed as novel. The new project consequence is its application to the
specific nonlinear-word gap left by F-089, beyond the prior polynomial-degree
and linear-zero-transcript tests. No original failed evidence is rewritten.

Full-mission THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED;
CORE_ADMISSION=false; FULL_MISSION_O1_O6=OPEN. This worker's artifact is local;
HANDOFF_STATUS=IN_PROGRESS pending parent integration and remote verification.

## 1. The physically paid hypothesis

Let W be the unchanged finite-word checkpoint, P the pinned reference program,
and z the current legal input/state (including the actual RNG state). A uniform
finite constructor C(W,P) produces stored cells E, initial persistent state H,
layout/code metadata, and a finite query program. For each step the program must
causally produce the required observation and successor representation; an
inductive relation to the reference state must hold under every legal next input.

A cell is a physical bounded-bit record, not an arbitrary real number. The
constructor may use nonlinear functions of any checkpoint coordinates. An
address is computed from current z, already available H, and prior returned
records. The decoder may be nonlinear and branch on returned words. Every
checkpoint-dependent bit in generated code, base addresses, allocation/layout,
indexes, branch tables, caches and previous computations must reside in E or H.
A source-dependent executable or pointer is not free advice.

For the actual hypothesis, charge:

- Original and transformed checkpoint storage separately, in every memory tier
- Construction reads/writes, native arithmetic, code generation and workspaces
- Hot H, live KV/state, metadata, output buffers, scratch, allocator reserve and
  fragmentation against the complete 8 GiB GPU peak when resident there
- Host/SSD/GPU cell reads, writes and whole transactions; a logical 8-byte read
  is not an 8-byte PCIe/SSD transfer when the service uses larger transactions
- Input/state inspection, multiword addresses, decoder arithmetic, synchronization,
  actual guard/certificate failures, repair/fallback and future state forcing
- Initialization/prefill/TTFT and a specified finite amortization horizon

A conservative serial sufficient bound would be construction + initialization
+ prefill + sum over steps of selector, memory service, exact decoding, output/
state maintenance and all recovery work. Claimed overlap needs its own schedule.
A probe lower bound can reject a static interface; it cannot establish that this
complete upper bound meets the same-machine native 4B Q4 quantile targets.

### Favorable static relaxation actually used by the theorem

The source is x in F2^D. The uniform decoder has **one fixed public program**,
receives q independently of x, and returns the parity <q,x>. The encoding has S
fixed-address cells, each w bits; contents are arbitrary functions of all x.
All computation and construction may be free. Only probes are bounded by t.
Thus proving a rejection is stronger than charging the selector's real cost.

The h=0 result has no unprobed checkpoint-dependent initial bits. An extension
below grants one global h-bit H(x), read free. If original checkpoint storage
can also be queried, those addresses must be included in S. Padding a shorter
layout to S cells is allowed; hiding information in its variable length is not.
Writes made during a query are functions of its existing transcript and do not
supply new source information. A persistent state inherited from earlier calls
must be included in H or explicitly conditioned in the legal domain.

For deterministic exact decoders with a fixed worst-case probe cap this model
is precise. It also permits fixing internal coins of an always-exact decoder
whose same cap holds for every coin sequence. It does not prove bounds for
Monte Carlo errors or expected-only stopping times. Reference sampler/RNG state
is an input to a native transition, never a free correctness-error allowance.

## 2. Address-fiber theorem

Let Q be any set of binary linear query masks. Define

    I_Q(d) = max over subspaces V of dimension <=d of |Q intersect V|.

If every (x,q) in F2^D x Q is answered exactly in at most t probes, then

    |Q| <= S^t I_Q(min(D,t*w)).                         (1)

With one common query-independent h-bit H(x), replace t*w by h+t*w.

### Proof

Start with the full query set and source fiber F=F2^D. At any node, all queries
in its class have received the same reply prefix for every source in F.
Partition those queries by their next address; there are at most S classes.
For each address class separately, restrict F to a most frequent value of that
cell. A w-bit cell has at most 2^w values, so the retained fiber has at least
|F|/2^w sources. Give its common reply to that class and recurse.

Pad a query that halts early with ignored reads of a fixed existing cell. After
t rounds the query classes form at most S^t leaves, each with a nonempty source
fiber of size at least 2^(D-t*w). Each assigned query has a fixed answer on its
leaf fiber. If the assigned query span has rank r, these simultaneous linear
answer equations define an affine set with exactly 2^(D-r) elements. The fiber
is a subset of that set, even though the fiber itself need not be affine.
Consequently r<=t*w, and that leaf contains at most I_Q(t*w) masks. Summing the
leaf capacities proves (1).

Sibling leaves may use different source fibers. The proof needs no single
checkpoint that simultaneously realizes every chosen reply. It proves that no
uniform capped decoder works for all source/query pairs, not that all queries
are expensive on one shared checkpoint.

For h-bit advice, first choose a largest H-value fiber. It has at least
2^(D-h) sources and gives one shared initial advice value for **every query**.
Repeat the proof, obtaining r<=h+t*w. Query-dependent free advice is not covered.

More generally, if exactness is known on a Cartesian rectangle X0 x Q0 with
X0 subset F2^D, then each leaf rank is at most

    min(D, floor(D-log2|X0| + h+t*w)).                  (2)

The source entropy deficiency is not silently discarded. For a general legal
relation R subset F2^D x Q, first supply such a rectangle; R need not contain a
large one. Having many reachable queries separately for each x is insufficient.

## 3. Self-contained rank-one intersection bound

Let a<=b, D=a*b, and Q={u tensor v : u!=0, v!=0} over F2. Then

    |Q| = (2^a-1)(2^b-1).

For any subspace V of dimension <=d, define the ruling

    V_u = {v : u tensor v belongs to V}.

It is a linear subspace. Choose a basis u_1,...,u_a greedily: choose u_i outside
the span of its predecessors to maximize d_i=dim V_(u_i). The sequence d_i is
nonincreasing. The subspaces u_i tensor V_(u_i) are in direct sum inside V
because their left factors are independent. Hence sum_i d_i<=d, so

    d_i <= min(b,floor(d/i)).

Every nonzero u whose highest nonzero coordinate in this basis is i was eligible
when u_i was chosen; therefore dim V_u<=d_i. There are 2^(i-1) such u. Binary
nonzero rank-one tensors have unique nonzero factor pairs, giving the bound

    |Q intersect V| <= B(a,b,d)
      = sum_(i=1)^a 2^(i-1) [2^min(b,floor(d/i))-1].      (3)

No exact product-code weight formula, affine-cell assumption, field-valued cell
assumption, or zero-valued transcript is used. Combining (1) and (3), a necessary
condition for arbitrary nonlinear/adaptive encodings is

    (2^a-1)(2^b-1) <= S^t B(a,b,min(a*b,h+t*w)).         (4)

For h=0, the right side at the named point is below the left side for t<=4.

## 4. Exact finite consequence for the previously open point

Parameters: a=25, b=108, D=2700, S=50, w=64, h=0.

    |Q| = 10889035416951477172401260654660528635905

The exact integer certificate is generated by [verify.py](verify.py). The
right-side/left-side ratios, shown only as readable approximations, are:

- t=0: 0, rejected
- t=1: 8.470329728977333e-20, rejected
- t=2: 0.00007450580818969287, rejected
- t=3: 0.003727109398942823, rejected
- t=4: 0.5587936502405623, rejected
- t=5: 37.25290437240644, not excluded by this bound

The strongest rejected integer comparison is

    B(25,108,256) = 973555815717932701149233290412033
    50^4 B       = 6084723848237079382182708065075206250000
                < 10889035416951477172401260654660528635905.

This is an information obstruction even with unlimited preprocessing and free
online computation. No faster address selector can repair the rejected point.

Under **F-089's own historical favorable four-Q4-lane denominator**, five
64-bit reads cost at least 320/(4*2700)=4/135 of that tile's bit payload. That
is 2.5 times its old 8/675 traffic line. This comparison is a scoped historical
interface consequence, not a new universal traffic target, actual native-BF16
traffic measurement, latency ratio, or whole-model lower bound.

## 5. Native finite-word lift, with an honest output boundary

A parity result alone need not lift to native rounded arithmetic. Here a direct
small-integer embedding gives a precise lift for an **exposed MatVec primitive**.
For each binary X in F2^(25 x 108), use dense nonzero BF16 coefficients

    W_ij = 1 + X_ij in {1,2}.

Let the primitive accept every BF16 vector u in {0,1}^108 and return every row
sum using FP32 RNE multiply/add or FMA with +0 start, then BF16 output. Every
intermediate is a nonnegative integer <=216, so all products, reductions and
the final BF16 conversion are exact. Nonfinite values, underflow, signed-zero
ambiguity and rounding-order changes play no role on this restricted family.

Given the returned vector y and any binary r, integer postprocessing gives

    ( sum_i r_i y_i - popcount(r)*popcount(u) ) mod 2
      = <r tensor u, X> mod 2.                          (5)

The lower-bound relaxation grants this postprocessing free. A t-read exact
MatVec replacement at the same encoded-cell budget would therefore give a
t-read rank-one oracle; it is rejected for t<=4. The MatVec algorithm need not
receive r at all, so giving a richer (r,u)-dependent address selector only makes
the impossibility statement stronger.

The native source payload is 5400 BF16 bytes. Losslessly representing this
special source family by 2700 binary choices does not modify its native weights.
But if the original BF16 file remains query-accessible, its 675 extra 64-bit
addresses also count: S=725, not 50. The same formula then rejects t<=2, while
it does not reject t=3; these are distinct budgets. If the original is retained
only as an inaccessible archival copy during the query, its storage/build cost
still counts physically, but it does not add usable online addresses.

This is a symbolic native reduction, not a run of an HF model or a claim that
all kernels implement this reference primitive. It does **not** establish that
these 2^2700 checkpoints are publicly released HF models, that their binary
activations are reachable in a released model, or that a whole-program executor
must expose this hidden MatVec. A whole-program exact output/state algorithm may
avoid hidden row materialization. Those are real limitations, not details to be
filled in by calling the parity source a Transformer theorem.

The primitive consumes no RNG and has no state changes except writing its
output. A complete HF replacement still needs the full initial-state relation,
KV/layout/cache/control and RNG-consumption induction. This reduction does not
supply them or a tail-latency population.

## 6. Small exact witnesses for the nonclosed boundaries

### 6.1 Source-dependent reachability can defeat an unqualified application

One binary source bit x with only the legal query q=x has target x*q=x. Return
q without reading the source. Both query values occur across sources, yet the
legal pairs are {(0,0),(1,1)}, not F2 x F2. Native version: coefficient W in
{1,2}, legal current input u=W, output W*u=u*u in {1,4}; compute from u alone.
It uses one multiplication, output write, no query-time weight read, and an
input that already contains the coefficient. Producing that state initially
requires the corresponding source read or supplied input; nothing is free.
This is the smallest nontrivial source/query family and a counterexample to
applying (1) to arbitrary checkpoint-dependent reachability.

### 6.2 Free common advice must be charged, and cannot be query-dependent

For one source bit, H(x)=x permits the independently legal queries {0,1} with
zero cold reads. It uses one stored hot bit, one initialization source read,
and trivial Boolean decoding. Equation (4) correctly permits it with h=1.
With unrestricted query-dependent advice, even one free bit
H(q,x)=<q,x> answers any parity query; obtaining that bit is the original source
problem. Thus neither query-dependent preprocessing nor source-dependent code
can be smuggled into the h=0 theorem.

Global 8 GiB advice contains 68,719,476,736 bits, already much greater than this
2700-bit block. Giving all of it to the theorem makes the block bound vacuous.
Dividing it by matrices without a proved locality/direct-sum premise is exactly
the invalid shortcut in F-063; it is not done here.

A finite advice boundary can nevertheless be stated **for the named isolated
interface**: at t=2, equation (4) still rejects h=1383 free common bits (ratio
0.9203752379051519), while h=1384 is its first nonexcluded value (ratio
1.2255513198657284). Thus this theorem requires at least 1384 such bits for a
two-read proposal with S=50; it does not construct a decoder at 1384 bits.
Monotonicity of B in its dimension makes the h=1383 integer rejection cover
every smaller h. This is not a per-matrix allocation of the global hot budget.

### 6.3 A smallest genuinely nonlinear packed-cell constructor exists

For three source bits (x,y,z), store the three-bit word

    E(x,y,z) = (x, y, z XOR (x AND y)).                  (6)

One fixed-address read recovers x,y and then z=E_3 XOR(E_1 AND E_2). Every
binary linear query is now answered by at most three ANDs and two XORs, plus
one AND and one XOR for inversion (bit unpacking/address/output work extra).
This is a fully specified nonlinear, globally mixed, full-rate encoding and
cheap exact decoder. It satisfies (1): w=3,t=1 already permits rank 3.

Three input bits are minimal for a **nonlinear bijective full-rate Boolean
encoding**: all 1-bit permutations are affine, and on two bits the affine
group has 4*6=24 elements, exactly all 4! permutations. This minimality is
only for that stated class, not for all nonlinear redundant encodings.

For the corresponding dense native coefficients 1+x,1+y,1+z and binary u,
decode the three bits and compute their integer dot, always <=6. Every native
BF16/FP32 step is exact. Physical storage is at least one byte (or one 64-bit
word in the named model), compared with six original BF16 bytes, plus the
retained original and constructor read/write costs. The 64-bit query read is
eight bytes and therefore worse than reading those six BF16 bytes. Preparation
reads the three coefficients, checks membership in {1,2}, computes one AND and
one XOR, and writes the code word; the general checkpoint case needs exact
fallback. There is no asymptotic saving or target admission here.

This witness falsifies a claim that nonlinear mixed cells themselves are
impossible. It does not evade the proved 25x108 storage/read bound. An exact
8-source by 8-query integer truth check validates only equation (6) and its
small-integer lift, not any scalar-performance or model claim.

## 7. What specifically remains unknown

- A near-source-size global nonlinear checkpoint encoding with a paid causal
  selector, native exact decoder and target-fitting complete resource bound
- A valid all-model source/query reduction for released HF checkpoints and
  their reachable state relation, preserving the actual observation interface
- An adequate lower bound that jointly handles the complete global hot advice,
  external storage and source-dependent prior state without dividing advice
- Any complete native state/RNG lift, source construction, VRAM/TTFT or
  same-machine 4B latency upper bound for this hypothesis

The present bound resolves no full-mission O1-O6 obligation. It sharpens O4/O6's
restricted interface record and removes one concrete finite constructor target.
It is not useful to launch a larger SMT search or another scalar benchmark on
the now-rejected point. A surviving proposal must supply genuinely different
budgets, reachability information or a complete global constructor, then charge
the resulting preparation/state/address/native work.

The next justified constructive question is therefore specific: can one paid
global encoding and causally maintained hot state provide the source information
that an isolated low-probe cell cannot, while avoiding the eliminated dense
work in both initialization amortization and subsequent hot-state updates? An
answer must provide E(W), the initial-state constructor, the current-input
address/decoder program and an exact state-update recurrence, with actual bit
lengths and native costs. It must identify whether it uses globally shared
cells, common static advice or a checkpoint-dependent reachable-state relation;
these are distinct hypotheses. The two-read isolated case now has a quantitative
common-advice necessity. Neither increasing the target to five reads nor
asserting reachability supplies that construction, and no larger toy search or
backend run follows automatically from this exclusion.

## 8. Prior evidence, sources and verification

Prior interfaces inspected include F-059 through F-064, F-073, F-076 through
F-078, and F-087 through F-089, especially:

- [F-089 nonlinear probe-degree bound](../../docs/research/E0_ADAPTIVE_NONLINEAR_PROBE_DEGREE_GATE.md)
- [F-088 packed linear word bound](../../docs/research/E0_ADAPTIVE_PACKED_LINEAR_WORD_GATE.md)
- [Global advice synergy boundary](../../docs/research/E0_GLOBAL_ADVICE_SYNERGY_FRONTIER.md)
- [Prior communication/probe barriers](../../docs/research/E0_GENERAL_FINITE_WORD_RANK_ONE_PROBE_BARRIER.md)

The standard cell-probe-to-asymmetric-communication setting and its limitation
are described in Wang and Yin, [Certificates in Data Structures, §§1 and 3](https://arxiv.org/pdf/1404.5743).
That source was read directly. Equations (1)-(5) are self-contained derivations
here; no theorem about BF16 or Transformer reachability is attributed to it.
The source also explains why communication techniques alone should not be
advertised as a target-scale general lower-bound breakthrough.

Two independent mathematical reviewers checked the modal-tree/rank proof and
its assumptions; both checked the greedy-rulings derivation, and separately
confirmed the t=4 integer rejection. This is manual mathematical review, not
formal proof-assistant verification. The old exact Schaathun/product-code
intersection formula was cross-checked by a reviewer but is not a dependency
of this report or its certificate.

The [plan](PLAN.md) discloses that analytical exploration preceded registration.
[verify.py](verify.py) uses only exact integers and fractions for rejection;
decimal ratios are descriptive. [certificate.json](certificate.json) and
[validation.json](validation.json) record the targeted results and document
checks. No model/checkpoint download, tensor execution, CPU/GPU latency run,
full repository suite, or native kernel test is performed.

Provenance at start: origin https://github.com/yjunhyuk920-ui/Vortex.git,
branch research/cloud-continuation-20260930, HEAD
da2fa6c46163e89b5349395c0916b6bd26ead67e. Remote branch/PR verification and root
ledger integration are owned by the parent. Existing dirty replay evidence was
left untouched. README was reviewed: its no-admitted-core/no-target-success
statements remain true; the newly closed finite frontier should be linked by
the parent in its integration update. No remote write was made by this worker.
