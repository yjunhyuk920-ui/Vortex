# Causal cost construction, 2026-09-30

This is an isolated working artifact, not a new admitted core or publication.
Current inspected local head: e5faad8a594e822d8fc13f0ec3a76c0ddee7d750,
research/cloud-continuation-20260930. Other workers' modifications are untouched.

## Intended theorem and scope

For every unmodified public HF dense checkpoint W in the fixed mission, every
legal input/context/RNG state and supported pinned native ABI, construct a finite
causal executor preserving exposed outputs and an inductive successor-state
correspondence. The whole paid schedule must fit one GPU's total 8 GiB peak and
meet same-machine native 4B-Q4 p50 <=1.2x / p95 <=1.5x, with unchanged TTFT.
No hypothetical future vectors, free decoding, semantic weight changes,
unlimited preparation, or uncharged repair are permitted.

O1 construction, O2 causal execution, O3 native/RNG/state correctness, O4 full
resource bounds, O5 target budgets and O6 independent artifacts all remain OPEN.
The obligation under active construction is O2/O4: a source or schedule that
reduces dominant arithmetic AND memory work, before backend/model runs.

## Work controls

No numerical experiment is preregistered or admitted here. No tests, model/GPU
runs, root-ledger writes, commits or publication will be used as primary progress.
Initial symbolic exploration and reading preceded this note. Later reference
experiments, if justified by a new target-linked lemma, need a separate dated
preregistration before their results. Current runtime/hardware status: NOT TESTED.

## Important negative-evidence scopes

- Whole-program table-width bound concerns full-domain literal message allocation
  on an unsimplified word graph, not every semantic native relation encoding
- The dirty-trace and suffix witnesses are legal scalar subprograms, not a proof
  that those vectors occur at every released checkpoint's reachable state
- Direct packed MatVec and ordered-grammar constructions retain their specific
  complete source scans; they are not universal cold-probe lower bounds
- These logical gaps do not supply a constructor, decoder, causal invariant or
  affordable query schedule. Calling one an opening is not progress toward cost
  closure without its executable finite procedure

## Supplemental native algebraic-lift guard

This conditional guard was constructed while seeking a genuinely cheaper source.
It is auxiliary; it does not supply that source.

ABI: finite BF16 coefficients and inputs, FP32 fused multiply-add, RNE with
 gradual underflow, seed +0, no observed FP status flags, no nonfinite values.
For each row i, preprocessing reads every coefficient and stores
A_i=sum_j |w_ij| exactly and v_i=min_{w_ij !=0} v2(w_ij). One current-vector scan
computes X=max_j |x_j| and v_x=min_{x_j !=0} v2(x_j).

Let e_i=v_i+v_x. Accept the row only if

  -149 <= e_i <= 104
  A_i X <= (2^24-1) 2^e_i.

Every exact prefix is a multiple k 2^e_i with |k| <=2^24-1. Every such word is
exactly FP32-representable in the accepted exponent range. Induction therefore
shows no FMA rounds; an independently supplied exact algebraic sum yields the
same native word. Exact cancellation under RNE gives +0. With a +0 seed, a row or
vector that is entirely zero likewise yields +0, including signed-zero products.
Nonfinite operands, other seeds/ABIs/flags take the original paid path.

At N<=16384, A_i is an integer <2^275 in units 2^-133; five 64-bit limbs suffice.
The BF16 maximum has <=8 significand bits. A guarded record can reserve 48 bytes
for the norm, valuation and flags. The query guard needs a bounded five-limb
small-integer multiplication plus exponent-aware comparison per row, a shared
N-word vector scan, all record/address traffic, and ordinary output dispatch.
The constructor's entire coefficient scan, decoding/additions and record writes
count, as does every failed guard and original fallback.

This is not a generic native lift: take N=16384 and all w_j=x_j=255/128. Here
e=-14, A_i X/2^e=16384*65025=1065369600, so the guard rejects. Prefix 259 has
scaled numerator 16841475, an odd integer above 2^24; it must genuinely round.
This scalar family is not claimed to be reached by a specific HF checkpoint.

No independently cheap non-full-sweep exact algebraic source has been derived
that composes with the guard while escaping the recorded pattern-table, static
DAG, fixed-linear-code and packed full-scan costs. The guard must not be reported
as a dominant-cost reduction, core admission, or whole-mission progress.

## Three concrete mechanisms compared; none qualifies as a new core

These are materially different computations, but the source screens below show
why they are not three newly qualifying principles. A new label does not reopen
the already recorded evidence. No expensive implementation follows.

1. Native event execution: preserve the old native trace or current accumulator,
   construct an original-order index, and execute only changed/nonabsorbed FMA
   events. A valid omitted event needs a proof that its entire native word effect
   is unchanged. If K events survive and E is event-discovery/index/guard work,
   arithmetic and coefficient traffic can fall by tenfold only if both K/P and
   (event metadata + source bytes)/original bytes fit 0.1, with E and failed
   discovery included. The existing generous absorption ceiling is at most
   1.2556% removable on its frozen real projection; exact temporal/product reuse
   also misses the required fraction. Thus that particular source is stopped,
   without rerunning it. A new current-input index alone cannot invent additional
   native no-ops. The fully dirty scalar witness is the hardest all-input case;
   actual checkpoint reachability remains an unproved separate restriction

2. Algebraic evaluation behind the exact grid guard above: the actual information
   source would be an independently constructed static arithmetic data structure.
   The guard proves the native-order mapping only on accepted rows. A concrete
   fallback implementation decodes every original BF16 coefficient, multiplies
   exact significands, accumulates at the common dyadic scale, then encodes FP32.
   That is finite and causal, but still reads 2P bytes and does P products, plus
   the guards. Current packed/grammar/fixed-linear constructors do not supply a
   cheaper cold source; no 10x claim is made. The all-255/128 family above forces
   guard failure and original full work, even before a native/HF coverage proof

3. Causal weight-stationary scheduling: retain a fixed set R of original weights,
   tile every remaining projection, prefetch its next tile into a second buffer,
   execute the original native kernel, preserve its exact successor state, and
   advance in the original causal layer/token order. The finite procedure is
   fully specified once the original kernel schedule and buffer capacity are
   supplied. Input/output, KV, allocator, buffers, synchronization and repeated
   weight transfers count. Independent preparation/current-token bit columns can
   improve utilization, but they do not add future token activations. Keeping a
   cold tile across several token evaluations requires those inputs to exist or
   a separately paid exact substitute. Ordinary autoregressive causality supplies
   neither. Consequently this source-preserving schedule retains essentially all
   dominant work and is stopped as a standalone route

For the third schedule, the recorded parameter inventory is P=405849243648. Its
original two-byte payload is B=811698487296 bytes =755.953125 GiB. Even granting
all 8 GiB to useful permanent weights and no KV/workspace, a complete native
weight scan transfers B-8589934592=803108552704 bytes per warm token after the
first. That is 98.941733325% of the payload, not <=10%. This is a bound for this
complete-native-scan schedule, NOT every executor or compressed/cold structure.
Any overlap changes elapsed-time composition only; it does not remove these
bytes or products. Offloading the complete arithmetic to CPU trades PCIe for
CPU-memory traffic and CPU arithmetic; both remain charged, with no new measured
service-rate or source-saving premise. No large hypothetical hardware is granted.

If the same tile is actually usable for K causally available target vectors,
this schedule can share its read across K uses, paying K native arithmetic
uses. That yields a concrete traffic-only amortization condition, not a source
of K vectors and not a compute reduction. Expanding branches or predicting them
must pay the branching/draft computation and exact repair; the existing causal
block failures cannot be hidden by the weight-stationary name.

## Scientific blocker and unblocking fact

No genuinely new procedure reducing BOTH dominant coefficient information access
and calculation by >=10x was derived in this assignment. The auxiliary grid guard
supplies one conditional native mapping, while the executed constructions above
retain dense source work. This is not completion of the research or a universal
impossibility theorem. There is no qualified backend/model experiment to run.

The next fact must be algorithmic, not a new test count: a finite checkpoint-
derived representation with explicit current-state address/decoder operations
that produces the required native observer/state cut before the original dense
region, together with an actual paid size/query/repair bound; or a proved causal
schedule that makes enough already-correct target computation reusable while
also reducing arithmetic. A reachability claim can help only if its invariant
is constructed, maintained and checked cheaply without executing the region it
is meant to remove. The grid guard can then discharge part of native lifting
on its accepted domain; it cannot be used to assume the missing source/coverage.

No full-mission O1-O6 obligation closes here. THEORY_STATUS=NOT_ESTABLISHED;
HARDWARE_STATUS=NOT_TESTED; CORE_ADMISSION=false; HANDOFF_STATUS=IN_PROGRESS.
