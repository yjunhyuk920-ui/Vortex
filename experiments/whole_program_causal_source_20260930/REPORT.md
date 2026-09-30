# Whole-program causal execution: an exact constructor, but no cheap source

## Result first

**No qualifying >=10x whole-program source was constructed.** Three genuinely
different execution principles were examined: continuation-state minimization,
exact relational elimination across the complete native transition, and exact
incremental trace repair. None currently has a sufficient paid target bound.
They must not be renamed as three newly admitted cores.

The strongest bounded construction is a finite **native-relation elimination
compiler**. It can compile directly to tables producing exposed logits and
required successor state, without reconstructing hidden matrix accumulators at
query time. It preserves each original native instruction as a Boolean relation,
so it does not incorrectly treat rounded arithmetic as a semiring. Its explicit
constructor, however, enumerates joint boundary assignments. Literal tables have
an overwhelming static-width obstruction, while conditioning on current inputs
and forward-substituting retains the original dense source traversal.

This isolates a sharper requirement for this particular whole-program route:
a **constructible, exactly evaluable representation of joint native separator
relations**, including correlations between operands and the causal reachable
state domain. Scalar-map compactness, few output words, and a small state label
do not prove that requirement. No such representation with the needed paid bound
is supplied here. This is an identified scientific blocker, not a new universal
impossibility theorem or a claim that every dense operator must be evaluated.

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
THREE_QUALIFYING_NEW_PRINCIPLES=false
FULL_MISSION_O1_O6=OPEN
HANDOFF_STATUS=IN_PROGRESS

## 1. Fixed theorem, scope, and observation boundary

The final theorem remains [the fixed mission](../../MISSION_AND_WORKING_PRINCIPLES.md)
and [O1-O6](../../docs/CONSTRUCTIVE_THEORY_CONTRACT.md), as recorded in the
[theory plan](RESEARCH_PLAN.md): arbitrary public unmodified HF dense checkpoints;
all legal inputs, contexts and original RNG; exact exposed outputs, including
logits when exposed; required successor state for all legal continuations;
batch one; one GPU with total peak <=8 GiB; same-machine native 4B Q4 warm
p50 <=1.2x and p95 <=1.5x; existing TTFT; every resource and fallback paid.

For a fixed, deterministic reference ABI write

    F_W(s, a) = (o, s_next).

The state s includes original RNG and required cache/control state. The
observation o includes every exposed native word and layout/alias/control field,
not merely the sampled token. Legal readback operations on a compressed state
are observations too. The same native RNG transition and consumption contract
must be preserved; compiler randomness is not substituted for that RNG. For a
consumption-sensitive contract, include the required draw count and ordered
externally visible effects among result roots as well. Matching only a final
RNG state is not asserted to prove an arbitrary consumption/effect contract.

The constructive statements below apply to a **supplied finite native program**
with explicit primitive semantics and bounds on its legal control flow. A
complete HF-to-this-program compiler, all public checkpoint/operator support,
reference reduction/NaN/FTZ/transcendental semantics, and all legal shape/control
cases are still O1/O3 obligations. No finite universal frontend is asserted by
calling an observed tensor graph the whole reference machine.

This package is self-contained theory and source inspection. It does not run a
model, GPU, numerical experiment, solver, backend, or enlarged scalar test suite.

## 2. Three principles, exact algorithms and paid entry conditions

Here U_native denotes the corresponding original dominant contribution, not a
measured final 4B latency. C/H below is only an explicitly justified finite
session amortization; cold initialization and TTFT remain separate obligations.

| Principle | Reversed premise and finite algorithm | Exactness contract | Actual >=10x condition and obstruction |
|---|---|---|---|
| Continuation-state minimization | Do not represent distinctions invisible to every future legal interaction. Enumerate reachable reference states, evaluate transitions, refine behavioral classes, then execute the quotient transition table | Equal class implies identical observations and equal successor classes for every legal input; include state readback and RNG | Constructor/H + class transition lookup + all output/state movement <=0.1 U_native. Reachability enumeration and class construction are fully charged. Neither a small class count nor a cheap transition source is established |
| Whole-transition native-relation elimination | Do not materialize each internal instruction result online. Compile exact instruction relations, eliminate internal variables, and store finite witness tables only for observed outputs/state | Existentially eliminating internal variables preserves the exact relation (s,a,o,s_next); determinism gives one correct observation/state tuple | Constructor/H + witness/address work + all table/source/output movement <=0.1 U_native. Literal compilation has exponential joint-width costs; online conditioning with ordinary forward substitution still visits all source factors |
| Exact incremental trace repair | Do not begin each transition with an empty computation. Retain an exact prior trace, propagate changed native words, stop a propagation edge only after exact equality | Induction over the dependency graph gives exactly the new reference trace and roots; changed shape/effect guards take fully paid reference reconstruction | Initial trace/H + dirty instruction work + dependency discovery + old/new trace I/O + guards/fallback <=0.1 U_native. Dense input changes can dirty every accumulator; fine traces have substantial storage and coarse traces retain full dense kernels |

These mechanisms differ in **time-state equivalence**, **within-step relation
elimination**, and **change-sensitive reevaluation**. None relies on a target
future token, a supplied answer trace, a free selector, or an unpaid exact
fallback. Their displayed inequalities are admission requirements, not proved
coverage, target latency, or existence of a qualifying implementation.

### Prior failures remain controlling

- Quotienting is semantically stronger than storing exact prefixes, but its
  explicit enumeration below remains the F-019/F-020 paid-advice obstacle. The
  existing prefix/RNG behavioral-state witness in EXP-089A already distinguished
  short state descriptions from cheap transitions. The existing
  [native-shortcut screen](../../docs/research/E0_NATIVE_EXACT_SHORTCUT_FRONTIER.md)
  explicitly identifies compiled checkpoint transducers as this rejected family.
  It is not reopened here
- Relational elimination is richer than the previous TOP/singleton demand
  executor: it carries joint constraints, not supplied desired outputs. A table,
  ROBDD, AIG, or grammar encoding is still subject to the recorded F-021/F-022,
  F-052/F-062/F-075 and source-accounting failures. Merely changing encodings or
  trying elimination orders does not reopen those families
- Native word-change propagation is different from a frozen tangent operator,
  approximate residual transport, or token-prefix reuse. It receives no free
  delta source and does not invalidate F-037/F-044/F-057/F-065. Its fully dirty
  case below stops the unconditional saving claim before any model experiment

No new qualifying core is selected. The next sections audit the strongest exact
whole-program constructor to expose what a genuinely different compact source
would have to supply, rather than promote a representation as an execution win.

## 3. Finite continuation quotient: exact, but construction is not free

Fix a finite bounded reference state space, legal input alphabet A, and initial
states (including all permitted original RNG states). Incorporate legality,
termination and readback into observable transitions; different available legal
operations cannot be silently merged. A finite maximum context belongs to this
bounded specification, not an invented relaxation of the mission.

A completely explicit constructor is:

1. Enumerate initial states and breadth-first enumerate every reachable state.
   Store native state words and exact equality keys; evaluate F_W(s,a) for every
   available input a, storing observations and the successor identifier
2. Initially put states together only when their immediate observations for all
   inputs agree. Refine a class whenever two members have different successor
   class identifiers for some input. Repeat until no class splits
3. Store one transition per class/input and keep an initialization procedure
   mapping each permitted initial state to its class. The latter is a paid
   lookup or comparison, not an assumed encoder
4. At runtime read the class/input record, emit the exact observation, and move
   to its successor class. Materialize any required state readback through the
   same proved observation interface, charging its complete work

Let S be the number of reached states, L=ceil(log2(max(S,2))), B_s the state
payload bits, B_o the maximum observation bits, and A the alphabet size. Even
before refinement, the literal stored payload can be bounded by

    S*B_s + S*A*(B_o + L),

plus equality indexes, pointers, legal-input guards, original weights and
workspace. Transition construction calls the original program up to S*A times.
One explicit refinement implementation forms A-entry signatures and stable
merge-sorts them. At most S splitting rounds give a conservative

    O(S^2 log(S) * A*(B_o+L)) bit-comparison/copy work,

in addition to native transition evaluation, reachability indexing and storage.
These are finite costs, not a declaration that the resulting S is small.

Correctness follows directly at the fixed point: class members have equal
observations and class-equivalent successors for every input. Induction over
any legal continuation therefore preserves all observations and RNG/state
behavior. If exact native cache words are an observable readback, distinct cache
words cannot merge into one class unless that readback remains distinguishable
by some additional stored information, which must then count as part of the
state. When only behavioral state is required, this argument does not require
raw cache materialization.

Few output words do not imply few classes: states can agree now and differ on a
later read or input. A class label of L bits also does not generate its outgoing
transition. Thus minimization supplies an exact finite baseline, but neither a
cheap constructor nor a cheap implicit table for the unchanged mission.

## 4. A complete whole-program relational source constructor

### 4.1 Preserve native arithmetic by changing the algebra of the proof

Unroll the supplied finite control bounds and put the program in explicit
single-assignment form, with guards for conditional/inactive instructions and
merges for branch results. Charge that expansion. An inactive guard enforces
unchanged effect-state and a defined canonical unused value; it does not execute
a raw side effect. Side effects, RNG, native memory versions, layouts and required
control fields must be explicit variables; unknown effects are not silently
declared pure. Let z=(s,a) be external inputs, y=(o,s_next) be required external
results, and h all other internal words.
For each original instruction v, define its exact word relation

    R_v(inputs_v, output_v) = 1 iff output_v is its original native result.

A primitive relation can initially be represented by the opcode, original
constant operands and exact primitive-relation evaluator. This Boolean evaluator
must terminate safely on every enumerated assignment: for invalid/undefined
local tuples return false, rather than dereference an unsafe raw address or
execute an invalid side effect. Defined reference exceptions/error results are
represented according to the reference, not reclassified as invalid. On valid
active assignments, evaluate the supplied exact primitive semantics and compare
the complete word(s); charge that work. This is an obligation on the supplied
finite primitive semantics, not a claim that every HF operator has been lowered.
Alternatively build its entire truth relation by enumerating all finite operand
words and executing the primitive. That exponential construction is charged,
never implicitly granted as a small lookup table.

The exact whole-program relation is

    R_W(z,y) = OR_h AND_v R_v.

The AND and OR operate on Boolean truth indicators. No native FP32 reassociation,
real-arithmetic inverse, or semiring law for rounded accumulation is invoked.
Acyclic native execution and exact local relations imply, by instruction-order
induction, that each legal z has exactly its original y as a satisfying result.

### 4.2 Eliminate hidden variables without retaining their online values

Choose a deterministic finite variable order: all hidden words first, followed
by output words; keep z symbolic throughout construction. An arbitrary fixed
order suffices for termination; finding an optimal order is not a free step.

For the next variable x, collect every current factor whose scope includes x.
Let B be the union of these scopes and S=B minus x. For **each complete native
assignment to S**:

1. Enumerate every native word x in its finite domain
2. Evaluate all bucket factors on (S,x), including original primitive evaluators
3. Write the Boolean OR of their conjunctions into a new factor on S
4. If x is an output word, additionally store the lexicographically first
   satisfying native x, or an explicit no-witness flag when no value satisfies

Replace the bucket by that factor and continue. Hidden-word witnesses are not
stored. Output-word witness tables are stored with their precise separator
variable order and word widths. If the reference is partial, its legality/error
result must be retained; it is not dropped from the boundary.

**Query algorithm.** Start with the actual z. Read output witness tables in
reverse output-elimination order. Every table is indexed only by z and output
words already recovered in that reverse order. Form the full multiword address,
read its record, and write the recovered output word. For legal z, the final
existence condition holds and elimination guarantees a witness at every step.
The recovered tuple satisfies R_W; determinism makes it exactly F_W(z).
No hidden accumulator, hidden row result, or full execution trace needs to be
reconstructed by this query procedure.

This is an explicit finite **source of the whole observation/state tuple**.
It proves that separately materializing every matrix's internal state is not a
logical requirement. It does not prove that the source is compact or affordable.

### 4.3 Full sufficient inventory of the literal constructor

For bucket t let b_t be the total bit width of B, d_t its factor count, w_t the
eliminated word width, and e_t=b_t-w_t. Let L_t be a uniform upper bound, over
every full bucket assignment, on factor evaluations, all table probes, Boolean
operations and multiword address/loop work. Assignment-dependent costs may
instead be summed explicitly. An explicit sufficient construction inventory is

    original source scan + graph/order/index construction
      + sum_t 2^(b_t) * L_t
      + all produced table/witness writes and allocation work.

A Boolean message uses 2^(e_t) bits. An output witness table can use
(w_t+1)*2^(e_t) bits, including a validity bit. A conservative storage bound is
the sum of all such tables plus original weights, primitive records, graph,
indices and scratch. A liveness schedule can free consumed message tables, but
that changes the actual peak only after its schedule is accounted for.

For a query, each output table read pays its entire w_t-bit result, validity
handling, separator gathering, multiword address construction, physical memory
transactions/page faults and output write. Addressing a 2^e-entry table needs e
index bits; it is not one fixed 64-bit operation when e>64. Required state,
original/transformed storage, host/SSD/GPU placement and synchronization remain.

No table is actually built in this study. The algorithm's finite but enormous
preprocessing/storage is precisely why it is not an admitted source. Calling the
query one lookup per output without these terms would reproduce the rejected
free-advice shortcut.

## 5. Concrete proof attempt: dense syntactic separators do not stay small

A tempting proof would derive small compiled tables just from the narrow scalar
instruction fan-in or the number of final output words. That proof fails even
for one ordered native dense MatVec with N inputs and N output streams:

    a_(i,0)=+0
    a_(i,j)=FMA32(w_(i,j), x_j, a_(i,j-1)).

Use word variables x_j (BF16) and a_(i,j) (FP32), and exact instruction factors.
Consider the **unconditioned syntactic interaction graph**, joining word
variables that occur in one factor. The accumulator variables in each row form
a connected path. Contract each row path to one vertex r_i. Every x_j is
adjacent to every r_i. Thus the graph has K_(N,N) as a minor.

For completeness, K_(N,N) contains a clique minor of size N+1: pair left vertex
i with right vertex i for i=1,...,N-1, and leave the two last vertices singleton.
All these connected branch sets touch one another. The elementary treewidth
minor property then forces an elimination bag of at least N+1 word variables.
To locate the large bag within the actual compiler, append the N symbolic input
variables to its order after all accumulator eliminations. Bags during these
last N input eliminations have at most N variables. Hence a bag of size at least
N+1 must already occur while an accumulator is eliminated, creating a message
on at least N remaining variables. For this literal word-table procedure, each
remaining variable has at least 16 bits. At some elimination step at least N
such variables remain in the separator, so its allocated Boolean message has
at least

    2^(16*N) bits.

At N=16,384 this is 2^262144 bits, before witness payload, original weights,
indices and workspace. This statement concerns full-domain **literal table
allocation on the unsimplified word graph**. It does not lower-bound every
semantic relation representation, bit-level compilation, sparse relation,
reachable-state restricted program, circuit, or native execution algorithm.
Zero/constant factor simplification and genuine semantic cancellations may
change that graph; their exact discovery and encoding costs must be proved.

Conditioning on the current x before elimination removes these shared input
variables from the graph. But the direct finite forward-substitution schedule
then computes the original N^2 FMA factors and consumes their coefficient
source. That is a cost of this schedule, not a theorem that every conditioned
algorithm must read N^2 coefficients. It identifies where the unproved source
would have to enter: **before** replacing the whole program by ordinary dense
forward substitution, while preserving joint native relations compactly.

The [complete scalar charts](../binade_transition_20260930/native_charts/REPORT.md)
compact one accumulator's dependence on its initial scalar value after the
addend stream is supplied. They do not encode these joint unknown-input and
cross-row separator relations. The
[grammar accounting](../tree_grammar_source_accounting_20260930/REPORT.md),
[suffix coverage limit](../heterogeneous_source_20260930/REPORT.md), and
[packed arithmetic audit](../static_matvec_literature_20260930/SOURCE_NOTE.md)
remain intact. None supplies the missing joint relation constructor.

## 6. Concrete proof attempt: exact dirty propagation is not a small delta source

For a supplied fixed DAG, retain each prior native word and its dependency list.
Compare changed external inputs, mark their consumers dirty, and process dirty
nodes in topological order. Reexecute the original instruction, compare its old
and new complete words, store the new word and propagate only if they differ.
Enqueue each node at most once after its predecessors have settled. Outputs and
required state follow by topological induction. Dynamic topology, cache growth,
aliases and effects require explicit structural guards and a paid trace rebuild.

A finite symbolic counterexample refutes the proposed universal premise that a
changed dense input vector or finite rounding necessarily leaves a small dirty
region. This witness does not instantiate a single-token transition in a released
checkpoint. For the same N-by-N ordered scalar program, choose any coefficients

    w_(i,j) in {1 + k/128 : k=0,...,127},

old inputs x_j=+0 and new inputs x_j=1. At N<=16,384 all new prefix sums are
positive multiples of 1/128, less than 32,768. Their scaled integers are below
2^22, so each prefix is exactly representable in FP32. Every old accumulator is
+0 and **every one of the N^2 new accumulator words differs**. There is no
rounding reconvergence at which this trace algorithm can stop. Distinct row
patterns can be selected; no row-equality shortcut is part of this argument.

This is a native subprogram witness, not a claim about reachable states of a
particular released checkpoint or a measured p50/p95 population. Another
algorithm may exploit these inputs or coefficients; the witness is scoped to
the asserted dirty-propagation bound. No numerical test is necessary or run.

At scalar granularity, one retained FP32 accumulator per coefficient occupies
4*N^2 bytes: 1 GiB for one 16,384-square, before operands, graph edges, versions,
metadata and working copies. Illustratively, **if** one such record were retained
for each of the registered 405,849,243,648 parameters, the record payload alone
would be 1,623,396,974,592 bytes. This is not an assertion that every parameter
is a per-token FMA or that all traces must be GPU-resident. Host/cold traces have
paid storage/movement. At tensor granularity the trace may be much smaller, but
a dirty tensor node executes its whole original dense kernel.

Thus a real small-change source/coverage theorem, not the word comparison itself,
would be needed. No such theorem is derived from the input alphabet, causality,
finite precision or the short description of a token history.

## 7. The missing whole-program lemma, with no oracle promoted

The precise requirement for the relational route is stronger than a short formula
for one scalar map. For each replaced region of the native program, a candidate
must construct a representation C of its relation on the region boundary and,
where used, a causally maintainable reachable-state invariant I. It must provide:

1. A finite constructor from original checkpoint and pinned native program,
   including discovery of I/C and checking any legality guards
2. Exact equivalence of the region and C on I, preserving all observer roots,
   native word semantics, memory effects, RNG and successor-state correspondence
3. An explicit causal evaluator that obtains boundary inputs and produces needed
   outputs without first evaluating the eliminated original region
4. Paid upper bounds on constructor, invariant maintenance, relation joins,
   witness selection, address generation, storage, every memory tier, output,
   state, guard failures and exact repair over the unchanged workload domain
5. A complete schedule satisfying the entry saving and final target, with no
   unproved compression ratio, small invariant, solver efficiency or cheap decoder

The attempted proof from scalar fan-in fails by section 5's joint-width witness.
The attempted proof from temporal locality fails by section 6's fully dirty
witness. The attempted proof from short behavioral-state descriptions fails by
section 3's transition-table construction. Exposed logits/readbacks also prevent
silently weakening the observer to a token decision.

A different semantic compression could defeat these particular obstructions.
None is constructed here. Stating this lemma does not close it, and it is not
claimed as a newly invented general solution: it makes the existing whole-program
source obligation concrete for one algorithm. Further table/order sweeps,
bit-blast growth, trace instrumentation or tiny examples would not supply it.
At this point I cannot construct a further qualified route from these mechanisms;
the scientifically justified disposition is **no core promotion**, not invented
novelty or a blanket impossibility claim.

## 8. Target accounting and O1-O6 disposition

For any candidate, a sufficient paid serial session bound must include

    load + compile + prefill
      + sum_steps(query + source/table transfers + observer/state writes
                  + address/guard/verification work + allocation/synchronization
                  + actual repair/fallback).

Any claimed overlap needs its own bounded schedule. GPU peak includes all hot
weights/tables/code, live state/KV, old/new buffers, workspace, allocator reserve
and fragmentation. CPU/RAM/SSD/PCIe/HBM and finite construction amortization count.
The registered two-byte parameter payload is 811,698,487,296 bytes, not a claim
that every parameter must be read every token or live on the GPU. Query-only
table cost does not discharge startup/TTFT or total transformed storage.

No complete service-time upper bound, certified same-machine baseline comparison,
frozen workload quantile protocol or target allocation is supplied by this
package. No raw work ratio is promoted to the required p50/p95 latency ratio.

- **O1 OPEN:** finite bounded-program constructors are described; uniform cheap
  arbitrary-public-checkpoint/native frontend and construction are absent
- **O2 OPEN:** a complete observation/state relation is constructible in the
  bounded model, but the only explicit source pays prohibitive tables or the
  original dense traversal; complete cheap load/prefill/generation is absent
- **O3 OPEN:** scoped exact relation/quotient/trace inductions are derived; actual
  all-HF/all-kernel semantics, state/RNG and continuation coverage remain absent
- **O4 OPEN:** explicit finite inventories expose construction/table/trace costs;
  no sufficient affordable bound closes the source obligation
- **O5 OPEN:** target 405B, <=8 GiB, TTFT and 4B p50/p95 comparison are NOT TESTED
- **O6 OPEN:** present derivations and document checks are inspectable; no complete
  independently checked final theorem or executable fast backend is delivered

No full-mission obligation is marked closed by these bounded algorithms.

## 9. Prior art, provenance and validation boundary

The general algorithms are established techniques, not novelty claims.
[Dechter, *Bucket elimination: A unifying framework for reasoning* (1999)](https://ics.uci.edu/~dechter/publications/r76A.pdf)
provides the variable-elimination and width/conditioning framework.
[Acar et al., *A Library for Self-Adjusting Computation* (2006)](https://www.cs.cmu.edu/~blelloch/papers/ABBHT06.pdf)
and [Acar, Blume and Donham, *A Consistent Semantics of Self-Adjusting Computation*](https://arxiv.org/abs/1106.0478)
provide background on change propagation and its semantic correctness. These
sources do not establish the required Transformer compression or target bound.
The explicit native relation, clique-minor instantiation, dirty-stream witness
and mission-specific accounting above are direct derivations in this report.

Local start was 4ac2dd1d7133f9d13eb210dbd58e9db2fa9512db. During this task the
parent advanced the shared checkout; local HEAD and the remote branch were read
at [d94f1df60c8cfac37ba434cd87618f1980ebf43f](https://github.com/yjunhyuk920-ui/Vortex/commit/d94f1df60c8cfac37ba434cd87618f1980ebf43f),
`research/cloud-continuation-20260930`, origin
`https://github.com/yjunhyuk920-ui/Vortex.git`. The branch-scoped all-state PR
query returned an empty list. A first encoded branch URL was rejected as an
invalid path; the supported Git-ref read then supplied the actual SHA. This
worker performed no remote write, commit, publication or root-ledger edit.

The root README was reviewed: its statement that no qualifying cheap source is
established remains accurate. Any new link/frontier entry belongs to parent
integration. This package's plan openly records that initial analytic exploration
preceded the plan; no numerical preregistration or hidden experiment is implied.

Only text, link, arithmetic-inventory and checksum checks are performed here.
They are not runtime, numerical, model, hardware or formal-proof verification.
The old failed evidence and native scalar packages are not edited.
