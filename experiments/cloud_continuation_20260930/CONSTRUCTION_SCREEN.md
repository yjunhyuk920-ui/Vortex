# Three finite constructions screened against the unchanged mission

This is a theoretical screening record, not three qualifying new discoveries.
No model experiment is selected from these constructions. The exact target,
O1–O6, output/RNG/state contract and complete-cost rules remain unchanged.

## 1. Compose native accumulator transitions

Replace numerical reassociation by function composition. For a sequential
separate-product reference, define
`f_j(a) = RN32(a + RN32(w_j*x_j))` and `F = f_n o ... o f_1`.
Associativity of function composition does not require floating-point addition
to be associative. Evaluating `F` on the original accumulator gives the original
ordered result, provided all special values, signs and stores are represented.

A literal finite implementation constructs every leaf map and composes tables.
One FP32-to-FP32 table already occupies `4*2^32 = 16 GiB`. An actual kernel with
r live FP32 accumulators requires its real register frontier and instruction
schedule, not this scalar reference; the literal joint table has
`4*r*2^(32*r)` bytes. These are costs of the explicit table construction, not
lower bounds on every possible compressed map.

Even granting free map compression/composition, online construction from the
heterogeneous products visits all coefficients and performs all products. For
the selected separate multiply/add model this leaves approximately half of
arithmetic and all coefficient traffic. It does not supply a 10x paid path.
Precomputing responses for arbitrary finite BF16 input pairs has
`65280^2` entries, requiring 15.875 GiB of FP32 payload even before incoming-state
dependence. General finite tables are already scoped by F-060/F-062.

The missing construction is a compact transition producer that itself bypasses
the original dominant leaf work. Compact representation alone is insufficient.
[Kapre–DeHon](https://ic.ese.upenn.edu/pdf/fpaccum_arith2007.pdf) provides an exact
sequential-result parallelization precedent, but checks the native recurrence
at every position and charges correction; it is not evidence of cheap matrix
effects. Correctly rounded exact-real summation is also not the required native
rounding contract.

Decision: retain the semantic abstraction only; no core admission or large table
experiment. Reopening requires a new direct producer and paid query/construction
bounds, not another associative spelling of the same work.

## 2. Globally nonlinear adaptive word encoding

Reverse the restriction to linear advice/fixed probe sets: an encoded word may
choose the next address. The F-089 interface is not excluded by several linear
data-structure bounds: a 25x108 binary tile, 50 64-bit words and two adaptive probes
for a rank-one parity query. This is only an unresolved capacity interface.

A finite realization could enumerate all encoders and adaptive decoders, checking
all matrix/query pairs before returning the first correct program. A literal
encoder has `3200*2^2700` description bits, and enumerating those descriptions has
`2^(3200*2^2700)` possibilities. This explicitly terminates in a finite universe,
but does not meet construction or storage budgets. These figures describe that
exhaustive algorithm, not a universal impossibility theorem.

A non-enumerative small encoder/decoder with a native numerical lift could in
principle avoid most coefficient reads. No such algorithm is supplied. Boolean
parity, globally shared advice, native products/carries/rounding and successor
state cannot be silently substituted for one another.

Decision: OPEN interface, no admitted algorithm. The decisive requirement is the
actual encoder, adaptive address program, decoder and joint global/native cost
proof. A capacity inequality or another small synthesis timeout is not enough.

## 3. Compile exact continuation equivalence

Reverse storage of native tensors: represent a state by its class under all
legal future observations and transitions, including original RNG effects.
For a bounded legal context C, a finite construction can enumerate reachable
prefixes, execute their original transitions and refine a partition until every
class has identical observations and successors. Execution then uses the class
transition table and original sampler. The inductive congruence is sufficient
only if initialization, observations and every legal successor are all preserved.

Let R denote enumerated states and E transitions. The constructor pays original
transition evaluations over E, stored observations/states/edges and partition
refinement. For vocabulary 128256, even the simple enumeration through length two
has `1+V+V^2 = 16,449,729,793` nodes. Storing a BF16 V-logit vector per node costs
`4,219,553,088,662,016` bytes before edges or hidden state. This is the naive
construction's cost, not a lower bound on the minimal quotient.

The possible 10x route would require a small quotient generated directly from
program structure with cheap exact observations and successors. None has been
constructed. Prefix enumeration repeats F-019/F-020; storing prefix IDs while
replaying the body repeats EXP089's paid behavioral witness.

Decision: retain correctness specification only. Require a non-enumerative
continuation constructor and cheap heterogeneous effect producer before execution.

## Result and next scientific condition

No survivor has a qualifying >=10x full-cost route. There is no justification for
another model run or backend build for these direct realizations. Neither this
screen nor the old ledgers prove the fixed mission universally impossible.

The exact next missing item is an explicit non-enumerative source program in one
of the above open interfaces (or another genuinely different one), with O2/O3
native semantics and O4/O5 cost closure. Calling such a program a selector,
compiler or state code does not construct it. The research goal remains unmet;
these exclusions are not presented as increased feasibility or goal completion.

THEORY_STATUS=NOT_ESTABLISHED
CORE_ADMISSION=false
THREE_QUALIFYING_NEW_PRINCIPLES=false
FULL_MISSION_O1_O6=OPEN
