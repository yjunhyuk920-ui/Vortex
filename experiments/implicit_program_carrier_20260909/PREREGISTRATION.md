# Implicit program-carrier frontier — preregistration

Date: 2026-09-09 Asia/Seoul

Base scientific/persistence head:

```text
925949770cfb231ea1d79b6609023e5e18478440
research/nonlinear-router-frontier-20260908
```

This round keeps the fixed VORTEX mission and
`docs/CONSTRUCTIVE_THEORY_CONTRACT.md` unchanged. It starts only after the
previous implicit-direct-query round was committed, pushed, remotely verified,
and the worktree was clean.

The preceding round already rejects, in their exact recorded scopes:

```text
complete one-sided linear query-image dictionaries
duplicate weight-value / identical accumulator-state transition sharing
coordinate/product-preserving safe gauges
```

It also preserves all older exclusions: reconstruct-all coding, an unspecified
rounding witness after an exact sum, Gauss-Jordan row-program replay, static
Lupanov/Steiner programs with free instruction fetch, Boolean-semiring exact
lifts, full-scan broadword packing, literal answer tables and finite-state tables.

The new question is more specific:

> Can checkpoint-dependent **program information** be carried implicitly by
> routing, a query-time structural synthesizer, or a fully paid encoded-state
> nonlinear operator, so that the runtime does not fetch roughly one original
> coefficient bit/word merely to know what cheap operation to execute?

No result in this round may be inspected before this preregistration is committed
and pushed.

## Intended final theorem

The round succeeds only if one principle yields a uniform finite constructor for
every legal unchanged checkpoint and a complete causal algorithm satisfying:

```text
G = Compile(checkpoint bytes)

addresses/program = RouteOrSynthesize(
    G,
    current causal query/state,
    prior returned words
)

payload = Read(G, addresses/program)

(effects, state') = DecodeAndExecute(payload, current query/state)

effects = exact reference finite-word dense effects
state'  = exact required reference successor state
RNG'    = exact reference RNG state/consumption
```

for every checkpoint and every legal continuation, while every constructor,
representation, route/program bit, RAM/SSD/PCIe/HBM movement, GPU allocation,
workspace, arithmetic/reduction, state/KV, initialization, verification,
repair/rollback and fallback cost is charged.

The universal paid runtime must remove/amortize at least 90% of the original
dense parameter-effect work/traffic as a core-entry condition and ultimately
close the same-machine native-4B-Q4 p50/p95 and TTFT objectives. A favorable
binary construction is only a necessary-condition control until native semantics
and the whole causal executor are supplied.

## Frozen common E0 control

The first decisive controls use exact GF(2) MatVec because the prior causal
bridge establishes it as a necessary information-bearing subclass without
claiming that GF(2) is the native target ABI.

For the square control:

```text
m = n = 16,384
dense coefficient bits       = 268,435,456
favorable leaf+add slots      = 536,870,912
```

The registered whole-model non-embedding source remains:

```text
403,747,897,344 coefficient bits
883 matrices
```

The historical `8/675` line is used only as a logical comparison, not as the
final latency theorem.

The completed previous suites are not rerun:

```text
implicit-direct-query final 15/15
direct-global 10/10
prior nonlinear 15/15
prior nonlinear/geometry 28/28
causal-global 14/14
Boolean exhaustive controls
native-global 3,510-file replay
```

## Principle I — address-only alias router

### Reversed premise

Do not store the checkpoint-dependent selector as an ordinary payload word or
instruction. Put the selector into the **routing relation itself**. Runtime
issues a fixed checkpoint-independent sequence of logical alias reads; the
compiled alias map redirects each read to a query-side value table.

This directly tests whether “the address already knows the weight pattern” is a
real source or merely moves the source bits into uncharged page-table/routing
metadata.

### Finite binary candidate

For an `m x n` binary matrix, use the same deterministic 10/11-bit partition of
the input dimension as the prior direct-query control, but reverse which side is
checkpoint dependent.

For each query block `B` and every local pattern `p`, runtime constructs the
checkpoint-independent query table:

```text
Q[B,p] = <p, v_B> mod 2.
```

This table depends only on the current query and can be built by a finite
subset-parity recurrence.

The compiler stores, for every row `i` and block `B`, one alias mapping:

```text
A[i,B] -> Q[B, W[i,B]].
```

The runtime does not fetch `W[i,B]` as an ordinary value. It walks the logical
aliases in fixed row/block order:

```text
for i in rows:
    y_i = 0
    for B in blocks:
        y_i ^= AliasRead(A[i,B])
```

The abstract alias map is deliberately favorable: one logical dereference may
perform whatever physical translation is required, but every translation bit,
cache entry, miss, page-table/routing lookup and backing byte is paid. A real
virtual-memory/PTE implementation is an optional physical realization, not a
free oracle.

### Stored representation and address rule

The representation is:

```text
block partition metadata
alias relation (i,B) -> local pattern p
any physical routing/page-table form of that relation
```

The runtime logical address is checkpoint independent:

```text
logical_address = deterministic(i,B).
```

The checkpoint is present only in the alias relation that resolves the logical
address to local pattern `p`.

### Exactness and state equation

For every row:

```text
XOR_B AliasRead(A[i,B])
= XOR_B <W[i,B], v_B>
= <W_i, v>
= (Wv)_i.
```

Thus the binary operator is exact. If a native lift were later proved for every
dense operator and substituted in topological order, identical incoming
state/RNG would imply identical effects and successor state/RNG. No such native
lift is assumed here.

### Termination and paid upper bound

All loops are finite. For the registered 10/11 partition at `n=16,384`:

```text
blocks                         1,638
query-table recurrence ops     1,679,770
logical alias reads           26,836,992
candidate table+row XOR ops   28,516,762
candidate / leaf+add baseline  0.053116608411073685
```

This supplies a concrete >=90% **arithmetic-slot** route in the favorable binary
model before routing cost.

Every one of the following is separately charged:

```text
alias-map construction
complete persistent alias/routing representation
translation cache/TLB capacity
translation misses/page walks/routing reads
query table construction and workspace
logical alias reads
row XORs/output writes
CPU/RAM/SSD/PCIe/HBM/GPU movement
KV/state/native lift/fallback if any
```

### Strongest adversary and frozen falsifier

Use an arbitrary incompressible binary matrix. The alias relation must distinguish
all `2^(mn)` matrices. Because concatenating the local target patterns over all
`(i,B)` recovers the matrix exactly, any lossless alias representation requires
at least `mn` checkpoint-information bits before physical routing overhead.

The decisive question is not storage capacity alone but **query-time use** of
those bits. If the implementation claims they need not be read because routing
hardware “already knows” them, explicitly account for the persistent routing
state and every cache/miss/translation operation. As a concrete physical upper
representation, a 64-bit descriptor per alias at `16,384^2` with 1,638 blocks
would already be `64/10`-scale metadata rather than compressed source.

Reject the principle as a core if checkpoint routing information is merely the
old coefficient/program stream under a metadata name, or if its physical
translation payload/cache state fails the target line. Do not weaken the
falsifier by assuming an unbounded zero-cost routing fabric.

## Principle II — query-time Patricia pattern synthesizer

### Reversed premise

Do not precompute every possible query image and do not fetch one flat selector
per row/block. Compile only the **structural sharing among the checkpoint's row
patterns**, then synthesize the required block parities by traversing that
structure once for the current query.

This attacks a different source of program cost: checkpoint patterns may share
prefix structure even when their complete patterns are all different.

### Finite binary candidate

Partition each matrix row into blocks of width at most 64. For each column block
`B`, collect the `m` row patterns:

```text
p_i = W[i,B] in {0,1}^|B|.
```

Build a deterministic compressed binary Patricia trie of the distinct patterns.
Every compressed edge stores the exact checkpoint bits on the skipped segment,
represented as a local 64-bit mask plus its depth interval. Every leaf stores
the finite set/list of rows carrying that pattern.

At query time, traverse the complete trie once. If an edge from a parent covers
mask `M_e`, update the parity carried to the child by:

```text
parity_child = parity_parent XOR parity(M_e AND v_B).
```

At a leaf, the carried parity is exactly `<p_i,v_B>` for every member row.
Accumulate that bit into each row's result for this column block. Repeat for all
blocks.

No full `2^64` query table exists and no one-sided output image is stored.

### Stored representation and address rule

Per block store:

```text
Patricia node topology
edge intervals
edge label masks
leaf row memberships
```

Runtime addresses follow the deterministic trie topology. There is no search
over all possible patterns; every compiled edge is visited exactly once.

### Exactness equation

Along the unique root-to-leaf path for pattern `p`, the compressed edge masks
partition the coordinates/bits of `p`. Therefore:

```text
XOR_(edge e on path) parity(M_e AND v_B)
= parity(p AND v_B)
= <p,v_B>.
```

XOR across all column blocks gives exact `Wv`.

### Termination and favorable >=90% route

For `d_B` distinct block patterns, a Patricia trie has at most `2d_B-2` edges.
For the square `16,384` control with 64-bit blocks:

```text
blocks                         256
max edges/block               32,766
max edge evaluations        8,388,096
row-block leaf contributions 4,194,304
edge+leaf events            12,582,400
event / leaf+add baseline    0.023436546325683594
```

The registered favorable primitive gives one exact 64-bit
`parity(mask & query_word)` per edge. This is a concrete >90% word-arithmetic
route before checkpoint-program traffic and metadata are charged.

The complete paid bound additionally includes every edge-label word read,
topology/child address, row-membership read/scatter, trie build, workspace,
output accumulator, RAM/SSD/PCIe/HBM/GPU traffic and all native/state work.

### Strongest adversary and frozen falsifier

Use row-block patterns chosen independently so almost every row is distinct and
shared prefixes end near `log2(m)`. The implementation is not allowed to count
one edge parity event while omitting the checkpoint edge label that defines that
parity.

The concrete first falsifier is stronger and simpler: with one 64-bit label word
per Patricia edge, the maximum-edge square control reads:

```text
8,388,096 edge-label words
= 536,838,144 edge-label bits
~= 1.9998779296875 x the raw binary matrix bits
```

before topology and row memberships. If a packed/compressed label encoding is
introduced, prove its arbitrary-matrix worst-case size and the exact runtime
decoder; do not assume random-pattern compressibility. Reject the candidate if
the shared trie saves arithmetic but returns to source-scale checkpoint program
traffic.

## Principle III — rank-normal encoded state with a fully paid transformed nonlinearity

### Reversed premise

The preceding gauge round rejected only encodings that demanded the original
coordinatewise product remain cheap in the transformed coordinates. Remove that
restriction completely. Allow a dense checkpoint-dependent gauge that makes
each linear map sparse, but **construct and pay the transformed nonlinear
operator explicitly** instead of calling it free.

This is not the prior safe-gauge class.

### Finite binary candidate

For every binary `m x n` matrix `W`, deterministic Gaussian elimination produces
invertible row/column transforms `A` and `B` such that:

```text
A W B = J_r
```

where `J_r` is the rank-normal form with an `r x r` identity block and zeros
elsewhere.

Encode the input/output wires as:

```text
z_in  = B^-1 x
z_out = A y.
```

Then the dense linear operator is replaced exactly by:

```text
z_out = J_r z_in,
```

which uses at most `r` copied coordinates and therefore can remove essentially
all dense linear arithmetic for full-rank square matrices.

For every nonlinearity `N` between encoded wires, the compiler must build the
actual conjugated operation:

```text
N_enc(z) = E_out N(E_in^-1 z)
```

and charge it. For a binary two-input Hadamard/AND node:

```text
H_enc(z1,z2)
  = E_out( (E_1^-1 z1) AND (E_2^-1 z2) ).
```

The baseline finite implementation of `H_enc` is the explicit factorized
program: apply the two inverse transforms, execute coordinatewise AND, then apply
the output transform. A direct compiled bilinear tensor/table is allowed only if
its full storage/build/runtime is cheaper and explicitly represented.

### Stored representation and addresses

Store finite elimination/gauge descriptions and every transformed operator needed
by the graph:

```text
A, B or their exact executable programs
rank-normal linear map J_r
inverse/forward maps required by transformed nonlinear/residual/cache nodes
all layout/branch/state encoding metadata
```

Runtime addresses are those of the explicit transform/nonlinear programs. No
“stay encoded” step may suppress a required conversion merely because it is
checkpoint dependent.

### Exactness and successor-state equation

Every rewritten node is defined by exact conjugation:

```text
z_out = E_out F(E_in^-1 z_in).
```

Induction over the complete graph gives `z=E h` at every encoded wire. Public
logits, KV/cache and all externally required state are decoded exactly before
the reference sampler. Hence, if every rewritten node is implemented exactly,
RNG consumption/state is unchanged. This is a proof obligation of the candidate,
not an assumed native result.

### >=90% route

For a full-rank `n x n` GF(2) dense map, rank-normal form changes `n^2` binary
coefficient effects into at most `n` coordinate copies:

```text
linear effect fraction <= 1/n.
```

At `n=16,384` the isolated linear operator therefore has far more than the
required 90% removal. The entire candidate survives only if the newly paid
nonlinear/residual/cache transformations do not restore dense work.

### Strongest adversary and frozen falsifier

Use the smallest graph that forces two independent dense maps to meet at a
coordinatewise product:

```text
a = W1 x
b = W2 x
c = a AND b
```

with arbitrary invertible dense binary `W1,W2`.

Choose the maximally favorable rank-normal output encodings:

```text
z1 = W1^-1 a = x
z2 = W2^-1 b = x.
```

Then the exact transformed product in these coordinates is:

```text
c = (W1 z1) AND (W2 z2).
```

The literal exact transformed nonlinearity therefore reintroduces both original
dense MatVecs that the encodings removed. Any alternative direct bilinear
implementation must be finite and pay its tensor/circuit/program bits; an
undefined “cheap conjugated AND” is not admitted.

Reject this rank-normal implementation if its explicit nonlinear nodes restore
the removed dense work. Do **not** promote that rejection into a theorem that all
joint graph encodings are impossible; a globally co-designed encoding/nonlinear
representation remains a separate open class unless actually closed.

## Frozen ordering and selection

Principle I is selected first because it attacks the exact unresolved F-050
program-fetch premise while keeping a concrete arithmetic route and fixed
checkpoint-independent runtime addresses.

If its paid routing state returns to source-scale work/traffic, execute Principle
II rather than stopping. Principle II tests structural query-time synthesis
instead of address-only indirection.

If Principle II also fails, execute Principle III. Principle III explicitly
opens the branch left by the previous safe-gauge theorem by constructing the
transformed nonlinearity and paying it.

No candidate rejection, helper theorem, test result or commit is the mission
result. After all three, preserve their evidence and continue constructive
research if no complete theory has emerged.

## Required artifacts and O1--O6 before execution

After execution, create an immutable fresh result directory and record:

```text
algorithm/constructor descriptions
exact finite controls
all registered cost arithmetic
claim boundaries
REPORT.md / REPORT_KO.md
VALIDATION.md
obligations.json
HANDOFF.md
affected root ledgers
```

Current obligations before execution:

```text
O1 OPEN
O2 OPEN
O3 OPEN
O4 OPEN
O5 OPEN
O6 PARTIAL only after reproducible artifacts exist
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
```

No 405B, CUDA, single-GPU `<=8 GiB`, PCIe/SSD/HBM, native4BQ4 p50/p95 or TTFT
claim is authorized by this E0 round unless actually executed later.
