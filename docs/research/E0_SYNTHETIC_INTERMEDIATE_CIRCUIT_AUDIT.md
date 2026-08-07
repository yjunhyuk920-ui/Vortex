# E0 Synthetic-Intermediate Exact-Circuit Audit

Date: 2026-08-08
Evidence level: E0 `DERIVED`
Model or hardware execution: none

## Question

Does adding compiler-created exact coefficient vectors to the terminal-only
Hamming tree create a materially new cold-backed execution class, and does any
such static class have a fully charged route to the registered
`1.185185185%` operation and traffic allowance?

## Decision

```text
REJECT_STATIC_SYNTHETIC_INTERMEDIATE_TREE_OR_DAG_AS_NEW_CORE
KEEP_QUERY_ADAPTIVE_COLD_BACKED_INFORMATION_SOURCE_OPEN_AT_E0
```

A synthetic Hamming/Steiner tree can be exact, but it is a restricted static
linear circuit already contained by archived EXP-072B. Moving that circuit
from hot memory to cold storage changes residency, not the computation or its
information source. No static candidate passes E0 novelty plus resource
closure, so no EXP-083 number, compiler, runtime, larger download, or hardware
work is authorized.

This audit does **not** reject every cold-backed online executor. It leaves one
materially different interface open: the current causal activation may select
a small query-dependent subset of cold information, provided selection,
probes, misses, verification, fallback, and state are all charged.

## Exact class boundary

Let a quantized matrix have coefficient-vector terminals
`S = {w_1, ..., w_m}` and let `x` be the current causal activation. A synthetic
node is any exact coefficient vector `v` that is not necessarily in `S`; its
runtime value is `z_v = v . x`.

For a directed synthetic tree edge `u -> v`, exact evaluation is

```text
z_v = z_u + (v - u) . x.
```

Expanding the sparse or dense delta into scalar constant-input products and
adds produces a static linear straight-line program. A general static linear
DAG may also share arbitrary previously constructed linear forms, so the exact
containment is

```text
terminal Hamming tree (EXP-082A)
  strict subset of synthetic Hamming/Steiner trees
  subset of static exact linear DAGs
  subset of the exact nonlocal arithmetic-DAG search archived as EXP-072B.
```

The first containment is strict because a Steiner vertex need not be a source
row or column. The second is generally strict because a DAG can reuse multiple
non-tree linear forms and fan them out. EXP-072B already defined synthetic
coefficient vectors and recurrences such as `z_k = a z_i + b z_j`; a Steiner
tree is therefore a topology restriction, not a new information source.

EXP-053's AIG and EXP-054's decision diagram are alternative exact function
representations, not formal supersets in the chain above. If a cold runtime
traverses only an activation-dependent portion of a program, however, its
selector and probe path become the same kind of online data-structure burden:
the path must be derived causally and fully charged. Renaming static circuit
instructions as cold pages does not supply that selector.

## What EXP-082A does and does not lower-bound

For any terminal set in a metric, let `MST(S)` be the terminal-only minimum
spanning-tree cost and `SMT(S)` the Steiner-tree cost. Doubling a Steiner tree
gives a terminal tour of cost `2 SMT(S)`; metric shortcutting cannot increase
that tour, and an MST is no more expensive than a terminal tour. Therefore

```text
SMT(S) >= MST(S) / 2.
```

Applying only this generic relation to EXP-082A's registered favorable lower
bound gives

```text
terminal-tree lower bound       1.562367394%
generic Steiner lower bound     0.781183697%
registered final allowance      1.185185185%
```

The result is inconclusive for Steiner vertices. It neither promotes the class
nor proves it cheap.

A second exact bound is available in a Hamming graph. If coordinate `j` uses
`d_j` distinct terminal symbols, every connected tree containing the terminals
must perform at least `d_j - 1` changes in that coordinate. Hence

```text
SMT(S) >= sum_j (d_j - 1).
```

For 4-bit coefficients `d_j <= 16`, so this certificate can be far below dense
work on wide matrices. It is a valid cheap certificate but not a universal
route to the target.

## Distributional counterevidence for synthetic trees

Jiang, Miller, and Pritikin prove that for almost all moderately sized sets of
`k` random vertices in the binary hypercube `Q_n`, the optimal Steiner cost is
`(1/3 +/- epsilon) k n`; they also give a near-matching randomized construction
in `O(k n)` time. Thus arbitrary Steiner vertices typically buy a constant
factor, not the roughly `84.4x` reduction required by `1.185185185%`.

The binary result also gives a legitimate lower-bound route for 4-bit Hamming
terminals. Map each 4-bit symbol through any binary predicate. Projecting a
4-bit Steiner tree yields a connected binary subgraph after zero-cost edges
are contracted, so its binary cost cannot exceed the original cost. A balanced
projection can therefore expose a hard terminal set without assuming that
synthetic nodes remain 4-bit terminals.

This is distributional and asymptotic evidence. It is not a finite certificate
for the pinned real checkpoint, and no such claim is made here.

Source:

- Tao Jiang, Zevi Miller, and Dan Pritikin, *Near Optimal Bounds for Steiner
  Trees in the Hypercube*, SIAM Journal on Computing 40(5), 2011,
  <https://sites.miamioh.edu/jiangt/files/2021/12/Steiner-SIAM.pdf>.

## Counterevidence for general static linear circuits

Synthetic DAGs are more general than trees, so tree lower bounds cannot close
them. The relevant exact reference point is linear-circuit complexity. Boyar
and Find summarize Lupanov's `(1 + o(1)) n^2 / log n` upper bound for every
binary `n x n` matrix and give the counting result that almost every such
matrix needs

```text
n^2 / (4 log_2 n) - o(n^2 / log n)
```

fan-in-two XOR gates even when cancellation is allowed. The leading-term gate
fractions are `1.785714%` at `n = 16,384` and `1.592312%` at `n = 53,248`, both
above the registered allowance.

Those two percentages are diagnostics, not finite target certificates: the
unspecified lower-order term matters, the theorem is for square binary maps,
and a released Transformer matrix is not sampled uniformly. Nevertheless it
blocks any universal claim based only on “general shared linear forms.” An
exact scalar add/subtract/integer-constant-multiply circuit for integer MatVec
reduces modulo two to a binary linear circuit on binary coefficients;
arbitrary static sharing in that model therefore has hard instances unless the
proposal exploits additional checkpoint or activation structure.

Source:

- Joan Boyar and Magnus Gausdal Find, *Cancellation-free circuits: An approach
  for proving superlinear lower bounds for linear Boolean operators*, 2012,
  <https://arxiv.org/abs/1207.5321>.

The published terminal-tree route used by EXP-082A remains documented in
Anand, van den Brand, and McCarty, *The Structural Complexity of Matrix-Vector
Multiplication*, <https://arxiv.org/html/2502.21240v3>. Its low-weight-tree
guarantee applies only when the stated structural dimension is favorable; it
does not imply that arbitrary dense matrices have low-weight trees.

## Fully charged cold-backed boundary

Cold placement removes the 8 GiB requirement from the complete program, but
not from live state and not from per-query work or traffic.

For a static circuit with `s` executed scalar coefficient/gate contributions
per query and dense coefficient count `P`, a necessary favorable condition is

```text
operation_fraction = s / P <= 0.011851851851851851.
```

Every executed gate must also obtain its operands, opcode, constants or delta,
and destination. Unless resident, those program bytes are cold traffic and
must share the registered `2.235174179 GiB/token` allowance with weight data,
metadata, verification, misses, and fallback. Intermediate values, the KV
cache, model-resident state, runtime buffers, and the interpreter must jointly
fit the 8 GiB hot cap.

The complete lossless backing on which an exact static program depends must
distinguish the arbitrary matrix. If the program is self-contained, its
artifact carries that information; if it references original weights, the
checkpoint carries it. EXP-072A therefore still requires at least
`1,623,396,974,592` bits (`188.98828125 GiB`) in the worst-case total lossless
backing before opcodes or indexes. Cold storage can make this a capacity rather
than hot-residency problem, but the currently recorded `97.618 GiB` filesystem
cannot hold even that favorable backing. A later disk upgrade would remove
only this local capacity obstacle; it would not prove query-time operations,
traffic, or latency.

Offline compilation may amortize discovery, but it cannot erase program
storage or the instructions and deltas used by each token. Conversely, if only
a query-dependent subprogram is read, the mechanism must explain how the
current `x` identifies it without a dense scan. That causal information source
is the remaining research object.

## E0 scorecard

| Gate | Finding | Verdict |
|---|---|---|
| Exactness | A synthetic tree/DAG can reproduce `W x` exactly | possible, not sufficient |
| Novelty | Static synthetic trees are contained by archived EXP-072B | fail |
| Terminal-tree lower bound | Generic Steiner consequence is only `0.781183697%` | inconclusive |
| Random-tree upside | Almost-all hypercube cost is near one third of dense | adverse evidence |
| General static sharing | Almost-all binary matrices need `Theta(n^2/log n)` gates | adverse, not a real-weight certificate |
| Self-contained hot storage | `188.98828125 GiB` information versus 8 GiB | fail by EXP-072A |
| Cold static storage | capacity can move cold; current disk is too small | unresolved after upgrade |
| Per-query operations and traffic | no sub-`1.185185185%` static schedule or encoding | fail to close |
| Causal query-dependent selection | no source specified yet | open E0 question |

## Consequence for the next ticket

Do not synthesize a Steiner tree or reopen EXP-072B under a cold-storage name.
The next resource equation must be for a **query-adaptive cold-backed
executor**, not a static circuit: a causal selector returns a small set of cold
pages or operations for the observed `x`, exact verification fails closed, and
all selector/probe/miss/fallback/state costs are included. Until both that
equation and an information source exist, the authoritative state remains
`NO_SURVIVING_CANDIDATE`.

## Claim boundary

All numeric transformations in this note are proofs or arithmetic over already
registered evidence. No model weights, GPU, CUDA path, private Ubuntu server,
large checkpoint, or physical storage/bandwidth path was executed. This is an
E0 class audit, not E1 evidence and not E2-E7 success.
