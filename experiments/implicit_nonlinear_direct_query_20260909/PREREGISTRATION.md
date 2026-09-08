# Implicit nonlinear direct-query representation — preregistration

Date: 2026-09-09 Asia/Seoul

Base scientific head:

```text
9fad90e91bccc026db5dacd6286e6c17d58f9bc1
research/nonlinear-router-frontier-20260908
```

This round keeps the fixed mission and `docs/CONSTRUCTIVE_THEORY_CONTRACT.md`
unchanged.  The target object is a finite automatic compiler for every legal
unchanged checkpoint whose runtime does **not** reconstruct most checkpoint
words and does **not** fetch a checkpoint-dependent program at roughly one bit
per original weight:

```text
G = Compile(checkpoint bytes)
addresses = Address(G, current causal input/state, previous returned words)
payload = Read(G, addresses)
(effects, state') = Decode(payload, current input/state)
```

`effects` must be sufficient to reproduce the original finite-word logits and
every required successor-state field; the same sampler must then consume the
same target RNG words.  All construction, persistent storage, RAM/SSD,
PCIe/HBM, GPU allocation/workspace, addresses, metadata, arithmetic,
verification, repair/fallback, KV/state and initialization costs are paid.

The prior completed `10/10`, `15/15`, `28/28`, `14/14`, Boolean exhaustive and
3,510-file native-global validations are not part of this round and will not be
rerun merely for reassurance.

## Intended final theorem

The round succeeds only if it constructs one uniform finite algorithm such that
for every checkpoint and legal continuation

```text
Decode(Query(Compile(W), q_t, s_t))
    = ReferenceDenseEffects(W, q_t, s_t)

s'_(t+1) = E(ReferenceState_(t+1))
RNG'_(t+1) = RNGReference_(t+1)
```

inductively from equal initialization, while its fully paid worst-case route
removes at least 90% of the original dense parameter-effect work/traffic and
admits sufficient whole-model upper bounds.  A binary-only fast arm with a
dense native fallback is exact but is a **failed core** unless the universal
paid bound also passes.

## Frozen common population and first Gate

For binary necessary-condition calculations use the already registered 405B
non-embedding population (`403,747,897,344` coefficient bits, 883 matrices).
The `>=90%` criterion is a core-entry screen, not final p50/p95 proof.  Target
hardware quantities remain `NOT TESTED` unless actually executed.

## Principle I — sparse image frame with direct query addresses

### Reversed premise

Do not encode the checkpoint into a static instruction stream.  Encode a
checkpoint-independent family of **query atoms** once, compile only their
checkpoint images, and let the current right factor itself determine the read
addresses.

### Finite candidate

For a binary `m x n` matrix, partition input coordinates into blocks of width
10 or 11.  If `n=10q+r`, use `r` blocks of width 11 and `q-r` blocks of width
10 (the registered dimensions satisfy `q>=r`).  For every block `B` and every
nonzero binary pattern `p` on that block, compile

```text
Y[B,p] = W_B p  in F2^m.
```

The representation stores packed `Y[B,p]` vectors.  The address rule parses
the current binary right factor block-by-block; the block word itself is the
address.  Zero blocks perform no read.  The decoder XORs the selected images:

```text
Wv = XOR_B Y[B, v_B].
```

There is no checkpoint-dependent selector/program fetch.  Constructor and
decoder loops are finite.  Naive constructor upper bound is

```text
sum_B (2^|B|-1) * m * |B|  bit operations,
```

and persistent transformed payload is

```text
m * sum_B (2^|B|-1) bits
```

before addresses/alignment.  Runtime source payload is at most
`m*floor(n/10)` bits, so the binary path has a concrete <=10% source-effect
route.

For arbitrary native checkpoints the candidate is fail-closed: unless an exact
native superposition law is proved, it executes the original reference dense
operator and charges that fallback.  This gives a universal exact algorithm,
but not a universal fast algorithm.

### Exactness/state equation

GF(2) arm exactness is the displayed block decomposition.  On the native
fallback arm, the original operator is executed in the original finite-word
order.  Therefore equal incoming state/RNG gives equal effects and successor
state/RNG.  No fallback cost is free.

### Paid terms

Charge transformed image construction/storage, original checkpoint retention,
block address arithmetic, all image words read, output XORs, CPU/RAM/SSD,
PCIe/HBM/GPU workspace, native fallback, KV/state and any verification.

### Strongest adversary and cheapest falsifier

Use an arbitrary native finite-word projection whose reference result is not
linear under decomposition of the right factor because product/reduction
rounding is observable.  The already established zero-exact-sum FP32 rounding
gadget is admissible prior evidence and will not be rerun.  Also compare the
explicit transformed storage against the raw binary matrix; if the code simply
recreates the closed Mailman/Four-Russians storage explosion, stop it rather
than tune block widths.

## Principle II — value-class inverted native executor

### Reversed premise

Do not evaluate rows independently.  Invert the matrix by input coordinate and
finite weight value, reuse one exact product for every row carrying the same
weight word, while preserving each row's original reduction order.

### Finite candidate

For each column `j`, compile exact equivalence classes of rows by the complete
finite coefficient descriptor used by the reference leaf product (weight bits,
scale/bias metadata when part of that leaf semantics).  Store each descriptor
and a finite row-membership list/bitset.

Runtime processes original leaf positions in reference order.  For each class
at position `j` it evaluates the exact reference leaf product once for the
current `x_j`, writes that identical leaf word to every member row, and then
executes the original per-row reduction tree/order.  If a backend leaf is not
referentially determined by the stored descriptor and `x_j`, the compiler
marks the operator for exact reference fallback.

### Exactness/state equation

Every row receives bit-identical leaf products in the same reduction slots and
the identical reduction operation/order.  Thus the dense effect is bit-identical.
Replacing every admitted dense operator in topological order preserves all
logits/KV/state; the unchanged sampler consumes the same RNG.  Fallback is
identical by definition and fully charged.

### Termination and paid upper bound

Sorting/hash-grouping every finite column terminates.  Worst case has `m`
classes per column, `mn` memberships, `mn` product evaluations and the original
`m(n-1)` reduction additions, plus class/index traffic.  Storage includes every
descriptor and membership, original/checkpoint backing, RAM/SSD, transfers,
output buffers, reduction workspace, metadata and fallback.

### >=90% route

The only route is large exact value multiplicity: product evaluations fall from
`mn` to `sum_j d_j` where `d_j` is the number of distinct leaf descriptors.
All reduction additions remain unless an independent exact mechanism removes
them.  Therefore a universal >=90% claim requires a bound that survives an
arbitrary high-distinctness checkpoint.

### Strongest adversary and cheapest falsifier

Use columns with all distinct legal finite weight words whenever the alphabet
permits it, and otherwise distribute values as evenly as possible.  Independently,
the unchanged `m(n-1)` row reductions give a cheap arithmetic floor.  Reject if
either floor exceeds 10%; do not build a kernel after that.

## Principle III — encoded-state gauge conjugation

### Reversed premise

Do not query a dense matrix in the reference coordinates.  Carry hidden states
in compiler-chosen encoded coordinates so a dense map is conjugated into a
simpler map; pay the transformed nonlinear/residual operators instead of
silently assuming they stay coordinatewise.

### Finite candidate

The universally native-safe arm uses finite coordinate permutations on each
wire.  The compiler enumerates/chooses permutations and rewrites producer and
consumer indices consistently.  A dense operator becomes

```text
W' = P_out W P_in^{-1}.
```

Elementwise finite-word functions are merely permuted, residual additions use
the same permutation on both operands, and cache/state layouts carry the
inverse mapping.  Therefore this arm is exact for arbitrary finite words.

The high-upside extension permits general invertible linear gauges over the
binary necessary-condition algebra to try to canonicalize `W`.  It is admitted
only if the compiler also gives an exact cheap transformed Hadamard/gating
operator; that operator may not be called free.

### Exactness/state equation

For a safe encoded wire `z=P h`, each rewritten operator satisfies

```text
z_out = P_out F(P_in^{-1} z_in).
```

Induction over the graph gives `z_t=P_t h_t`; decoding public logits/KV and
sampling therefore reproduces the original state/RNG exactly.  All permutation
and layout work is paid.

### Paid upper bound

Charge permutation construction/storage, transformed weight layout or original
backing, every read of `W'`, encoded KV/cache, gather/scatter, GPU workspace,
nonlinear conjugation, and native fallback.  The safe permutation arm has the
same number of arbitrary dense coefficients as the original.

### >=90% route, adversary and falsifier

The hoped route is to use a non-monomial gauge that makes dense maps sparse.
The strongest adversary is a dense SwiGLU/Hadamard-style node between unrelated
dense maps.  Over the GF(2) necessary-condition algebra, prove/check the
automorphism lemma: an invertible linear map preserving coordinatewise product

```text
E(x .* y) = E(x) .* E(y)
```

must permute the primitive coordinate idempotents.  If true, any gauge keeping
the nonlinear node coordinatewise is only a permutation and cannot sparsify an
arbitrary dense support pattern.  A dense gauge merely transfers the missing
producer into a dense transformed nonlinear operator and is rejected as an
undefined primitive.

## Frozen ordering

Principle I is selected first because it has an explicit direct-query address
rule and a genuine <=10% binary source-read path without checkpoint-dependent
program fetch.  If native exactness/storage closes it, immediately execute
Principle II and then Principle III.  A rejection theorem is not the mission
result; the round continues through all three unless an earlier candidate
actually closes O1--O5.

## O1--O6 before execution

```text
O1 OPEN
O2 OPEN
O3 OPEN
O4 OPEN
O5 OPEN
O6 PARTIAL only after reproducible artifacts exist
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
```
