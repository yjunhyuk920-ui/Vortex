# Implicit program-carrier frontier — 2026-09-09

## Decision

```text
CONSTRUCT_EXACT_GF2_ADDRESS_ONLY_ALIAS_ROUTER
REJECT_EXPLICIT_ALIAS_DESCRIPTOR_REALIZATION_AS_TARGET CORE

CONSTRUCT_EXACT_GF2_QUERY_TIME_PATRICIA_PATTERN_SYNTHESIZER
REJECT_WORD_LABEL_PATRICIA_REALIZATION_AS TARGET CORE

CONSTRUCT_ARBITRARY_GF2_RANK_NORMAL_STATE_COMPILER
CONSTRUCT_AND_PAY_LITERAL_TRANSFORMED_HADAMARD
REJECT_PER_OPERATOR_RANK_NORMAL_PLUS_LITERAL_HADAMARD AS TARGET CORE

KEEP_GENERAL_NONLINEAR_IMPLICIT_PROGRAM SOURCE OPEN
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
```

The fixed VORTEX mission and O1--O6 contract remain unchanged. This round starts
from clean remote head
`925949770cfb231ea1d79b6609023e5e18478440`. All three principles and their
falsifiers were preregistered and pushed before result inspection at:

```text
52facca89659224b8a101f75601ec275740c3e2f
```

Canonical E0 result:

```text
results/e0_implicit_program_carrier_gate/summary.json
SHA-256 f6f911fce9383ab6b82a1ac4ad5ef79f354a8ca58d79d21ff71193000c8d5070
```

The focused new suite passed `11/11` at the current source state. None of the
previous completed suites was rerun.

The round does **not** produce the arbitrary-native 405B executor. Its useful
advance is narrower: three places that could have hidden the checkpoint-dependent
program are now executable finite objects rather than labels. In all three
explicit realizations, the apparent dense-arithmetic win is paid back by routing
metadata, checkpoint edge labels, or a transformed nonlinear operator.

## 1. Frozen binary control

The first necessary-condition control is exact GF(2) MatVec. On the square
registered shape:

```text
m = n                              16,384
binary coefficients               268,435,456
favorable leaf+add slots           536,870,912
```

Across the registered 883 non-embedding matrices:

```text
binary source bits                 403,747,897,344
binary source GiB                   47.00244140625
```

Binary success is not native Q4/BF16/FP32 success. The historical `8/675` line
is a favorable logical comparison only; the final p50/p95/TTFT theorem remains
unproved.

## 2. Principle I — address-only alias router

### 2.1 Exact finite construction

The previous query-side image frame put checkpoint information in stored output
images. P1 instead makes the **query values checkpoint independent** and moves
checkpoint information entirely into the route.

Partition the current binary right factor into deterministic 10/11-bit blocks.
For each block `B`, runtime builds:

```text
Q[B,p] = <p, v_B> mod 2
```

for all local patterns `p`. This is a finite query-only subset-parity table.

The compiler stores one alias relation per row/block:

```text
A[i,B] -> Q[B, W[i,B]].
```

Runtime then follows a fixed logical schedule that knows only `(i,B)`:

```text
for i:
    y_i = XOR_B AliasRead(A[i,B]).
```

The exact relation is immediate:

```text
XOR_B AliasRead(A[i,B])
= XOR_B <W[i,B],v_B>
= <W_i,v>.
```

The implementation exhaustively matches direct GF(2) MatVec over the frozen
small controls. The checkpoint is not silently lost: it is exactly the compiled
logical-alias-to-pattern mapping.

### 2.2 Why it initially looks strong

For `16,384 x 16,384`:

```text
blocks                              1,638
query-table recurrence ops          1,679,770
logical alias reads                26,836,992
table + row-XOR ops                28,516,762
candidate / leaf+add baseline       0.053116608411073685
```

Thus the arithmetic-slot screen removes about `94.69%` before routing itself is
paid. This is not a fake arithmetic route.

### 2.3 The missing information did not disappear

For arbitrary `W`, concatenating all alias targets `W[i,B]` reconstructs every
coefficient bit. Therefore the lossless routing relation itself carries at least:

```text
mn bits
```

of checkpoint information. This is only an information statement about the
representation; it is **not** a proof that every nonlinear router must read all
those bits on every query.

The current concrete physical realization uses one 64-bit descriptor per alias.
For one `16,384` square matrix:

```text
descriptor / source ratio           6.3984375x
descriptor GiB                      0.199951171875
```

Across all 883 registered matrices:

```text
alias count                         40,365,964,800
64-bit descriptor bits              2,583,421,747,200
descriptor GiB                      300.74987411499023
descriptor / binary source          6.398601117664475x
all descriptors fit 8 GiB           NO
```

If those descriptors/page-table/routing entries are streamed/translated once per
logical alias, the route has simply replaced coefficient reads by a larger
program/routing stream. If an implementation claims a hardware translation
cache prevents those reads, the cache state, fills, misses and backing pages are
part of O4/O5 and must be accounted. The present E0 implementation therefore
fails its paid physical realization.

Claim boundary: this rejects the **explicit 64-bit descriptor alias carrier** and
the idea that routing metadata may be omitted from cost. It does not prove that
all possible nonlinear address encodings require one descriptor fetch per alias.

## 3. Principle II — query-time Patricia pattern synthesizer

### 3.1 Exact finite construction

P2 avoids both a complete image table and a flat row/block selector. For every
64-column block, collect the row patterns and build a deterministic compressed
Patricia trie of the **patterns that actually occur**.

Each compressed path segment stores the exact one-bits of its checkpoint label.
At query time a DFS carries a parity value. For an edge/segment mask `M_e`:

```text
parity_child = parity_parent XOR parity(M_e AND v_B).
```

At a leaf the accumulated parity equals the exact local dot product for every
row sharing that pattern. XORing leaf contributions across column blocks returns
exact `Wv`.

The implementation handles duplicate patterns, rectangular controls and every
small query exactly. There is no `2^64` query table.

### 3.2 Favorable event count

With `d_B` distinct patterns, a binary Patricia trie has at most `2d_B-2`
ordinary edges. For the maximally distinct `16,384` square control and 64-bit
blocks:

```text
blocks                              256
max edges/block                    32,766
max edge evaluations               8,388,096
row-block leaf contributions       4,194,304
edge + leaf events                12,582,400
event / leaf+add baseline          0.023436546325683594
```

So this representation has an even stronger nominal arithmetic route than P1.

### 3.3 Program traffic falsifier

The actual parity primitive needs the checkpoint-dependent edge label. Under the
preregistered favorable realization of one 64-bit mask word per edge:

```text
edge-label bits                     536,838,144
edge-label MiB                       63.99609375
edge-label / binary source           1.9998779296875x
```

This is **before** child/topology data and row memberships. The event count was
small because one word operation replaced many scalar bits; the checkpoint
program was not small. On an arbitrary high-distinctness pattern family, no
checkpoint compressibility may be assumed.

A future packed Patricia representation is not rejected by the 64-bit-word
number alone, but it must provide a worst-case arbitrary-matrix size and a
runtime decoder. The current explicit program returns to source-scale-or-worse
program traffic and is rejected as the core.

Claim boundary: this rejects the compiled Patricia realization, not all query-
time structural synthesis and not all succinct exact parity data structures.

## 4. Principle III — rank-normal encoded state with paid transformed nonlinearity

### 4.1 Arbitrary GF(2) rank-normal compiler

The previous safe-gauge theorem forced product-preserving encodings to be
coordinate permutations. P3 deliberately drops product preservation.

For **every** binary `m x n` matrix, deterministic row/column elimination builds
invertible `A,B` such that:

```text
A W B = J_r
```

where `J_r` contains one `r x r` identity block. Encode wires by:

```text
z_in  = B^-1 x
z_out = A y.
```

Then:

```text
z_out = J_r z_in.
```

The implementation verifies exact normal form on arbitrary small rectangular
matrices and exact encoded query equality. For a full-rank square map, isolated
linear coefficient effects drop from `n^2` to `n`, i.e. `1/n`.

### 4.2 This time the nonlinear operator is not free

For an encoded nonlinear node, the exact program is explicitly:

```text
N_enc(z) = E_out N(E_in^-1 z).
```

The strongest minimal adversary uses two independently dense invertible maps
meeting at AND:

```text
a = W1 x
b = W2 x
c = a AND b.
```

Choose the maximally favorable encodings that trivialize both projections:

```text
z1 = W1^-1 a = x
z2 = W2^-1 b = x.
```

Then the exact transformed product is not coordinatewise AND on `z1,z2`; it is:

```text
c = (W1 z1) AND (W2 z2).
```

The implemented literal factorized transformed nonlinearity therefore computes
the two dense maps that the rank-normal projections removed. Exhaustive 4-bit
controls match the native-coordinate result exactly.

At `n=16,384` under the frozen scalar-slot ledger:

```text
isolated linear effect fraction            1/16384
isolated >=90% route                       YES

baseline two dense maps + AND slots        1,073,758,208
rank-normal + literal transformed AND      1,073,790,976
whole micrograph fraction                   1.0000305171124708
whole micrograph >=90% route               NO
```

Thus the specific “rank-normalize every operator independently, then literally
conjugate the nonlinearity” implementation moves rather than removes dense work.

Claim boundary: this is not a theorem against a globally co-designed nonlinear
state representation. An alternative direct bilinear/tensor implementation is
open only if its constructor, tensor/circuit/program storage and runtime traffic
are explicit and cheaper.

### 4.3 Concurrent native-gauge refinement

While this preregistered round was executing, the same remote research branch
received an independently constructed native-gauge package at
`experiments/codex_gauge_transport_20260909/REPORT.md`. It sharpens, rather than
invalidates, the claim boundary above.

Its opaque-word butterfly is an explicit finite non-coordinatewise encoding
with exact transformed native multiplication:

```text
E(native_mul(E^-1 z, E^-1 w))
```

using `3*n*log2(n)/2` XORs plus `n` original native products for already encoded
inputs/output. Therefore **no quadratic lower bound for every transformed
elementwise gate is valid**. The failure proved in this round is only the chosen
per-operator rank-normal gauge plus its literal factorized transformed AND.

The concurrent package also shows that native coordinate permutations must carry
the original logical leaf/reduction schedule; algebraically permuting columns and
then reducing in new physical order can change finite-word results.

Crucially, that butterfly construction still has no cheap arbitrary dense
projection `E F_W E^-1`: decoding, executing `F_W` in full, and re-encoding keeps
all original dense coefficient work. The combined frontier is thus the joint
construction of a checkpoint-dependent `E` for which **both** transformed graph
gates and the arbitrary dense projection are exact and target-cheap.

## 5. Candidate comparison

| Principle | Arbitrary finite constructor | Exact control | Nominal >=90% route | Paid result |
|---|---|---|---|---|
| P1 address-only alias router | yes, arbitrary GF(2) | exhaustive small MatVec exact | 5.31% arithmetic slots | explicit descriptor carrier 300.75 GiB model-wide; rejected |
| P2 Patricia synthesizer | yes, arbitrary GF(2) | exhaustive small MatVec exact | 2.34% event fraction | edge labels alone ~2.0x source in frozen realization; rejected |
| P3 rank-normal encoded state | yes, arbitrary rectangular GF(2) | rank normal + transformed AND exact | isolated dense maps `1/n` | literal nonlinear conjugation restores both dense maps; rejected |

All three candidates reverse different information-flow assumptions, and all
three were actually constructed. No one supplies the arbitrary-native producer.

## 6. Paid cost ledger

| Cost | P1 alias | P2 Patricia | P3 rank-normal + transformed nonlinear |
|---|---|---|---|
| Constructor | finite block-pattern alias compiler | finite per-block compressed trie build | deterministic row/column elimination + transformed-node compiler |
| Persistent representation | lossless alias relation; explicit descriptor layout | trie topology + edge masks + leaf memberships | A/B/J or executable programs + every conjugated graph operator |
| Storage | info lower >= source; explicit model descriptor 300.75 GiB | 64-bit edge masks alone ~=2x source on square worst-edge realization | dense gauges/programs plus transformed nonlinear programs |
| Query arithmetic | ~5.31% leaf+add slots on square control | ~2.34% edge+leaf event fraction | isolated linear maps `1/n`, but micrograph >100% baseline |
| Address/routing | fixed logical aliases but translations fully paid | deterministic DFS; child/topology reads paid | transform/nonlinear program addresses paid |
| Runtime traffic | descriptor/page-routing stream if not resident; no free TLB | all used edge labels + topology/memberships | dense transform programs/effects restored at nonlinear nodes |
| GPU/workspace | query subset tables + row accumulators + routing cache + KV | trie traversal/query word/row accumulators + KV | encoded wires, transform workspace, KV/cache encodings |
| Verification/fallback | none free; dense fallback destroys fast-core route | none free | exact literal program needs no repair but is not fast |
| Native lift | absent | absent | binary graph control only; arbitrary native conjugation absent |
| Whole causal state/RNG | absent | absent | algebraic encoding induction only; native causal implementation absent |

Expanded cold storage is allowed only when its bytes, addresses and query reads
are charged. No metadata carrier is treated as physical free information.

## 7. O1--O6

### O1 — OPEN

All three constructors are uniform on their declared GF(2) domains, but no
arbitrary native Q4/BF16/FP32 checkpoint has a target-cheap implicit program
carrier and executor.

### O2 — OPEN

No surviving program carrier is integrated into complete loading, prefill,
decode, attention/MLP, KV/cache and sampling for an arbitrary checkpoint.

### O3 — OPEN

The exact controls establish GF(2) operator equality and the algebra of encoded
state only. They do not establish native finite-word ordered reductions, logits,
KV/cache, RNG consumption/state or all successor state for all legal continuations.

### O4 — OPEN

The explicit candidate costs are finite and adverse. No surviving whole-model
upper bound closes representation, routing/program traffic, arithmetic, state,
initialization and fallback simultaneously.

### O5 — OPEN

No 405B checkpoint, CUDA, single-GPU <=8 GiB run, PCIe/SSD/HBM schedule,
same-machine native4BQ4 p50/p95 or TTFT run occurred.

### O6 — PARTIAL

Preregistration, source, focused tests, deterministic result JSON/checksum and
this report are independently checkable. Whole-theory O6 remains incomplete
while O1--O5 remain open.

## 8. Validation

New focused suite only:

```text
11/11 PASS
```

Canonical result:

```text
results/e0_implicit_program_carrier_gate/summary.json
SHA-256 f6f911fce9383ab6b82a1ac4ad5ef79f354a8ca58d79d21ff71193000c8d5070
```

The previous implicit 15/15, direct-global 10/10, nonlinear 15/15,
nonlinear/geometry 28/28, causal-global 14/14, Boolean exhaustive controls and
native-global 3,510-file replay were deliberately not rerun.

## 9. Literature boundary

The current literature still does not provide the missing arbitrary-dense exact
upper source. Anand--van den Brand--McCarty obtain subquadratic preprocessed
MatVec for structured/low-VC matrices, not all arbitrary dense matrices:
`https://arxiv.org/abs/2502.21240`.

Ko's 2025 lower bound explicitly allows nonlinear preprocessing for a static
linear-operator problem and strengthens evidence that nonlinear advice is not
automatically free, but its asymptotic random-operator statement is not promoted
into the finite global-side-data/casual VORTEX theorem:
`https://arxiv.org/abs/2509.02730`.

The older exact static-circuit/Lupanov boundary remains F-050. P1/P2 were
specifically attempts to give that circuit family a new **implicit program
carrier**; both explicit carriers fail their paid metadata/program traffic.

## 10. Next constructive gap

The remaining source must do more than move arbitrary checkpoint bits into:

```text
payload values
address translation metadata
query-time structural edge labels
or a transformed nonlinear operator that reconstructs the same dense maps.
```

The next genuinely different round should attack how **multiple exact query
answers share checkpoint information over time or across graph cuts without
requiring source-scale program state on each token**, while preserving dense
legal right-factor changes and charging any dynamic state. No old response-cache
or sparse-delta assumption may be revived.
