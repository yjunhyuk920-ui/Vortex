# E0 Sparse Functional Dictionary Frontier

## Verdict

```text
REJECT_FULL_BILINEAR_EVALUATION_TABLE
REJECT_DIRECT_LOCAL_RANK_ONE_TRUTH_TABLE
REJECT_STANDARD_COORDINATE_LDC_NAME_AS_CONSTRUCTION
REJECT_QUERY_CARDINALITY AS A CONSTRUCTOR OR TARGET LOWER BOUND
KEEP_ALIGNED_SPARSE_FUNCTIONAL_DICTIONARY AS AN UNCONSTRUCTED INTERFACE
NO SURVIVING CANDIDATE
```

This audit found a materially precise open interface, not a VORTEX component.
No model, checkpoint, backend, download, server, or hardware action occurred.

## Elementary explanation

Imagine that the checkpoint is a very large box of switches.  A question asks
for the parity of a special rank-one pattern of switches.

There are three ways to prepare answers.

1. Store the answer to every possible question.  The answer book has more than
   `2^39,254,527` pages.  It cannot be addressed or stored.
2. Divide the box into small squares and store every answer inside each square.
   A square large enough to reduce work to the registered compute allowance
   expands even the favorable one-bit checkpoint to about `480.37 TiB`.
3. Replace the checkpoint by a reversible dictionary of mixed switch groups.
   If every requested pattern were the XOR of a few dictionary groups, reading
   those groups would give the exact answer.

The third idea is the only one that survives the first information-count Gate.
But we do not have the dictionary or the fast rule that finds its groups.  A
random dictionary has many possible short combinations, but those combinations
are almost never the particular rank-one patterns that are needed.  "There are
enough keys" is not the same as "the keys open these locks."

## 1. Exact candidate interface

Reduce to the favorable binary subclass.  Let `W in F_2^D` be the checkpoint
and let the compiler choose dictionary atoms `g_1,...,g_S in F_2^D`.  It stores

```text
c_j = <g_j, W>.
```

For every supported query mask `q`, the decoder must produce a sparse set
`T(q)` satisfying

```text
q = XOR(j in T(q), g_j).                                 (1)
```

Then the exact answer is

```text
<q,W> = XOR(j in T(q), c_j).                             (2)
```

Unlike F-055 and F-072, the cold source may be completely re-encoded and
non-systematic.  It need not retain raw coordinates plus a hot subspace.  This
is also why the interface is not yet an implementation: it must provide an
implicit atom layout, a sparse exact decoder, finite addresses, and a native
bounded-word lift.  Materializing `S` arbitrary `D`-bit atom masks would itself
be impossible.

The favorable binary storage grant is

```text
D registered parameter bits           405,849,243,648
complete 8 GiB extra bits               68,719,476,736
S dictionary form bits                 474,568,720,384
64-bit words                             7,415,136,256.
```

This grants the whole extra state to a single bit plane and ignores native
summary widths, metadata, decoding, arithmetic, and physical random reads.

## 2. The independently rank-one tuple count

One binary `m x n` block has exactly

```text
1 + (2^m - 1)(2^n - 1)
```

distinct rank-at-most-one masks.  Across the frozen 884 disjoint matrices, an
independently selectable tuple has a pair-description upper exponent

```text
L = sum_i (m_i+n_i) = 39,254,528.
```

Every side has length at least 1,024.  After normalizing each factor by
`2^(m_i+n_i)`, the total multiplicative deficit is less than
`884 * 2^-1023 < 1/2`.  Therefore the exact integer floor is

```text
floor(log2 |Q_tuple|) = 39,254,527.                       (3)
```

This proves the literal all-answer table is impossible.  It does **not** prove
that an ordinary causal Transformer trace realizes all independent tuples.
The tuple is a stronger audit interface used to test the proposed universal
dictionary.

## 3. Direct local truth tables fail storage before execution

Tile a binary matrix into `b x b` squares.  A tile has

```text
(2^b - 1)^2
```

nonzero rank-one masks.  Storing their exact parity answers lets a query read
one bit per tile, so its favorable query fraction is `1/b^2`.  The persistent
one-bit storage ratio is

```text
(2^b - 1)^2 / b^2.                                       (4)
```

The smallest nontrivial case, `b=2`, already stores 9 forms for 4 source bits,
a `2.25x` expansion, while still reading one quarter of the source.  Meeting
the registered coefficient allowance needs only the favorable `b=10` case:

```text
query fraction                          1/100 = 1%
forms per tile                          1,046,529
storage ratio                           10,465.29x
model-wide favorable one-bit storage    480.3654 TiB.
```

This ignores tile boundaries, word widths, native numerical values, and every
other cost.  Direct Four-Russians-style answer tables are therefore rejected.
Compressing that redundant table back near the source entropy while retaining
local access is exactly the missing locally decodable representation; it
cannot be assumed for free.

## 4. Why cardinality does not reject a global sparse dictionary

With `S` stored one-bit forms, at most `t` selected forms have no more than

```text
sum(j<=t,C(S,j)) < (3S/t)^t                              (5)
```

possible selections.  Equation (5) remains smaller than the strict query-count
lower bound through

```text
t = 2,020,681 bit-form probes.
```

At `2,020,682`, this particular counting argument stops rejecting.  If a
64-bit read grants all `2^64` internal linear combinations, the corresponding
boundary is only

```text
largest count-rejected value            494,025 words
first value not rejected                 494,026 words
payload at that boundary                 3,952,208 bytes.
```

These are not algorithms and not latency estimates.

The converse capacity check is also important.  The elementary bound

```text
C(S,t) >= (S/t)^t
```

reaches the complete query-name upper bound at `2,216,796` bit subsets.  A
random binary dictionary can simultaneously avoid every dependency of weight
at most `4,433,592`: the union-bound logarithm is only
`81,102,563.35`, far below the `405,849,243,648` data dimensions.  The
64-bit favorable analogue reaches capacity at `510,961` words; its dependency
union exponent is `942,453,087.82`, again far below the data dimension.

Full row rank can be imposed at the same time.  For a random `D x S` binary
matrix, a union bound over nonzero left-kernel vectors puts rank-deficiency
probability below `2^(D-S)`.  Adding this to the small-dependency probability
is still strictly below one by tens of billions of exponent bits.

Thus a full-rank, high-girth dictionary can have enough distinct short sums.
This proves only that **counting the sums cannot settle the problem**.  It does
not prove that those sums contain the rank-one set.  Alignment is the hard
condition:

```text
for every rank-one q, find |T(q)| small with q=XOR(T(q)). (6)
```

## 5. Literature boundary

Ramamoorthy and Rashtchian define exactly the non-systematic binary linear
model: store `s>n` linear functions and answer a query from `t` selected stored
bits.  They prove finite rigidity bounds for rank-one binary queries, but the
known bounds are far below a dense-fraction target.  They also record both a
polynomial-space `O(n/log n)` query upper bound and the open linear-space gap;
for the systematic model, they explicitly state that it is unknown whether a
linear-dimensional subspace puts every rank-one query at `o(n)` distance.

Those results identify the frontier; they do not supply VORTEX's near-linear
layout, native arithmetic, or physical decoder.  The source audited here is:

- S. N. Ramamoorthy and C. Rashtchian, *Equivalence of Systematic Linear Data
  Structures and Matrix Rigidity*, ITCS 2020:
  https://doi.org/10.4230/LIPIcs.ITCS.2020.35

Standard locally decodable codes recover declared message coordinates.  Merely
renaming the desired bilinear function as a coordinate either creates the
rejected exponential answer table or leaves the recovery-set construction
unspecified.  A genuine functional code with equation (1) is the sparse
dictionary problem itself, not evidence that it has been solved.

## 6. Missing obligations

Before this interface can become a candidate, one construction must supply all
of the following:

1. an explicit or succinctly generated dictionary fitting the persistent
   storage contract;
2. a proof that every legal rank-one/native query has a sufficiently sparse
   representation;
3. a sub-dense algorithm that finds `T(q)` without scanning atom metadata;
4. native Q4/BF16/FP32 value width, multiplication, reduction, and rounding
   order identical to the reference implementation;
5. physical word/page addresses and measured random-read traffic;
6. the 405B/32-token equation including state, scheduler, verification,
   fallback, and nonlinear downstream work.

None is currently supplied.  Therefore `OMEGA-FUNCDICT` is not admitted as a
runtime component, and `NO_SURVIVING_CANDIDATE` remains authoritative.

## 7. Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_sparse_functional_dictionary_frontier.py `
  --output-dir results\e0_sparse_functional_dictionary_frontier

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_sparse_functional_dictionary_frontier -v
```

Authority:

```text
results/e0_sparse_functional_dictionary_frontier/summary.json
results/e0_sparse_functional_dictionary_frontier/checksums.sha256
```
