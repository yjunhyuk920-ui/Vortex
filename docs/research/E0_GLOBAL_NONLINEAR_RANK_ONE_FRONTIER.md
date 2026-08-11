# E0 Global Nonlinear Rank-One Frontier Audit

## Verdict

```text
REJECT_BOOLEAN_ZERO_RECTANGLE_AS_AN_EXACT_NUMERICAL_SOURCE
REJECT_KPI25_LIMITED_INDEPENDENCE_AS_A_RANK_ONE_TARGET_BOUND
REJECT_WHOLE_MATVEC_LOWER_BOUNDS_AS_A_2.5%-SCALAR_BOUND
KEEP_GENERAL_NONLINEAR_NUMERICAL_RANK_ONE_GAP_OPEN
KEEP_UNIVERSAL_2.5% NOT ESTABLISHED
KEEP_NO_SURVIVING_CANDIDATE
```

This audit does not prove that universal 2.5% is impossible. It also does not
construct it. It closes two of the strongest currently visible routes without
changing the user's single-stream, reference-exact, 8 GiB, fully charged
contract.

## Elementary explanation

The Boolean construction asks a very easy kind of question:

> Is there at least one red marble in this box?

If a stored label says that a large box contains no red marbles, every smaller
box inside it is also empty. One label can therefore settle many later
questions.

The Transformer needs a different answer:

> Add every positive and negative number in this box. What is the exact total?

Finding one nonzero number does not settle that question. `+1` and `-1` can
cancel. A label that answers the exact total for every possible one-cell box
must reveal every cell, because asking about a one-cell box returns that cell.
That label is therefore at least as informative as the original block.

The latest limited-independence lower bound fails for another simple reason.
Three rank-one questions can be chosen so that the third is always the XOR of
the first two. Three answers tied by that rule cannot behave like three
independent coins. The theorem requires much more independence, so its premise
is false for the complete rank-one query family.

## 1. The claim that actually needs proof

Let `W` be any unchanged supported dense checkpoint, `q=(r,u)` any legal
causal decision query, and `S_c(W)` the charged checkpoint representation.
A universal 2.5% theorem must exhibit one uniform compiler and decoder such
that

```text
for every W and q:
    Decode(Compile(W), W, q) = reference_exact(r^T W u)
```

and, for every accepted 32-token block,

```text
charged physical checkpoint bytes
+ advice/metadata bytes
+ address and scheduler bytes
+ state exactization bytes
+ nonlinear repropagation bytes
+ miss and fallback bytes
    <= S_c(W) / 40,

hot accelerator state <= 8 GiB,
steady-state single-stream service <= 20 ms/token.
```

An average-case margin, a selected checkpoint class, a Boolean answer, free
query computation, a hidden big-O constant, cross-user throughput, or an
uncharged fallback does not prove this statement.

## 2. Larsen--Williams is genuinely global and nonlinear

Larsen and Williams, Theorem 2.3, preprocess an arbitrary Boolean
`n x n` matrix and answer a Boolean vector-matrix-vector query with

```text
O(n^(3/2) / sqrt(w)) word probes
```

while storing the original matrix plus

```text
O(n^(3/2) * sqrt(w)) redundant bits.
```

The constructor greedily stores large all-zero rectangles. For a query
rectangle `U x V`, it removes the portions covered by those rectangles and
forms the remaining set `Q`.

- If `Q` is small, it probes every entry in `Q`. Covered entries are already
  known to be zero.
- If `Q` is large, preprocessing maximality proves that `U x V` cannot be an
  all-zero rectangle, so the Boolean answer is one.

This is an important audit target because it is globally nonlocal,
checkpoint-dependent, nonlinear, and data-dependent. It is not a local truth
table or a linear sketch.

### 2.1 Finite favorable substitution

For the largest registered square, `n=16,384`, and a 64-bit word, the leading
monomials are

```text
one-bit matrix              268,435,456 bits = 32 MiB
n^(3/2) / sqrt(64)          262,144 word probes = 2 MiB
n^(3/2) * sqrt(64)          16,777,216 bits     = 2 MiB
leading probe ratio         1/16 = 6.25%
leading redundancy ratio    1/16 = 6.25%
```

These are not concrete upper bounds: the theorem uses big-O notation and the
cell-probe model makes query computation free. The published statement also
does not provide a 32-query shared-probe bound. Even its favorable leading
single-query traffic is 2.5 times the requested 2.5% line for this one-bit
matrix.

That comparison alone is not a rejection: hidden constants could only make
the published guarantee less concrete, while a new multiquery construction
could in principle amortize work. The decisive mismatch is the output
semantics.

### 2.2 Why the zero-rectangle proof does not produce an exact sum

Over the Boolean semiring, `u^T A v` asks whether the selected rectangle has
at least one `1`. Exact numerical arithmetic asks for a parity, count, signed
sum, Q4 accumulation, BF16 result, or native rounded result. These are not
interchangeable. For example,

```text
[+1, -1] has nonempty support and exact sum 0
[+1, +1] has nonempty support and exact sum 2.
```

The Boolean data structure returns the same nonempty answer for both.

The deeper obstruction is heredity. Zero has the special property that every
intersection with an all-zero rectangle contributes exactly zero. An arbitrary
numeric rectangle has no one-value summary with that property.

### 2.3 Scoped exact-summary lemma

Let a block contain `m*n` symbols from an alphabet of size `A`. Suppose a
summary `sigma(B)` alone, without raw probes inside the covered block, returns
the exact aggregate for every subrectangle, and a singleton aggregate returns
its cell value.

For every position `(i,j)`, query the singleton rectangle `{i} x {j}`. Its
answer recovers `B[i,j]`. Therefore

```text
sigma(B) = sigma(B')  implies  B = B'.
```

The summary is injective and needs at least

```text
m*n*log2(A) bits,
```

which is the raw information content of the block.

This lemma rejects the direct lift “replace each all-zero rectangle by one
exact numerical rectangle summary.” It does not cover a global adaptive
scheme that probes some covered cells, couples many blocks, or uses a
different representation. It is deliberately not called a universal
cell-probe lower bound.

## 3. Why the 2025 local-PRG lower bound does not apply

Korten--Pitassi--Impagliazzo Theorem 1 applies when the columns of a Boolean
data-structure problem support a `k`-wise independent answer distribution. It
requires an even query time `t` and

```text
k > t*w + 1,
```

where `w` is the cell word length.

For the full distinct GF(2) rank-one query family, fix nonzero `r` and choose
distinct nonzero `u,v` with `u+v` nonzero. Define

```text
q1 = r tensor u
q2 = r tensor v
q3 = r tensor (u+v).
```

Then

```text
q1 XOR q2 XOR q3 = 0,
```

so for every possible matrix `W`, not merely on average,

```text
<q1,W> XOR <q2,W> XOR <q3,W> = 0.
```

The three-answer vector occupies at most four of its eight possible bit
patterns. No distribution over checkpoints can make these three coordinates
independent. Uniform `W` does make every pair of distinct nonzero linear
queries independent, so the maximum independence of the complete family is
exactly `k=2`.

The theorem's smallest legal even `t` is two. With 64-bit words it already
requires

```text
k > 2*64 + 1,
minimum integer k = 130,
```

while the query family has `k=2`. The premise is false. Restricting to an
independent basis removes the three-query dependency but reduces the problem
to coordinate retrieval; it supplies no lower bound for all rank-one queries.

This failure does not make the target easy. It says only that this theorem
cannot certify the required lower bound.

## 4. Whole matrix-vector lower bounds lose the needed factor

Clifford--Gronlund--Larsen prove strong static cell-probe lower bounds for
returning the entire vector `W*u` over sufficiently large finite fields. Our
remaining interface returns one scalar `r^T W u`.

A scalar oracle can reconstruct `W*u` by making `n` calls with
`r=e_1,...,e_n`. Consequently a whole-vector lower bound of `T` probes implies
only that at least one scalar call costs `ceil(T/n)` probes.

Even granting the idealized strongest possible premise `T=n^2` at
`n=16,384` yields only

```text
scalar lower bound = n = 16,384 words,
requested 2.5%     = n^2/40 = 6,710,886.4 words.
```

The valid reduction is more than 409 times too weak before accounting for the
published theorem's field, space, error, and word-size conditions. It cannot
resolve the scalar target.

## 5. The 2026 dynamic lower bound is a different model

The 2026 Multiphase result proves an
`Omega((log n/log log n)^2)` update-or-query lower bound for dynamic inner
product over GF(2). It includes an update phase and does not provide a static
checkpoint theorem with free 8 GiB global advice. Its scale is also
polylogarithmic, not `n^2/40`. It is relevant evidence about the known static
lower-bound barrier, not a VORTEX target certificate.

## 6. What remains open

The audit leaves exactly one unclosed interface:

```text
arbitrary unchanged bounded-word checkpoint
+ at most 8 GiB globally nonlocal nonlinear advice
+ adaptive data-dependent probes
+ exact native numerical rank-one output
+ fully charged finite computation and physical traffic.
```

Neither a target-fitting constructor nor a covering lower bound is known.
Therefore the only sound project classification remains

```text
PRIMARY RESEARCH AND PROTOTYPE TRACK / REVISE
UNIVERSAL 2.5%: NOT ESTABLISHED
GENERAL IMPOSSIBILITY: NOT PROVED
CORE CANDIDATE: NONE
```

## 7. Why batching is not the missing theorem

There are two different denominators that must not be confused. The registered
gate allows `1/40` of the checkpoint for the entire 32-token block, so its
weight allowance per certified token is

```text
(1/40) / 32 = 1/1280 = 0.078125% of the checkpoint.
```

A full weight sweep shared by 40 independent requests has arithmetic weight
traffic `1/40 = 2.5%` per output. That is 32 times the registered per-token
allowance, and its block read fraction remains `rho=1`, not `rho=1/40`.
Matching only the weight-I/O-per-output number would require 1,280 independent
outputs per full sweep. Even that would still change and fail the registered
block-fraction contract:

- it is cross-request throughput, not one stream's token cadence;
- each request still waits for a full sweep;
- it does not provide 40 causally consecutive future tokens of one stream;
- metadata, KV, state, compute, and shared-link bytes still have to be added.

Likewise, a speculative block gets `1/40` only if 40 future tokens are already
correct. An arbitrary checkpoint can reject the first proposal. Treating
perfect lookahead as free assumes the answer that the system is meant to
compute.

Thus batching is a possible alternative throughput study, not a proof of the
registered universal single-stream or 2.5%-per-block claim.

## 8. Reproduction

```powershell
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_global_nonlinear_rank_one_frontier.py `
  --output-dir results\e0_global_nonlinear_rank_one_frontier

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_global_nonlinear_rank_one_frontier -v
```

The derivation uses only deterministic integer arithmetic and performs zero
model forwards and zero hardware actions.

## Primary sources

- Larsen and Williams, *Faster Online Matrix-Vector Multiplication*:
  https://arxiv.org/abs/1605.01695
- Korten, Pitassi, and Impagliazzo, *Stronger Cell Probe Lower Bounds via
  Local PRGs* (revision 1):
  https://eccc.weizmann.ac.il/report/2025/030/
- Clifford, Gronlund, and Larsen, *New Unconditional Hardness Results for
  Dynamic and Online Problems*: https://arxiv.org/abs/1504.01836
- Ko, *An Omega((log n/log log n)^2) Cell-Probe Lower Bound for Dynamic
  Boolean Data Structures*: https://eccc.weizmann.ac.il/report/2026/047/
