# E0 Finite-Word Discontinuous Bilinear Source Audit

Status: COMPLETE / CONSTRUCTOR FAMILY REJECTED
Decision: NO_FINITE_WORD_CONSTRUCTOR_SURVIVES_E0_KEEP_GENERAL_RANK_ONE_CELL_PROBE_OPEN
Project classification: NO_SURVIVING_CANDIDATE

## Question

Can automatic preprocessing of an unchanged arbitrary dense checkpoint create
a bounded-word, non-exhaustive exact data structure for every Bilinear Cross
Residual

\[
f_W(r,u)=r^\mathsf{T}Wu
\]

while charging construction, persistent state, word probes, physical traffic,
addressing, arithmetic, verification, misses, and fallback?

Two ceilings are kept separate:

- the user's requested universal read ceiling is \(1/40=2.5\%\);
- the registered complete p50 allowance is tighter:
  \(8/675=1.185185\ldots\%\), because all other runtime costs must also fit.

A universal result may not assume a favorable checkpoint distribution,
activation margin, low rank, sparsity, compressibility, or repeated query.
Those can support a checkpoint-class theorem or measured fast path, but not
the word **arbitrary**.

## Decision

No examined finite-word constructor survives E0.

The strongest explicit constructor genuinely makes data-dependent bounded-word
probes and is exact on a favorable binary-parity subclass. It can make the
query-only traffic fit 2.5%, and even the tighter \(8/675\) ceiling. It does so
only by materializing an exponential answer table whose storage, construction,
and address width fail by enormous margins.

This is a rejection of the declared constructors, not a universal cell-probe
lower bound. A globally nonlocal nonlinear rank-one data structure outside
these constructions remains open. Consequently:

\[
\boxed{\text{universal 2.5% guarantee: NOT ESTABLISHED}}
\]

and not:

\[
\boxed{\text{universal 2.5% guarantee: impossible}}
\]

## Concrete constructor: Block Rank-One Truth Table

Take a binary matrix \(W\) and partition it into \(a\) by \(b\) blocks
\(W_{pq}\). For every block, compile the complete finite table

\[
T_{pq}[x,y]
=
x^\mathsf{T}W_{pq}y \pmod 2,
\qquad
x\in\{0,1\}^{a},\quad y\in\{0,1\}^{b}.
\]

For a query, split \(r\) and \(u\) into matching blocks and compute

\[
r^\mathsf{T}Wu
=
\bigoplus_{p,q}T_{pq}[r_p,u_q].
\]

This is not an exact-field circuit in disguise. The bit patterns
\((r_p,u_q)\) form a discontinuous address, and one word is probed for each
matrix block.

The implementation in
vortex_runtime/finite_word_bilinear_source.py compiles this layout and checks
it against direct GF(2) evaluation. The reference controls exhaust all query
pairs for 16 deterministic small matrices: 296/296 exact answers, with 57/57
basis-entry recoveries.

GF(2) parity is deliberately favorable:

- every table answer needs only one bit;
- no signed Q4/BF16 accumulation bits are charged;
- no floating-point rounding state is charged;
- one physical probe and one XOR per block are granted.

A Q4/BF16 table would require at least as much representation for this layout.
Failure therefore rejects this constructor before native arithmetic makes it
harder. It does not prove that every other finite-word constructor must use
the same layout.

## Complete cost equation

Let \(D\) be the number of matrix coefficients and compare with a Q4 artifact
of \(D/2\) bytes. With one \(w\)-bit physical word per block:

\[
N_{\mathrm{blocks}}\approx\frac{D}{ab}.
\]

The favorable operation fraction is

\[
\rho_{\mathrm{op}}=\frac{2}{ab}.
\]

The ideal one-bit payload fraction is

\[
\rho_{\mathrm{ideal\ payload}}=\frac{1}{4ab}.
\]

Physical traffic cannot read a fraction of a word, so it is

\[
\rho_{\mathrm{word\ traffic}}=\frac{w}{4ab}.
\]

Each block table contains \(2^{a+b}\) bits. Its storage relative to the entire
Q4 checkpoint is therefore

\[
\rho_{\mathrm{table\ storage}}
=
\frac{2^{a+b}}{4ab}.
\]

Even granting one unit of construction per table entry, construction costs at
least

\[
E_{\mathrm{build}}
=
\frac{2^{a+b}}{ab}
\]

dense-equivalent passes. Over \(N\) service tokens, the amortized build
fraction is

\[
\rho_{\mathrm{build/token}}
=
\frac{2^{a+b}}{abN}.
\]

These quantities are independent of favorable cancellation or model margins.
They arise directly from the declared layout.

## User-requested 2.5% ceiling

For \(w=64\), physical traffic requires \(ab\ge640\). The minimum-perimeter
integer layout selected by the calculator is \(25\) by \(26\):

| quantity | exact/result |
|---|---:|
| block area \(ab\) | 650 |
| query operations | \(1/325=0.307692\%\) |
| physical word traffic | \(8/325=2.461538\%\) |
| remaining traffic headroom | \(1/2600=0.038462\%\) |
| table perimeter \(a+b\) | 51 |
| table/Q4 storage ratio | \(281474976710656/325\) |
| table/Q4 storage ratio, decimal | \(8.660768514174031\times10^{11}\) |
| minimum table bytes at registered 405B geometry | 174,838,353,849,044,199,198,305 |
| minimum table GiB | \(1.628309058482239\times10^{14}\) |
| minimum address width | 78 bits |
| build dense-equivalents | \(3.4643074056696123\times10^{12}\) |
| build fraction per token over 20M tokens | 173,215.37028348062 |

Thus the constructor passes only the narrow query-read line. It fails
persistent storage, construction amortization, and 64-bit addressability.
Calling this a universal 2.5% executor would omit the cost that created the
small read.

## Registered complete p50 ceiling

The VORTEX runtime must fit all work inside \(8/675\), not merely 2.5%.
For \(w=64\), physical traffic requires \(ab\ge1350\). The favorable balanced
layout is \(37\) by \(37\):

| quantity | exact/result |
|---|---:|
| block area \(ab\) | 1369 |
| query operations | \(2/1369=0.146092\%\) |
| physical word traffic | \(16/1369=1.168736\%\) |
| table perimeter \(a+b\) | 74 |
| table/Q4 storage ratio | \(3.449500717947148\times10^{18}\) |
| minimum table GiB | \(6.485398215045125\times10^{20}\) |
| minimum address width | 100 bits |
| build dense-equivalents | \(1.3798002871788591\times10^{19}\) |
| build fraction per token over 20M tokens | \(6.899001435894296\times10^{11}\) |

The remaining query-traffic headroom is only
\(152/924075\approx0.016449\%\), before metadata, state, KV, scheduling,
verification, and fallback.

A 32-bit word changes the best layout to \(26\) by \(26\), but still needs a
79-bit address, a table \(1.665532406571929\times10^{12}\) times the Q4
checkpoint, and 333,106 dense-equivalent builds per service token.

## Other finite-word routes screened

### Mailman finite-alphabet factorization

Mailman preprocesses a finite-alphabet matrix and saves a logarithmic factor
in later exact MatVec operations. For Q4, the alphabet has 16 symbols. Even
granting the most favorable possible matrix dimension as large as the entire
registered coefficient population, \(D<16^{10}\), so a complete pattern chunk
has at most nine symbols. The favorable operation floor is therefore

\[
\frac{1}{9}=11.111\ldots\%,
\]

already above both ceilings. This is also a static finite-alphabet
factorization, already covered by F-050, not a new causal information source.

### Boolean vector-matrix-vector cell probes

Larsen and Williams prove a bounded-word cell-probe data structure for
Boolean \(u^\mathsf{T}Av\) with
\(O(n^{3/2}/\sqrt w)\) probes and \(O(n^{3/2}\sqrt w)\) redundant bits.
Its output is Boolean-semiring rectangle nonemptiness: it answers whether a
selected submatrix contains any one.

That is not parity, count, signed accumulation, Q4 dot product, or native BF16
result. The cell-probe model charges memory probes while treating all
computation, including query-set manipulation, as free.

At \(n=16384,w=64\), the favorable leading term is 262,144 word probes,
2 MiB, or \(1/64=1.5625\%\) of the corresponding Q4 matrix bytes before hidden
constants. More importantly, the theorem returns only one existence bit. It
cannot be imported as the numerical Bilinear Cross Residual source.

The later succinct Boolean work improves and tightly characterizes related
Boolean probe/redundancy tradeoffs. It likewise does not turn Boolean
nonemptiness into an exact signed numerical sum for free.

### Broadword/Kronecker packing

Packing many Q4 symbols into one machine word can reduce instruction count.
If every packed word is read, coefficient payload traffic remains 100%.
Packing is an execution optimization, not omitted-information recovery.

### Native-rounding finite automaton

BF16/Q4 execution has finitely many word states, but finiteness is not free
storage. An exhaustive transition/answer table is forbidden and exponential.
Generating transitions on demand still reads the coefficient stream unless a
different charged index supplies the missing answer. No general lower bound
for all rounding-aware data structures is claimed.

### Structured-matrix upper bounds

Recent MatVec results obtain subquadratic queries for matrices with bounded
structural parameters such as low VC dimension. They are valuable for a
checkpoint-class theorem. They do not imply an arbitrary-checkpoint theorem:
an arbitrary dense matrix may have the unfavorable parameter value. A
universal VORTEX claim would need either a bound that every supported
checkpoint satisfies or a separate worst-case constructor.

## Exact boundary left open

The following named routes are now closed as primary exact sources:

1. local Block Rank-One Truth Tables;
2. Mailman/static finite-alphabet factorization;
3. full-scan broadword packing;
4. importing Boolean nonemptiness probes as a numerical sum;
5. treating native rounding's finite state as a free answer table.

The following problem remains open:

> Given an arbitrary bounded-word matrix representation with globally nonlocal,
> nonlinear preprocessing and data-dependent probes, can exact rank-one
> numerical queries be answered below the complete VORTEX state, build,
> traffic, operation, and fallback budgets without an exhaustive answer table?

Neither a construction nor a matching lower bound is presently recorded.
Known lower bounds in the current repository do not close this model once
global 8 GiB advice and unrestricted nonlinear probes are admitted.

Therefore this audit justifies no model run, EXP-085 number, backend, kernel,
hardware action, 122B/405B execution, or E2-E7 promotion.

## Reproduction

Run:

    python -m unittest tests.test_finite_word_bilinear_source -v
    python scripts/derive_finite_word_bilinear_source.py
        --output-dir results/e0_finite_word_bilinear_source

Authoritative artifacts:

- results/e0_finite_word_bilinear_source/summary.json
- results/e0_finite_word_bilinear_source/checksums.sha256

Canonical summary SHA-256:

    2f785b0f3b5aac6f6192c59077a64ffdbdf90f2b96282bef04e8bd4ccc5b9007

## Primary sources

- Kasper Green Larsen and Ryan Williams,
  [Faster Online Matrix-Vector Multiplication](https://arxiv.org/abs/1605.01695),
  especially Theorem 2.3 and the Boolean-semiring query definition.
- Diptarka Chakraborty, Lior Kamma, and Kasper Green Larsen,
  [Tight Cell Probe Bounds for Succinct Boolean Matrix-Vector Multiplication](https://arxiv.org/abs/1711.04467).
- Edo Liberty and Steven W. Zucker,
  [The Mailman Algorithm: A Note on Matrix Vector Multiplication](https://www.cs.yale.edu/homes/el327/papers/matrixVectorApp.pdf).
- Emile Anand, Jan van den Brand, and Rose McCarty,
  [The Structural Complexity of Matrix-Vector Multiplication](https://arxiv.org/abs/2502.21240).
