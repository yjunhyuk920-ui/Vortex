# E0 General Finite-Word Rank-One Probe Barrier

## Verdict

The requested statement

\[
\forall W,s,u:\qquad \rho_{\mathrm{complete}}(W,s,u;A=32)\le 2.5\%
\]

is **not established**. No constructor in the repository proves it, and the
known general nonlinear/adaptive cell-probe theorems do not prove it
impossible at the registered scale. The correct classification remains:

```text
PRIMARY RESEARCH TRACK / REVISE
GENERAL NONLINEAR RANK-ONE GAP / OPEN
UNIVERSAL 2.5% / NOT ESTABLISHED
CORE CANDIDATE / NONE
```

This is not a semantic distinction. `2.5%` is not a knob that a scheduler can
set while retaining universal exactness. It is an upper bound on a minimum
certificate cost. A scheduler may stop at `2.5%` only when the evidence read
so far already determines the exact reference decision and exact accepted
state. Otherwise a sound executor must continue.


## Exact quantifier that a universal theory must prove

For one fixed exact executor `E`, let `C_E(W,q)` be every charged byte needed
to answer query/state `q` on checkpoint `W`, including exact weight evidence,
bound and index metadata, scheduler reads, KV/state exactization, spill, and
nonlinear replay. The requested theorem is not merely an average:

\[
\exists E\;\;\forall W,q:\qquad
E(W,q)=\operatorname{Reference}(W,q),
\qquad
\frac{C_E(W,q)}{S_c(W)}\le\frac1{40}.
\]

It must additionally satisfy the 8 GiB resident-state and 20 ms/token
constraints. A scheduler cap proves only `C_E(W,q) <= 1/40` on executions that
finish before the cap. If an interval remains open at the cap, stopping is
incorrect and continuing violates the claimed bound. Therefore the universal
quantifier can be discharged only by a complete constructor/proof, not by
assigning `rho = 2.5%`.

## Elementary explanation

Imagine a locked book whose answer depends on many pages. The executor may
keep an 8 GiB notebook prepared from the book and may choose pages
adaptively. The claim “always 2.5%” says that for **every possible book and
every possible question**, the notebook plus at most 2.5% of the pages always
determines the exact answer.

Choosing to close the book after 2.5% does not prove the answer. One needs
either:

1. a recipe showing how the notebook and those pages determine every answer;
   or
2. a theorem showing that no such recipe can exist.

The first recipe has not been found. The strongest theorems currently
available are far too weak to supply the second conclusion for VORTEX's
rank-one questions and global notebook.

## Frozen favorable lower-bound model

The audit gives a hypothetical data structure more power than the target
runtime:

- the unchanged checkpoint remains in systematic read-only memory;
- arbitrary checkpoint-only preprocessing may create one globally shared
  8 GiB advice string;
- reading every advice bit is free;
- checkpoint probes may be nonlinear, adaptive, data-dependent, and globally
  nonlocal;
- all computation between probes is free;
- the answer must be zero-error and exact.

Unbounded words, future traces, approximation, external provers, and an
exhaustive table of answers to all activation pairs remain forbidden. A lower
bound in this favorable model would also apply to a real bounded machine. An
upper bound in the model would still need native Q4/BF16, compute, address,
state, and traffic accounting before becoming a VORTEX constructor.

## Registered numbers

The DFloat11 reported storage figure is `551.22 GB`, or
`4,409,760,000,000` bits. Eight GiB is `68,719,476,736` advice bits, which is
`1.5583495867%` of that physical representation.

For one 32-token block:

| quantity | exact value |
|---|---:|
| requested 2.5% evidence | 110,244,000,000 bits = 13,780,500,000 bytes |
| requested evidence per certified token | 430,640,625 bytes |
| registered `8/675` block evidence | 52,263,822,222.222... bits |
| registered `8/675` evidence per token | 204,155,555.555... bytes |

Thus the user-facing phrase “2.5% for 32 tokens” is actually a
`1/1280 = 0.078125%` full-storage share per emitted token before compute,
metadata, KV, state exactization, or scheduler traffic.

This audit gives all 2.5% bytes to checkpoint probes and charges every other
term as zero. The real `rho_complete` numerator is larger because metadata,
KV/state exactization, scheduling, and replay must share the same budget.
Failure under the favorable accounting would be decisive; success would still
need the omitted costs.

## Nisan--Rudich--Saks help bits: valid but too small

Nisan, Rudich, and Saks allow `l` arbitrary help bits about `l+1` disjoint
instances of parity. Even though a decision tree for one instance may probe
variables belonging to the other instances, at least one parity retains its
full decision-tree depth. For parity this depth is the number of input bits.

This theorem really does cover arbitrary nonlinear shared advice. The finite
target substitution is nevertheless tiny because the global advice contains
`68,719,476,736` bits. Partitioning the entire database into `l+1` instances
leaves only:

| database interpretation | parity bits in each forced instance |
|---|---:|
| one bit per 405B parameter | 5 |
| four Q4 bits per parameter | 23 |
| all 551.22 GB reported physical bits | 64 |

The most favorable row forces 64 bit probes, or only one 64-bit word probe.
The requested 2.5% block budget contains 110,244,000,000 bits, a factor of
about 1.722 billion larger. Therefore this theorem closes the *kind* of
advice but not the target scale.

The 64-bit row is deliberately over-favorable: a lossless compressed file's
physical code bits are not automatically independent parity inputs. The legal
native numerical query also does not automatically expose the four Q4 bit
planes as independent parity questions, so the 23-bit Q4 row is another
favorable bit-input ceiling. The strict GF(2) one-bit-per-parameter surrogate
gives five. Since even the invalidly generous 64-bit ceiling is billions of
times too small, these caveats only weaken the available rejection further.

Primary source: Nisan, Rudich, and Saks, *Products and Help Bits in Decision
Trees*, FOCS 1994 / SIAM J. Comput. 1998,
<https://www.brics.dk/NS/94/3/BRICS-NS-94-3.pdf>.

## Ko 2025: arbitrary nonlinear preprocessing still does not reach rank-one

Ko's Theorem 1.3 is the most relevant recent progress against arbitrary
nonlinear preprocessing. Its explicit premises include

\[
\log |S|\ge 10^4 t\log s,
\qquad
m=\omega(s2^{5t}),
\]

where `S` is the collection from which the hard linear queries are selected,
`s` is the number of one-bit data-structure cells, and `t` is the probe count.

Three finite substitutions matter:

1. For a `16384 x 16384` GF(2) matrix, the rank-one query family has
   `log2 |S| < 32768`. With the full global advice included in `s`, the first
   premise does not certify even `t=1` at the displayed constant.
2. Even an **illegal favorable reshape** of all 551.22 GB physical bits into
   one square bit matrix permits only about ten probes under the first
   premise. Actual Transformer queries do not become rank-one queries of
   this artificial reshaping.
3. Replacing rank-one queries by *all* linear queries makes the collection
   exponentially larger. Even then the displayed entropy premise reaches
   only `10,492,896` bit probes, while the requested block budget is
   `110,244,000,000` bits: a factor greater than `10,506`. The separate
   `omega` premise supplies no finite target threshold, and the all-linear
   query family is not VORTEX's rank-one family.


There is an additional accounting mismatch. Ko's `t` charges probes into
the whole preprocessed `s`-cell data structure. The favorable VORTEX
lower-bound model instead lets the algorithm read all 8 GiB of nonlinear
advice for free and charges only probes into the original checkpoint. A lower
bound on total data-structure probes does not become the required lower bound
on cold checkpoint probes after making those advice probes free. The numeric
comparisons above are therefore optimistic screening ceilings, not a direct
application to `rho_complete`.
Consequently Ko's result is strong evidence that nonlinear preprocessing is
not magic, but it is not a VORTEX-scale constructor rejection.

Primary source: Young Kun Ko, *Lower Bounds for Linear Operators*, ECCC
TR25-155 (2025), <https://eccc.weizmann.ac.il/report/2025/155/download>.

## Chakraborty--Kamma--Larsen: global advice cannot be divided for free

For exact GF(2) `u^T M v`, Chakraborty, Kamma, and Larsen prove

\[
t r=\Omega(n^3/\log n)
\]

for a single `n x n` matrix when `n <= r <= n^2/4`, and a near-full-scan
bound for `r<n`. Their systematic model already permits all redundancy bits
to be read for free.

The largest registered square has `n=16384`, hence `n^2/4=67,108,864` bits.
The global 8 GiB grant is exactly 1024 times that upper endpoint. Replacing
the global advice with `8 GiB / number_of_matrices` is invalid: one nonlinear
advice bit may mix many matrices, and the theorem does not supply the required
model-wide direct sum. The theorem therefore cannot be applied to the target
by bookkeeping fiat.

Primary source: Chakraborty, Kamma, and Larsen, *Tight Cell Probe Bounds for
Succinct Boolean Matrix-Vector Multiplication*,
<https://cs.au.dk/~larsen/papers/BooleanMatrixVectorLB.pdf>.

## The deeper barrier

The remaining abstraction is the systematic nonlinear, or “common bits,”
model for linear queries. The linear restriction corresponds to matrix
rigidity and has useful bounds. Allowing arbitrary nonlinear common bits and
adaptive probes is substantially harder. The literature explicitly treats
the proposition that nonlinear preprocessing is no more powerful than linear
preprocessing as a long-standing linearization problem; Ko 2025 gives the
first substantial progress for large random query operators, not a linear
target-fraction theorem for the structured rank-one family.

Ramamoorthy and Rashtchian also identify rank-one GF(2) queries with the
vector-matrix-vector problem and connect the systematic **linear** model to
rigidity. Their general cell-probe result is in a much weaker high-error
regime and does not yield the registered finite traffic rejection.

Primary source: Ramamoorthy and Rashtchian, *Equivalence of Systematic Linear
Data Structures and Matrix Rigidity*, <https://arxiv.org/abs/1910.11921>.

## What can be proved now

The sound theorem is conditional:

\[
\boxed{
\text{If the exact token and exact accepted state are certified by 2.5%,}
\text{ then stopping at 2.5% is sound.}
}
\]

The universal termination theorem is different:

\[
\boxed{
\text{If refinement eventually reads and replays every dependency exactly,}
\text{ then the executor terminates with the reference result.}
}
\]

Combining these statements does **not** give a universal 2.5% performance
theorem. It gives a fail-closed algorithm with an input-dependent fast path
and a potentially full terminal path.

To make “always 2.5%” true, at least one contract must change: restrict the
checkpoint/workload distribution and measure a tail quantile; permit a much
larger query-dependent witness or modified checkpoint; permit approximation;
or provide hardware that performs the full work outside the charged resource.
Each option is a different goal and is not silently adopted here.

## Reproduction

```powershell
$env:PYTHONPATH = ".;.deps"
.deps\exp076-venv\Scripts\python.exe scripts\derive_general_rank_one_probe_gap.py --output-dir results\e0_general_rank_one_probe_gap
.deps\exp076-venv\Scripts\python.exe -m pytest -q tests\test_general_rank_one_probe_gap.py
```

No model forward, checkpoint download, backend, kernel, or hardware action is
part of this audit.
