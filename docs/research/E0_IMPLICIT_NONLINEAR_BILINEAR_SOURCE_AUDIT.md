# E0 Implicit Nonlinear Bilinear Source Audit

Date: 2026-08-09  
Evidence: E0 Phase A theorem audit plus Phase B exact-reference controls  
Model forwards, checkpoint mutation, and hardware actions: zero

## Question

Can exact nonlinear arithmetic and query-dependent branching itself supply a
new implicit checkpoint source for the Bilinear Cross Residual
`f_W(r,u)=r^T W u`, after trace-built linear code unions failed?

## Decision

```text
REJECT_EXACT_FIELD_NONLINEAR_BILINEAR_ARITHMETIC_AS_DISTINCT_CORE_CLASS
REJECT_TWIN_WIDTH_AND_GRAMMAR_REOPENING_AS_NEW_SOURCE
KEEP_FINITE_WORD_DISCONTINUOUS_QUERY_SOURCES_OPEN
KEEP_NO_SURVIVING_CANDIDATE
```

The result is a class-containment theorem, not a universal impossibility
proof. On an exact field and an open query domain, bounded algebraic
branching cannot be the missing adaptive information source: one fixed
full-dimensional branch computes the complete bilinear function, and
Baur--Strassen differentiation turns that scalar circuit into a static
arithmetic circuit for `W u` with constant-factor arithmetic overhead. Such a
circuit returns to the static arithmetic-DAG family already archived in
EXP-072B; it does not create a distinct Core Candidate.

The theorem does not cover native BF16/Q4 rounding, bitwise instructions,
floor/modular word operations, discontinuous data-dependent addressing over a
finite activation alphabet, or unrestricted cell-probe structures. Those are
the only honest remaining interpretation of an "implicit nonlinear" source.

Structurally valid conditions were established. Large-model performance
remains unverified.

## Exact-field path-collapse theorem

Fix the unchanged checkpoint matrix `W`. Let a preprocessor produce arbitrary
finite advice `D(W)`. Suppose a bounded query program uses field operations
`+,-,*,/`, comparisons of rational expressions, and finitely many advice-cell
probes, and returns

```text
f_W(r,u) = r^T W u
```

for every `(r,u)` in a nonempty open domain where its denominators are defined.
Assume its registered worst-case bound gives only finitely many branch/probe
paths.

At least one path is followed on a set with nonempty interior. Refine that set
until all comparison outcomes and integer probe addresses are fixed. The
resulting path is one static rational straight-line program `C_W(r,u)`. Since
`C_W` and `f_W` agree on a nonempty open set, their rational-function
difference is identically zero. Therefore the fixed path computes `f_W`
wherever both are defined; query adaptivity did not create a separate answer
source.

Now use the identity

```text
gradient_r f_W(r,u) = W u.
```

Baur and Strassen prove that a rational function and all its first partial
derivatives can be computed with at most a constant-factor increase in
arithmetic-circuit complexity. Under their all-unit-cost measure the explicit
factor is at most four. Applied here, a size-`s` exact bilinear path gives a
static arithmetic circuit for `W u` of size at most `4s`.

Primary source: Walter Baur and Volker Strassen, *The Complexity of Partial
Derivatives*, Theoretical Computer Science 22 (1983), pp. 317--330,
<https://web.vu.lt/mif/s.jukna/tropical/Baur-Strassen.pdf>.

This reduction is deliberately used only for **novelty containment**. It does
not prove a finite traffic lower bound, preserve a particular memory layout,
or show that the derived static circuit meets VORTEX's final budget.

## Registered finite implication

The complete p50 allowance is

```text
alpha = 8/675 = 1.185185185...% of dense.
```

If an exact-field scalar query path did meet `s/P <= alpha`, the
Baur--Strassen construction would certify a static arithmetic MatVec circuit
with the favorable operation ceiling

```text
4 alpha = 32/675 = 4.740740740...% of dense.
```

That is not a proof that the static circuit meets `alpha`; it is proof that
the allegedly new nonlinear algebraic mechanism is only a constant-factor
view of the already archived static arithmetic-DAG search. Reopening static
circuits would require an actual constructor, representation, and target-cheap
schedule that invalidates the prior premise. None was found here.

## Why the current twin-width result is not a new route

Kozma and Opler's 2026 result is exact and initially looks relevant: bounded
twin-width binary matrices can be preprocessed for near-linear MatVec. Its two
algorithmic paths, however, fall exactly into already closed VORTEX classes:

1. For general bounded twin-width, the paper derives low row/column Hamming
   coherence and invokes predecessor-row difference traversal. That is the
   same Hamming-differential source tested and rejected by EXP-082A/F-049.
2. For a supplied twin-order or mixed-free order, disjoint one-rectangle
   decompositions give an `O(n+|R|)` exact MatVec schedule. Each rectangle
   forms a fixed column sum and adds it to a fixed row interval. This is a
   static exact linear circuit, hence F-050/EXP-072B. Splitting a Q4 matrix
   into bitplanes or level indicators changes constants, not the class.

The general paper explicitly describes the Hamming-coherence predecessor
algorithm. Primary source: László Kozma and Michal Opler, *Fast and simple
multiplication of bounded twin-width matrices*, arXiv:2602.20023,
<https://arxiv.org/abs/2602.20023>.

No real Transformer twin-width measurement was run because the mechanism
failed the novelty Gate before E1. The conclusion is not that real weights
have high twin-width; it is that favorable twin-width would reopen a closed
static/differential class rather than provide the requested implicit nonlinear
source.

## Grammar compression audit

Lossless grammar-compressed MatVec evaluates a fixed grammar in time
proportional to its compressed representation. This is useful compression
engineering, but its query is a static program and therefore also belongs to
F-050. The published ML-matrix results report dataset-dependent compression
and speedups, not an automatic sub-`1.185185185%` route for arbitrary dense
Transformer checkpoints.

Primary source: Paolo Ferragina et al., *Improving Matrix-vector
Multiplication via Lossless Grammar-Compressed Matrices*, arXiv:2203.14540,
<https://arxiv.org/abs/2203.14540>.

## Why a stronger finite-word impossibility is not claimed

The closest cell-probe results do not currently close VORTEX's remaining
interface. Chakraborty, Kamma, and Larsen prove succinct Boolean
vector-matrix-vector tradeoffs over a bounded redundancy range, but EXP-071
already established that the registered per-block side-information grant is
outside the useful finite range and that no model-wide direct-sum theorem was
provided. Their Boolean operation also does not reproduce signed Q4/BF16
accumulation. Primary source: *Tight Cell Probe Bounds for Succinct Boolean
Matrix-Vector Multiplication*, <https://arxiv.org/abs/1711.04467>.

Ko's 2025 lower bound permits arbitrary nonlinear preprocessing of the input
vector and is important evidence that nonlinear advice is not automatically
free. Its hard query operators are random/large families; this audit did not
derive the finite constants or a direct-sum reduction for VORTEX's structured
rank-one bilinear query family. Primary source: Young Kun Ko, *Lower Bounds
for Linear Operators*, ECCC TR25-155,
<https://eccc.weizmann.ac.il/report/2025/155/>.

Accordingly, the finite-word ticket remains a construction-or-bound question.
Neither paper is promoted into a target-scale impossibility theorem here.

## Exact reference controls

The repository reference builds two different arithmetic paths for random
small rational matrices. Both add nonlinear degree-four or degree-six terms
that cancel exactly. Reverse-mode differentiation is then compared with an
independent exact `W u` implementation.

```text
seed                         11607714
cases                        64
path 0 / path 1              32 / 32
exact scalar matches         64 / 64
exact left-gradient matches  64 / 64
model forward calls          0
```

These controls demonstrate the derivative identity and guard against an
implementation that works only for syntactically linear expressions. They do
not prove the open-set rational identity theorem; that is the Phase A
argument above.

## Remaining admissible source class

The next construction must depend essentially on finite-word discontinuity,
not merely call an algebraic circuit nonlinear. It must state an actual
checkpoint constructor and an exact query equation using some explicit
combination of bitwise packing, modular/floor operations, bounded
subexponential tables, or discontinuous addressing. It must charge:

- full persistent representation and preprocessing;
- address generation and every cell/word probe;
- logical and physical traffic, arithmetic and word operations;
- hot state and intermediate storage;
- native BF16/Q4 reconstruction or a declared exact surrogate;
- certification, misses, fallback, and service-life amortization.

An exhaustive activation-answer table remains forbidden. A word-level method
that expands to a static arithmetic schedule is not new. No model row,
EXP-085 number, backend, kernel, download, private Ubuntu action, or hardware
test is authorized until one finite-word constructor survives E0.

## Reproduction

```powershell
$env:PYTHONPATH = ".;.deps"
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_nonlinear_bilinear_source.py `
  --output-dir results\e0_nonlinear_bilinear_source
.deps\exp076-venv\Scripts\python.exe -m pytest -q `
  tests\test_nonlinear_bilinear_source.py
.deps\exp076-venv\Scripts\python.exe -m pytest -q
.deps\exp076-venv\Scripts\python.exe scripts\run_validation.py
```

Authority:

```text
vortex_runtime/nonlinear_bilinear_source.py
scripts/derive_nonlinear_bilinear_source.py
tests/test_nonlinear_bilinear_source.py
results/e0_nonlinear_bilinear_source/summary.json
results/e0_nonlinear_bilinear_source/checksums.sha256
```

Observed in this session: five focused standard-library tests passed; all 64
exact value controls and all 64 exact gradient controls passed; a separate
temporary output directory reproduced the canonical summary byte-for-byte.
Canonical summary SHA-256:
`7bf00dc2691d11956105abfa1d6bb97444cfef6fea6b5e9ca6b8318525d7e057`.

The existing repository `pytest` package and standard validation dependencies
were inaccessible under the current sandbox ACL. The focused result is
therefore validated, but a full repository pytest regression and standard
validation pass are explicitly not claimed for this session. The failed
dependency launches occurred before any model or scientific validation row
and are not evidence.
