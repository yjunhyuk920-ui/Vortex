# E0 Fourier-Fiber Direct-Sum Frontier

## Verdict

```text
KEEP_FOURIER_FIBER_BOUND_FOR_ALL_LINEAR_TUPLES
KEEP_ARBITRARY_NONLINEAR_GLOBAL_ADVICE_AND_CROSS_BLOCK_PROBES_CHARGED
REJECT_THE_ALL_LINEAR_LIFT_AS_A_GENERAL_RANK_ONE_TARGET_RESOLUTION
KEEP_UNIVERSAL_2.5% NOT ESTABLISHED
KEEP_GENERAL_NONLINEAR_NUMERICAL_RANK_ONE_GAP OPEN
KEEP_NO_SURVIVING_CANDIDATE
```

This audit supplies a genuine direct-sum theorem that survives the XOR advice
counterexample. It treats the whole database at once, fixes one arbitrary
nonlinear advice value, and charges every adaptive probe into every block.
For the complete family of binary linear queries, the theorem forces more
than `26.2%` of the registered one-bit coefficient population in the worst
case even when all 8 GiB of advice and all computation are free.

That is not the VORTEX rank-one theorem. A Transformer query to one matrix has
coefficient shape `r tensor u`, and even 32 independently granted tuples form
a vastly smaller character family than all linear queries. A finite dimension
screen shows that query cardinality alone stops being informative at
`0.0281373%` of the registered coefficient population, 42.12x below the
registered p50 coefficient budget. The active scalar rank-one ticket therefore
remains open.

## Elementary explanation

Imagine a giant book of bits and an 8 GiB notebook prepared from the book.
Books that produce the same notebook are placed in one pile. Because the
notebook is much smaller than the book, at least one pile is enormous.

Now allow every possible parity question:

> Pick any set of pages. Is the number of `1` bits on those pages odd or even?

If an answerer opens at most `t` pages, its yes/no procedure is a decision
tree of depth `t`. Such a tree can only build patterns involving at most `t`
page positions at a time. But the complete collection of parity questions is
powerful enough to distinguish every book left in the pile. Therefore the
pile cannot contain more patterns than a radius-`t` Hamming ball.

XOR advice does not escape this argument. If the notebook reveals one box
only after other boxes are opened, those other openings are included in the
same total `t`.

The catch is equally important. A Transformer does not ask every possible
question about every weight bit. Its coefficient masks have the special form
`r tensor u`. There are enormously fewer such masks. The complete-parity
argument cannot silently replace that smaller question set.

## 1. Declared all-linear systematic model

Let

```text
x in F_2^D                         arbitrary database
R(x) in {0,1}^r                   arbitrary nonlinear global advice
q in F_2^D                         query
answer(q,x) = <q,x> mod 2          exact output
```

The query algorithm may read all of `R(x)` for free, perform unlimited free
computation, choose raw database coordinates adaptively, and probe across any
nominal matrix or layer boundary. It must be deterministic and zero-error.
Assume its worst-case raw-bit depth is `t`.

This is more favorable than the target machine. Words, addresses, cache
lines, query generation, arithmetic, metadata, verification, KV state, and
physical movement are all free.

If the database is partitioned as `x=(x_1,...,x_m)` and a simultaneous
interface returns each independently selected block parity, it also answers
the global query at no additional probe cost because

```text
<q,x> = <q_1,x_1> XOR ... XOR <q_m,x_m>.
```

That is the precise direct-sum interpretation used here.

## 2. Fiber lemma

The advice function has at most `2^r` values. Some advice value `a` therefore
has a fiber

```text
F_a = {x : R(x)=a}
```

with

```text
|F_a| >= 2^(D-r).                                      (1)
```

Fix `a`. For each query `q`, the adaptive algorithm is now an ordinary
coordinate decision tree of depth at most `t` on the inputs in `F_a`.

### 2.1 Decision tree to low-degree polynomial

Each leaf path fixes at most `t` coordinates. Its indicator is a product of
at most `t` factors of the form `x_i` or `1-x_i`. Summing the signed leaf
indicators gives a real multilinear polynomial of degree at most `t` that
agrees with the answer on `F_a`.

Therefore every restricted Walsh character

```text
chi_q(x) = (-1)^<q,x>
```

belongs to the restriction of the degree-`t` multilinear polynomial space.
That space has dimension at most

```text
B(D,t) = sum(j=0..t, C(D,j)).                          (2)
```

Adaptivity does not invalidate this step; it changes the tree, but no path
contains more than `t` queried coordinates.

### 2.2 Characters span every function on the fiber

Take the full Walsh--Hadamard matrix whose rows are databases and columns are
queries. Any two distinct rows are orthogonal over all `2^D` query columns.
Selecting the rows indexed by `F_a` preserves row independence. Hence the
restricted character columns span the complete function space

```text
R^(F_a),
```

whose dimension is `|F_a|`.

Combining this fact with (1) and (2) gives the finite theorem

```text
2^(D-r) <= |F_a| <= sum(j=0..t, C(D,j)).                (3)
```

This proof never divides advice by block count and never conditions on all
other blocks for free. Every raw coordinate used to unlock shared advice is a
probe in the same decision tree.

## 3. Entropy consequence

For `delta <= 1/2`, the standard Hamming-ball inequality gives

```text
sum(j=0..delta*D, C(D,j)) <= 2^(D*H_2(delta)).           (4)
```

Thus any rational `delta` satisfying

```text
H_2(delta) < 1-r/D                                      (5)
```

certifies

```text
t > delta*D.                                            (6)
```

The registered finite witness is

```text
D                                  405,849,243,648 bits
r                                   68,719,476,736 bits
delta                               131/500 = 0.262
H_2(delta)                          0.8297494277867027...
1-r/D                               0.8306773320104999...
strict margin                       0.0009279042237972...
minimum integer worst-case probes   106,332,501,836
minimum fraction                    26.200000000055...%
```

This is an unconditional finite coefficient-bit lower bound for the declared
all-linear tuple model, not an asymptotic hidden-constant statement.

### 3.1 Two target comparisons that must remain separate

Against the registered p50 coefficient-use budget,

```text
(8/675)*D                            4,810,065,109.902... probes
forced / registered budget          22.10625x.
```

Against the user's favorable DFloat `1/40` block-bit line,

```text
requested block                     110,244,000,000 bits
forced / requested                  0.9645196x
```

the one-bit-per-probe lower bound is still `3.55%` short. It may not be
rounded up into a 2.5% DFloat impossibility result. Conversely, a fully
independent all-linear query description itself contains `D` bits, or
`3.68137x` that block line, if query traffic is charged. Standard Transformer
queries have compact factor descriptions and cannot inherit that input-cost
rejection.

## 4. Why rank-one characters break the spanning step

For a binary `m x n` matrix, a rank-one query is described by one `m`-bit
left vector and one `n`-bit right vector. Therefore the number of query
characters is at most

```text
2^(m+n),
```

not `2^(mn)`. Restricted to an advice fiber, those columns need not span all
functions on the fiber. The equality

```text
character span dimension = |F_a|
```

that powered (3) is no longer available.

Grant every registered matrix an independently selectable rank-one pair and
even count duplicate pairs as distinct in the method's favor. The frozen
shape population reconstructs exactly `405,849,243,648` parameters and gives

```text
one model-wide tuple:   log2 query-count upper bound      39,254,528
32 independent tuples: log2 query-count upper bound   1,256,144,896.
```

Query cardinality caps the restricted-character column dimension. The
elementary inequality

```text
C(D,t) >= (D/t)^t
```

then gives finite points at which the low-degree space is already large
enough to contain every possible query column:

| Granted query family | `t` | Integer witness | Fraction of `D` |
|---|---:|---:|---:|
| one independent rank-one tuple | 2,309,090 | `17*t >= 39,254,528`, `t <= D/2^17` | 0.000568953% |
| 32 independent rank-one tuples | 114,194,991 | `11*t >= 1,256,144,896`, `t <= D/2^11` | 0.028137293% |

These are ceilings on what **query cardinality plus the low-degree dimension
argument** can force. They are not query algorithms and do not prove that
rank-one answers can be obtained with those probe counts.

Even the 32-tuple method ceiling is

```text
42.1215x below the registered p50 coefficient budget
965.401x below the user's DFloat block-bit line.
```

Therefore this direct-sum proof is strong enough to handle global advice
synergy but far too broad in its query premise and far too weak after the
rank-one restriction.

## 5. Executable controls

The reference control exhausts all `255` nonempty fibers of the three-bit
cube and all eight parity queries on each fiber:

```text
fiber/query checks             2,040
fiber-volume violations        0
maximum observed tree depth    3
```

A separate four-bit Walsh control checks all `120` distinct row pairs:

```text
orthogonality failures         0.
```

The production calculation also reconstructs the registered parameter count
from every matrix shape before computing the rank-one query cardinality.

## 6. Exact claim boundary

```text
PROVED      all-linear Fourier-fiber inequality (3)
PROVED      arbitrary nonlinear advice is allowed
PROVED      adaptive cross-block probes are jointly charged
PROVED      registered 26.2% all-linear coefficient-bit witness
MEASURED    2,040 exhaustive fiber/query controls; zero violations
MEASURED    120 Walsh orthogonality controls; zero failures

NOT PROVED  a lower bound for the restricted rank-one character matrix
NOT PROVED  causal Transformer query tuples are arbitrary all-linear masks
NOT PROVED  the user's DFloat 2.5% impossibility from the binary witness
NOT PROVED  universal 2.5% feasibility
NOT PROVED  universal 2.5% impossibility
```

No model forward, checkpoint mutation, experiment number, backend, kernel,
download, Ubuntu command, or hardware action occurred.

## 7. Remaining mathematical target

The missing theorem can now be stated more sharply. For every large advice
fiber `F` of arbitrary matrices, let `Q_rank1` be the rank-one character set.
One needs a target-scale lower bound on

```text
rank(H[F, Q_rank1])
```

or another measure that still forces deep adaptive coordinate trees after
global nonlinear advice. Query cardinality alone cannot provide it. A proof
must exploit the internal geometry of rank-one characters; a constructor must
exploit that same geometry with a complete bounded-word and physical-cost
equation.

Until one of those exists, the active General Finite-Word Rank-One Probe Gap
remains claimed.

## Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_fourier_fiber_direct_sum_frontier.py `
  --output-dir results\e0_fourier_fiber_direct_sum_frontier

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_fourier_fiber_direct_sum_frontier -v
```

Authoritative artifacts:

- `results/e0_fourier_fiber_direct_sum_frontier/summary.json`
- `results/e0_fourier_fiber_direct_sum_frontier/checksums.sha256`
- `vortex_runtime/fourier_fiber_direct_sum_frontier.py`
- `tests/test_fourier_fiber_direct_sum_frontier.py`

Canonical summary SHA-256:

```text
004f37f8b872bfa25ecfd0704941b8656a173829322d79f73a144447ad6a6ca3
```
