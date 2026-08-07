# E0 Causal Information-Source Search

Date: 2026-08-08
Evidence: E0 registered-shape derivation plus E1 synthetic/reference controls
Model, target hardware, and private Ubuntu server: not run

## Question

What information available from the current committed prefix and unchanged
checkpoint can choose a query-dependent subset of exact cold pages without a
dense discovery pass, target-future tokens, training, checkpoint modification,
an external prover, or a free trace?

## Decision

```text
REJECT_TABLE_ENUMERATION_AS_A_TARGET_FEASIBLE_SOURCE
REJECT_BOOLEAN_ABSORPTION_AS_A_NUMERICAL_TRANSFORMER_SOURCE
IDENTIFY_CAUSAL_RESIDUAL_ATLAS_AS_A_CONCRETE_CODED_CAUSAL_COLD_SOURCE
AUTHORIZE_ONLY_A_CHEAP_REAL_WEIGHT_RESIDUAL_CERTIFICATE_GATE
NO_SURVIVING_CANDIDATE_OR_POSITIVE_MILESTONE_YET
```

A concrete source now exists on paper: exact input/image pairs already produced
for committed prefix tokens. They can be converted causally into an input basis
`Q` and its exact-real image `Z = WQ`. A later activation is evaluated at the
cached center, while every out-of-span component remains an explicit residual.
Only residual column pages are revealed; unread pages remain inside a sound
operator-norm enclosure and any unresolved decision executes dense fallback.

This is named a **Causal Residual Atlas**. It is not the old tolerance-based
Atlas fast path and not a claim that later activations lie exactly in the prefix
span. The residual is never silently discarded.

The source has a nonempty favorable 405B logical budget region, but the decisive
premise is unmeasured: real causal residuals must become small enough to certify
at least `99.899840530%` of registered rank-16/token branches after favorable
64-token build amortization. Until a preregistered real-weight Gate establishes
that, the authoritative project state remains `NO_SURVIVING_CANDIDATE`.

## Source conservation audit

An unread coefficient can influence a numerical dot product unless information
about it arrives through at least one of the following channels.

| Channel | Concrete source | Result under the registered scope |
|---|---|---|
| Static checkpoint structure | compression, low rank, sparse/differential/static circuits | closed or adverse in EXP-053 through EXP-072B and EXP-082A |
| Exhaustive query advice | table/BDD entry for the observed activation | exact but table growth is prohibitive; closed as enumerative advice |
| Collapsing output algebra | Boolean OR/AND or another absorbing decision | genuine in Boolean data structures; ordinary signed numerical sums do not inherit it |
| Restricted current query | sparse/low-dimensional reachable activation | exact zero and exact temporal span failed; raw approximation has no contract |
| Committed-query amortization | earlier exact `(x, W x)` pairs | supplies the Causal Residual Atlas center without future information |
| Certificate | bound proving unread residual cannot change declared output | required together with the Atlas; source-free final-head and range bounds already failed |
| Randomness | unbiased or probabilistic estimate | not an exact omitted-contribution source without a declared mathematical contract |
| External computation | prover, remote worker, added accelerator | outside the fixed no-added-hardware mission |

The new candidate combines two channels that were not previously joined:
committed-query amortization supplies an exact center, and a certificate keeps
the remaining information explicit. Neither channel alone is sufficient.

## Why known preprocessed MatVec results do not supply the target source

Williams gives an exact finite-semiring scheme with query work
`O(n^2 / (epsilon log n)^2)` after `O(n^(2 + epsilon log_2 K))`
preprocessing. At `n = 16,384`, an `84.375x` operation reduction needs
`epsilon >= 0.656113324`. Even granting the tiny `K = 16` alphabet to the
whole arithmetic, the preprocessing exponent becomes at least
`4.624453296`, with about `1.1496e11` table states in the corresponding
per-group factor before stored vectors and links. Actual activation arithmetic
has a much larger state alphabet, and native rounded floating point is not the
associative finite semiring assumed by the theorem. This is an exact existence
result, not a target-feasible code. Source:
[Williams, *Matrix-Vector Multiplication in Sub-Quadratic Time*](https://people.csail.mit.edu/rrw/mat-vec3.pdf).

Larsen and Williams give a deterministic Boolean-semiring cell-probe structure
using `O(n^(7/4)/sqrt(w))` probes. Comparing coefficient operations to probes
would misleadingly suggest about `90.51x` at `n = 16,384`, `w = 64`. A packed
Boolean matrix already needs only `n^2/w` word probes, so the additional traffic
factor at that finite point is only

```text
(n^2 / w) / (n^(7/4) / sqrt(w))
= n^(1/4) / sqrt(w)
= 1.414213562x.
```

More importantly, the query procedure proves that an OR is one after finding
one witness or ruling out a zero rectangle. It does not recover signed counts,
integer dot products, BF16 sums, or all Transformer hidden values. Source:
[Larsen and Williams, *Faster Online Matrix-Vector Multiplication*](https://arxiv.org/abs/1605.01695).

Chakraborty, Kamma, and Larsen improve the asymptotic Boolean succinct
data-structure frontier and also separate Boolean-semiring and `F2` statements.
Their result reinforces rather than removes the algebra boundary: saturation
can collapse many matrix states to one Boolean answer, whereas numerical linear
maps must retain sum information. Source:
[Chakraborty, Kamma, and Larsen, *Tight Cell Probe Bounds for Succinct Boolean Matrix-Vector Multiplication*](https://arxiv.org/abs/1711.04467).

A small output certificate is also not automatically a cheap selector. Blanc,
Koch, Lange, and Tan obtain efficient certificate discovery under monotonicity
and exponential-in-certificate-size lower bounds for general non-monotone
functions. A signed, deeply composed Transformer therefore cannot treat
certificate existence as free discovery. Source:
[Blanc et al., *The Query Complexity of Certification*](https://arxiv.org/abs/2201.07736).

## Causal Residual Atlas algorithm

For one unchanged matrix `W in R^(m x n)`, let committed prefix inputs and
their already executed exact-real images be columns of `X` and `Y = W X`.
Automatic causal orthogonalization produces

```text
Q = X A
Z = Y A = W Q.
```

No target-future activation and no additional checkpoint scan is needed for
`Z`; it is a linear combination of results that the unchanged prefix execution
already produced.

For the current committed activation `x`:

```text
a = Q^T x
u = x - Q a
W x = Z a + W u.
```

Partition the input coordinates into cold column pages `G`. For a revealed
page set `S`, compute

```text
y_hat = Z a + sum_(G in S) W[:,G] u[G].
```

Compile a safe scalar `beta_G >= ||W[:,G]||_2` for every page. The unread
contribution then satisfies

```text
||W x - y_hat||_2
  <= sum_(G not in S) beta_G ||u[G]||_2.
```

The reference uses the Frobenius norm as a safe `beta_G`, validates supplied
bounds against the exact spectral norm on synthetic cases, rejects corrupt or
non-finite state, and returns exact dense output when every page is revealed.
A unique top-1 center with margin greater than twice a final L2 radius is
certified; otherwise the path is unresolved. End-to-end Transformer radius
propagation is intentionally absent and is the next decisive Gate, not an
assumed free operation.

## Why this is materially different from closed families

- `OnlineAtlasLinear` and EXP-069 commit only on exact span membership. EXP-069
  proves that real warm trajectories keep adding exact directions; it does not
  lower-bound the size of a fully retained and progressively revealed residual.
- EXP-079A used four fixed DCT directions. The new code is per-request and made
  only from committed causal prefix pairs whose exact images already exist.
  This changes the information source, not merely the rank.
- EXP-068 and CPTC bounded raw unread contributions. Here the bound applies to
  `u = x - QQ^T x`; the causal control variate can make the certified object
  materially smaller. Whether it actually does so is unverified.
- EXP-081A used a static nonlinear lookup and rank-eight finite-field residual
  code. The Atlas admits no lookup hit and claims no sparse exact residual; all
  residual pages remain available to a sound enclosure or dense fallback.
- A static circuit cannot create `Q` from the current committed request. The
  code therefore is query-specific without using target future tokens.

## Registered 405B favorable screen

The calculator uses the frozen EXP-080A/071 non-embedding shapes:

```text
dense coefficients                         403,747,897,344
matrix instances                           883
sum of matrix output rows                   19,997,952
shared causal-site input dimensions         12,918,784
p50 dense-equivalent allowance                  8/675
```

The following screening point is not a frozen experiment parameter. It proves
only that the mechanism is not rejected by storage arithmetic before measuring
its information premise:

```text
prefix rank                                      16
capsule scalar bytes                              2
service tokens for minimum build amortization    64
cold page width                                   64 columns
requested cold coefficient fraction              0.200000000%
```

Page rounding is charged independently for every matrix:

```text
capsule elements                         526,667,776
capsule bytes                           1,053,335,552 = 0.980995178 GiB
capsule read bytes/token                1,466,736,640 = 1.366004944 GiB
metadata blocks / bytes                     298,624 / 1,194,496
selected pages/token                          1,009
selected coefficients/token          1,411,989,504
actual cold coefficient fraction         0.349720584%
fast traffic fraction                    1.076872921%
64-token amortized traffic fraction      1.085025716%
fast operation fraction                  0.534634421%
64-token amortized operation fraction    0.557958575%
minimum build operations              6,026,930,176
unallocated part of 8 GiB                7.017892361 GiB
```

The unallocated number is not a peak-memory pass: KV, runtime workspaces,
double buffers, numerical-enclosure state, fallback overlap, and allocator
headroom are still absent.

If fast work is paid before every eventual fallback, the registered equation
requires:

```text
minimum certificate coverage from traffic       99.899840530%
minimum certificate coverage from operations    99.372773390%
```

Thus the traffic branch controls. The successful path would have a logical
traffic amplification of `92.861468x` before build amortization, above the
zero-overhead `84.375x` conservation floor. This is a real nonempty arithmetic
window, not evidence that the required certificate coverage exists.

The seemingly more accurate rank-16, requested-0.5% page point is already
infeasible: page rounding raises actual cold work to `0.732164701%`, and its
amortized traffic becomes `1.262688041%`, above `8/675`. Rank 32 with the
0.2% request also fails traffic before a certificate. The next Gate therefore
must not hide page granularity or rescue a failed residual premise with a rank
sweep.

## Strongest counterexample

Let the current `x` be orthogonal to every committed-prefix direction. Choose
each unread block so `W[:,G]u[G]` points along the same output vector. Then the
triangle/operator bound is tight and the residual can cross any smaller token
margin until essentially every decisive page is read. A one-token or
low-rank prompt is an immediate practical version of the same failure mode.

Consequences:

1. the mechanism has exact/fail-closed correctness but no universal sublinear
   performance guarantee;
2. a favorable Llama result cannot be generalized to every dense checkpoint;
3. the flagship route depends on a measured causal-prefix residual population,
   not on dimension counting alone;
4. short prompts, orthogonal arrivals, bound growth through nonlinear layers,
   native rounding, and 1,009 sequential page requests can independently reject
   the path.

## Cheapest next Gate

Do not build a cold runtime or download a larger model. On the already pinned
small unchanged checkpoint, the next ticket must preregister one rank, page
width, prefix/decode population, and exact stop rule, then measure an impossible
favorable oracle before a deployable bound:

1. use committed prompt activations only to build each causal basis;
2. reproduce the unchanged target trace exactly;
3. at each later projection, compute the best permitted prefix-subspace center;
4. reveal residual pages in the most favorable legal order;
5. propagate exact-reference residuals to determine the minimum page fraction
   that could preserve the frozen output contract;
6. reject immediately unless population coverage can still reach the
   `99.899840530%` traffic frontier with no family or short-prefix escape;
7. only a pass may authorize a deployable outward-rounded bound propagator.

The oracle may inspect the exact reference only to create a favorable lower
bound; it may not become the selector. Physical page I/O, a kernel, larger
checkpoint, and the Ubuntu server remain prohibited until the logical Gate
survives.

## Reference implementation and reproduction

```powershell
$env:PYTHONPATH = "repo;.deps"
python scripts/derive_causal_residual_atlas.py
python -m pytest tests/test_causal_residual_atlas.py -q
```

Authority:

```text
vortex_runtime/causal_residual_atlas.py
scripts/derive_causal_residual_atlas.py
tests/test_causal_residual_atlas.py
```

Ten focused tests currently pass. They cover pair-only `WQ` construction,
committed-prefix exact
reconstruction, late independent prefix directions, safe unread bounds,
all-page exact completion, strict top-1 certification, page rounding, build
amortization, corrupt/non-finite failure, and randomized no-false-bound cases.

## Claim boundary

Structurally valid conditions and a synthetic reference were established.
Real causal residual concentration, end-to-end numerical enclosure, output
certificate coverage, unchanged small-checkpoint operation replacement,
native BF16/Q4 equivalence, page latency, complete 8 GiB peak state, 122B/405B
execution, and E2-E7 remain `NOT TESTED`.
