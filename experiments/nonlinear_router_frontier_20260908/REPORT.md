# E0 nonlinear adaptive router cover gate

Date: 2026-09-08

## Verdict

```text
REJECT_FULLY_NONLINEAR_ADAPTIVE_TWO_WORD_25x108
REJECT_EVERY_TARGET-FEASIBLE_64BIT_WORD_ROUTER_WITH_BOTH_SIDES <= 128
REJECT_24x225_AND_25x216_LOCAL_ROUTE_UNDER_STRONG_INDEPENDENT_32_QUERY_UNION
KEEP_LEGAL_CAUSAL_REACHABILITY_OF_THE_HARD_32_QUERY_TUPLE_OPEN
DO_NOT_CLAIM_NATIVE_NUMERICAL_OR_GLOBAL_MODEL_CLOSURE
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
```

This is a new exact information-flow Gate. It is not a low-cost VORTEX
producer. Its value is that it removes the strongest previously open local
bounded-word seed without assuming linear storage or nonadaptive addressing.

## 1. Why this round was needed

The previous arbitrary-nonlinear cell-support degree Gate left a concrete
capacity survivor:

```text
source       25 x 108 binary matrix = 2,700 bits
storage      50 padded 64-bit words = 3,200 bits
query        every nonzero rank-one parity r^T W u
target       two 64-bit probes
traffic      128 / (4*2700) = 8/675
```

That result was only a necessary-dimension screen. It supplied no encoder,
address generator or decoder. The current round asks whether *arbitrary
nonlinear* cells and *value-adaptive* addresses could actually inhabit that
capacity gap.

## 2. Three principle comparison

The preregistration compared three different information-flow reversals.

1. **Checkpoint-space nonlinear words.** Compile arbitrary checkpoint
   functions into a near-source set of bounded words and answer a rank-one
   query from a tiny adaptive probe path. This was selected because its
   favorable payload can genuinely eliminate more than 90% of the local
   source traffic.
2. **Query-space joint restriction.** Generate `W` only on the 32-query factor
   envelope `R tensor U`. The envelope has dimension at most 1,024, but the
   repository has no implicit generator. A literal factor-space catalog is
   already rejected and evaluating `R^T W U` from raw `W` merely restores the
   dense contraction, so no hidden generator was promoted.
3. **Arithmetic-space native transition maps.** Compose exact reference-order
   rounded accumulator transitions. A literal activation-pattern transition
   catalog is the already rejected exponential finite-state table, while
   forming the transition from every activation contribution at query time
   retains the dense work. No non-table sub-dense representation was supplied.

P1 was therefore the only principle with a concrete finite target not already
invalidated by its own explicit implementation.

## 3. The fully nonlinear adaptive cover theorem

Let

```text
V = F2^D
E_j : V -> {0,1}^w, j in [S]
```

be *arbitrary* stored-cell functions. A deterministic query algorithm for a
linear parity query may choose each address after the first from the values
already read and may apply arbitrary final Boolean logic.

### 3.1 Largest-fiber recursion

Partition queries by their first address. Inside one query group, select a
value of that cell whose source fiber is largest. It retains at least
`2^(D-w)` source points. At the next depth, the fixed prior value makes every
query's next address definite; partition again by that address and choose its
largest value fiber inside the retained set. After `t-1` probes the retained
source set satisfies

```text
|F| >= 2^(D-(t-1)w).
```

There are at most `S^t` final route groups.

### 3.2 One final word limits the query span

Fix one final route group and let `L` be the binary span of all its query masks,
with `r=dim L`. On the retained source set `F`, every query answer is a function
of the final `w`-bit word, hence the complete answer vector assumes at most
`2^w` patterns.

Projection onto a basis of `L` is a rank-`r` linear map on all `V`; each global
projection fiber contains exactly `2^(D-r)` sources. Therefore

```text
|F| <= 2^w * 2^(D-r).
```

Together with the largest-fiber lower bound,

```text
2^(D-(t-1)w) <= 2^w 2^(D-r)
r <= t*w.
```

So every exact depth-`t` adaptive nonlinear router implies

```text
rank-one query family
  subset union of at most S^t linear subspaces,
each subspace dimension <= t*w.
```

This is stronger than the preregistered affine-first theorem. The affine
hypothesis was deleted only in the post-preregistration extension, not by
rewriting the preregistration.

## 4. Exact rank-one intersection

For a binary `m x n` matrix space, normalize `m<=n`. The rank-one points are
the Segre product of the two binary simplex projective systems. The repository's
existing Schaathun/product-code generalized-weight implementation gives the
maximum rank-one intersection of a `d`-dimensional linear subspace.

Writing

```text
d = q*n + r, 0 <= r < n,
```

the exact intersection is

```text
M(m,n,d) = (2^q - 1)(2^n - 1) + 2^q(2^r - 1).
```

The new module contains both the closed form and an independent deficit DP.
The DP matches the pre-existing generalized-weight controls and directly
matches the `24 x 225, d=256` frontier value.

## 5. Registered 25 x 108 result

For `w=64`, `t=2`, each route group's query span has dimension at most 128.
The exact intersection is

```text
M(25,108,128)
  = 2^108 + 2^21 - 3
  = 324518553658426726783156022673405.
```

Fifty cells create at most `50^2=2,500` adaptive route groups. Even the union
bound therefore covers at most

```text
811296384146066816957890056683512500
```

rank-one queries, while the required family has

```text
(2^25-1)(2^108-1)
= 10889035416951477172401260654660528635905.
```

The favorable coverage ratio is only

```text
0.0000745058081896844...
```

so the fully nonlinear, fully value-adaptive two-word scheme is impossible in
this exact binary model. This is not merely rejection of linear/affine cells.

## 6. Complete side <= 128 target scan

The same theorem was applied to every normalized rectangle with both sides at
most 128. Storage uses the repository's favorable proportional global-advice
allocation and 64-bit padding. For each rectangle the Gate grants the *maximum*
number of whole 64-bit probes that still satisfies the registered `8/675`
traffic line against four favorable Q4 lanes.

```text
rectangles checked                 8,256
zero-64-bit-probe physical budget  2,316
adaptive-cover rejected            5,940
unclosed                            0
```

Since `S^t M(m,n,tw)` is nondecreasing with `t`, rejection at the maximum
allowed probe count rejects every smaller probe count too.

Thus the previous `25 x 108` survivor was not an isolated miss: **there is no
target-feasible block-local deterministic adaptive 64-bit nonlinear word
router anywhere in the complete side-128 scan.**

## 7. New first local word frontier

An exact area-ordered scan through area 5,400 checked 23,669 normalized
rectangles before the first unclosed area. Exactly two minimum-area shapes are
not rejected by this Gate:

```text
24 x 225 : D=5400, granted bits=6319, padded words=99, max probes=4
25 x 216 : D=5400, granted bits=6319, padded words=99, max probes=4
traffic   : 4*64/(4*5400) = 8/675 exactly
```

The Gate's union-bound headroom is still only finite:

```text
24 x 225 coverage upper / required = 5.725598736143037...
25 x 216 coverage upper / required = 2.862799282753446...
```

These ratios are **not constructions**. No 99-word encoder, four-probe
value-adaptive address generator or decoder was found or assumed.

### 7.1 The remaining four-probe route must use successive value adaptivity

The first area-5,400 survivors were then split into two cheaper submodels.
Stored cells and final Boolean decoding remain completely nonlinear in both.

For **nonadaptive** four-probe routing, grouping by the set of `j` distinct
cells gives at most `C(S,j)` query subspaces of dimension `j*w`. Summing the
exact Segre intersection separately for `j=1..4` rejects both shapes:

```text
24 x 225 coverage upper / required = 0.22437430765584512...
25 x 216 coverage upper / required = 0.11218715076256036...
```

For **one-value-stage adaptive** routing, the first word may be arbitrary
nonlinear and its observed value may choose the complete remaining payload set.
The later payload values are not allowed to change addresses again. Choosing a
largest first-word fiber and grouping the payload set gives at most
`S*C(S-1,j)` subspaces of dimension `(j+1)w`, `j=0..3`. This stronger model is
also rejected:

```text
24 x 225 coverage upper / required = 0.8974972306222921...
25 x 216 coverage upper / required = 0.44874860277162165...
```

Therefore any surviving 99-word/four-probe constructor at the new local
frontier must use **multiple successive value-dependent routing stages**. It
cannot be a nonlinear fixed-support decoder or a one-key dispatcher followed
by three fixed payload reads.

### 7.2 Complete-query active-word descriptor Gate

The remaining multi-stage route can be tested against the repository's stronger
independently-selectable 32-query service interface without restricting the
cell contents or address adaptivity.

For one source `W`, let `U(W)` be the union of every encoded-cell address touched
while the decoder answers **all** nonzero rank-one queries on that source. The
query family includes every matrix unit, so the complete answer vector uniquely
identifies every source bit.

The descriptor

```text
(U(W), values of the encoded cells in U(W))
```

is therefore injective. To see this, suppose `W'` agrees with `W` on every cell
in `U(W)`. For any query, induction over the adaptive probes shows that `W'`
sees the same returned values and takes the same addresses as `W`; every address
on the `W` path lies in `U(W)`. Thus all rank-one answers agree, in particular
all matrix-unit answers, and `W'=W`.

If every source had `|U(W)|<=u`, the number of possible favorable descriptors
would be at most

```text
sum_(j=0..u) C(S,j) 2^(w*j).                            (11)
```

Equation (11) intentionally counts unreachable address/value combinations too.
For the area-5,400 frontier, `S=99`, `w=64`:

```text
sum_(j<=83) C(99,j) 2^(64j)  < 2^5400
sum_(j<=84) C(99,j) 2^(64j) >= 2^5400.
```

Hence every exact deterministic 99-word representation has **some source** for
which the complete rank-one query family activates at least 84 distinct cells.

From a union of at least 84 cells, greedily choose a rank-one query that adds a
new cell until 32 queries have been chosen (or the whole union has already been
covered). Therefore some independently-selectable 32-query tuple touches at
least 32 distinct encoded words. Even granting perfect reuse inside that tuple,
the payload is at least

```text
32 * 64 = 2,048 bits
2,048 / (4 * 5,400) = 64/675
(64/675) / (8/675) = 8.
```

So the remaining local 99-word route is **8x over the registered logical
payload line under the strong independent-32-query interface**, even with
arbitrary nonlinear cells, fully value-adaptive addresses and arbitrary
deterministic final logic.

This is not a batch-1 causal reachability theorem. The repository has not proved
that an ordinary legal Transformer continuation realizes the adversarial
independently-selected rank-one tuple. It also does not assign the 32 words to
SSD versus HBM or prove wall-clock latency. Those boundaries remain explicit.

## 8. Strongest adversary and cheapest falsifier

The adversary is the complete source universe of the declared binary rectangle
and every nonzero rank-one query, including all matrix units. No distribution,
checkpoint compressibility, low rank, row reuse or good prompt is assumed.

The falsifier is exact integer arithmetic:

```text
if S^t * M(m,n,min(D,t*w)) < (2^m-1)(2^n-1),
then the declared adaptive nonlinear word router cannot exist.
```

No model forward or hardware action is needed.

## 9. Cost and mission boundary

The theorem is deliberately favorable to the candidate. It grants:

- arbitrary nonlinear preprocessing functions;
- arbitrary deterministic address logic and decoder logic for free inside the
  E0 information Gate;
- perfectly packed word payloads;
- no metadata/cache-line/alignment cost;
- no native numerical lift cost.

Therefore a rejection is robust inside this binary block model. A survivor is
not a physical upper bound.

Still unresolved and charged by the fixed mission:

- whether the hard independent 32-query tuple can be forced by a legal causal
  Transformer continuation, or a causal-specific producer can avoid it;
- global cross-matrix nonlinear advice and its probe union;
- Q4/BF16/FP32 products, carries, native accumulation order and rounding;
- complete logits/KV/RNG successor state;
- all legal HF masks/backends/contexts/checkpoints;
- constructor time, selector metadata, persistent storage, RAM/SSD/PCIe/HBM,
  workspace and initialization amortization;
- actual 405B execution, one-GPU <=8 GiB, native4BQ4 p50/p95 and TTFT.

## 10. O1-O6

```text
O1 OPEN     no arbitrary-checkpoint native producer constructor
O2 OPEN     no complete causal loading/prefill/decode program from this Gate
O3 OPEN     binary parity theorem is not native finite-word model/state/RNG proof
O4 OPEN     no sufficient whole-model paid upper bound
O5 OPEN     no target hardware/latency execution
O6 PARTIAL  theorem, code, exact raw result and focused replay are checkable
```

The whole theory remains incomplete.

## 11. Validation and artifacts

Focused new tests:

```text
15/15 PASS
```

Related regression:

```text
28/28 PASS
```

It includes the new tests plus the previous adaptive-nonlinear degree and joint
Segre-factor geometry suites. No native-global 3,510-file replay was rerun.

Authoritative result:

```text
results/e0_nonlinear_router_cover_gate_v5/summary.json
SHA256 af113c03b561827fc4ac93b1beb7216eb9de9cc9de17c95e6660b681c66f03d2
```

The earlier `results/e0_nonlinear_router_cover_gate` output with SHA
`a69d0e2950c94001a7bbbeaeefbdec50473e8535387503013de721530f8a8dd2`
and `results/e0_nonlinear_router_cover_gate_v2` with SHA
`7f98edc8a7c18cde7f09e18c5261651e1207d5c0dc665f24cf3a539350a25415`
and `results/e0_nonlinear_router_cover_gate_v3` with SHA
`d3a49ae6487f79ec17848feac949741ca01c14493a65baa0b7b103bd06cbd581`
and `results/e0_nonlinear_router_cover_gate_v4` with SHA
`0e82591f0e4748852644abecac2b33c5cca7683917e2e4cd9bf8c9d057bdf981`
are retained as weaker pre-strengthening/pre-refinement results rather than
overwritten.

Reproduction:

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
& '.\experiments\native_global_transition_20260908\.venv\Scripts\python.exe' `
  -m unittest tests.test_nonlinear_router_cover_gate -v

& '.\experiments\native_global_transition_20260908\.venv\Scripts\python.exe' `
  scripts\derive_nonlinear_router_cover_gate.py `
  --output-dir results\e0_nonlinear_router_cover_gate_replay
```

The replay destination must be absent or empty.

## 12. Literature boundary checked in this continuation

Recent work did not supply the missing native producer. Chakraborty, Kamma and
Larsen's succinct Boolean MatVec upper bound is randomized and Boolean-semiring;
their F2 vector-matrix-vector result is a lower bound, not a zero-error F2
producer. Thus it does not lift the native ordered Q4/BF16/FP32 contract.
Young Kun Ko's 2025 result is important because it proves nontrivial static
cell-probe lower bounds even against nonlinear preprocessing for random linear
operators, but its quantitative statement does not close this VORTEX parameter
regime. Korten, Pitassi and Impagliazzo (2025) improve explicit static
cell-probe lower bounds while also giving barrier evidence around substantially
stronger constant-probe bounds; Loff, Koucky, Molli and Saks (STOC 2026) give a
natural-proofs barrier for data-structure lower bounds. These results are a
warning not to manufacture a universal four-probe impossibility theorem from
an unsupported step. Anand, van den Brand and McCarty's structural MatVec route
requires matrix structure such as bounded VC/Pollard dimension and therefore
does not instantiate the arbitrary-checkpoint producer either.
