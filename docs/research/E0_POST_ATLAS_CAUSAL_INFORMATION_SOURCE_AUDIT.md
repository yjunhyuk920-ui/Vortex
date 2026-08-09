# E0 Post-Atlas Causal Information-Source Audit

- Date: 2026-08-09 Asia/Seoul
- Status: COMPLETE E0 AUDIT; NO SURVIVING CANDIDATE
- Evidence ceiling: algebraic/reference E0 plus a labeled post-hoc EXP-083B diagnostic
- Hardware/model action: no model forward, download, private-server command, or GPU action

## Question

After EXP-083B rejected the common-spectral one-page Causal Residual Atlas,
does a materially different causal, checkpoint-derived, lossless source reveal
the signed unread contribution cheaply enough to close the complete registered
405B equation?

Rank, page, layer, selector, spectral-slack, and global-ball variations are not
new sources.  The strongest distinct idea is to stop reconstructing `W x` and
compute only decision functionals `v^T W x`, where `v` is a causal direction
that separates the proposed token from a competitor.

This audit names that interface a **Decision-Directional Source**.  Its most
favorable concrete form is a **Causal Decision-Dual Code**.

## Exact bilinear normal form

Let committed primal directions form an orthonormal basis `Q`, and let paid
causal decision directions form an orthonormal basis `P`.  Store both exact
checkpoint images:

```text
Z_x = W Q
Z_v = W^T P.
```

For the current activation and decision direction,

```text
x = Q a + u
v = P b + r.
```

Then:

```text
v^T W x
  = v^T Z_x a
  + (Z_v b)^T u
  + r^T W u.
```

The first two terms are exactly supplied by the primal and dual images.  The
last term is the **Bilinear Cross Residual**.  It is not numerical roundoff and
cannot be dropped as an approximation.

The missing-information claim has a direct indistinguishability witness.  For
nonzero `u` and `r`, choose

```text
Delta W = lambda r u^T.
```

Because `u` is orthogonal to `Q` and `r` is orthogonal to `P`,

```text
Delta W Q   = 0
Delta W^T P = 0,
```

while

```text
r^T Delta W u = lambda ||r||^2 ||u||^2 != 0.
```

Two checkpoint operators can therefore expose identical cached primal and dual
images yet give different current decision functionals.  A sound executor must
obtain the cross residual from paid lossless state, paid checkpoint probes, an
equivalent exact constructor, or fail-closed dense completion.

This theorem also sharpens the Atlas boundary.  Forward input/image pairs alone
are the case with no paid dual span and reduce to the same omitted-contribution
problem already rejected by EXP-083B.  Calling their bound directional does not
create new information.

Reference implementation and six controls:

```text
vortex_runtime/post_atlas_causal_source.py
tests/test_post_atlas_causal_source.py
```

## Dynamic causal dual-image constructor

Inference does not naturally compute `W^T v`.  For a prompt/token-dependent
direction through a nonlinear suffix, an exact raw-checkpoint constructor must
read the relevant dense coefficients unless another lossless query structure
already exists.  That other structure would itself be the missing invention
and cannot be granted free.

Grant one complete target-equivalent transpose construction for the entire
model, grant its state and every hit operation free, and amortize it over `N`
service tokens.  The traffic and operation lower term is:

```text
R_dual_build >= s / N,
```

where `s` is the number of fresh exact full-model dual directions.  Against the
registered p50 allowance `8/675 = 1.185185185%`:

| Fresh dual directions | Minimum `N`, all other costs free |
|---:|---:|
| 1 | 85 |
| 2 | 169 |
| 4 | 338 |
| 8 | 675 |
| 16 | 1,350 |

At the Atlas-registered 64-token service life, one direction alone costs
`1/64 = 1.5625%`.  The retained verifier raises the one-direction minimum to
109 tokens.  Adding one full dual construction to the already registered
Atlas traffic raises it to 999 tokens.  Every selector, basis application,
nonlinear relaxation, cross-residual query, miss, and fallback is still zero
in these grants.

Decision directions before the last linear slice depend on current nonlinear
activation, attention, and normalization state.  Reusing one static vocabulary
row does not supply those prompt/token-dependent pullbacks.  CROWN-style bound
propagation and branch-and-bound correctly manipulate primal/dual bounds, but
their backward affine propagation is verifier work, not a free causal source;
see [CROWN](https://arxiv.org/abs/1811.00866) and
[beta-CROWN](https://arxiv.org/abs/2103.06624).

Decision:

```text
REJECT_DYNAMIC_CAUSAL_DECISION_DUAL_CODE_UNDER_REGISTERED_64_TOKEN_CONSTRUCTOR
```

This is scoped to construction from raw checkpoint coefficients.  It does not
prove that every preprocessed exact bilinear query data structure is impossible.

## Static vocabulary-indexed dual code

At the final bias-free RMSNorm/LM-head slice, the positive common RMS scalar
cancels from exact-real greedy ordering.  A competitor direction can therefore
be composed through one last `down_proj`.  Avoiding a dynamic transpose by
precomputing every vocabulary direction gives the favorable shape:

```text
rows                  vocabulary V = 128,256
columns               MLP width I = 53,248
composite coefficients            = 6,829,375,488 per last down
favorable two-byte representation = 12.720703125 GiB per last down
full scan / target Q4 bytes        = 6.765979992%
registered target                 = 1.185185185%
miss factor                       = 5.708795618x
```

Applying the same table to all 126 down projections would store at least
`1.565242767 TiB` under the unrealistically favorable two-byte exact grant and
scan `8.525134790x` the complete registered non-embedding Q4 bytes per token.
Composed native values generally require a wider value or a correction code,
so this is a lower-cost screen, not an implementation estimate.

An exact MIPS/point-location index could in principle avoid scanning all rows,
but no finite exact high-dimensional index with the required storage, build,
query, and arbitrary-checkpoint guarantees is supplied here.  Approximate or
high-probability MIPS does not preserve the declared deterministic contract;
for example [BanditMIPS](https://arxiv.org/abs/2212.07551) explicitly trades
accuracy against work.  Even a final-slice index would replace only a small
suffix and does not construct the prompt-dependent decision directions needed
through earlier nonlinear layers.

Decision:

```text
REJECT_STATIC_FULL_VOCABULARY_DUAL_SCAN_AS_CORE
KEEP_EXACT_SUBLINEAR_DECISION_INDEX_UNSUPPORTED_NOT_REJECTED_UNIVERSALLY
```

## EXP-083B first-row directional diagnostic

The frozen EXP-083B result already contains the candidate/native pre-RMSNorm
vectors, final gain, BF16 tied head, and logits.  A read-only post-hoc script
performs zero model forwards:

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\audit_exp083b_decision_directions.py
```

Observed on `legal_holdout_english_01`:

```text
candidate/native BF16 winner                 21461 / 21461
candidate/native exact-linear winner         21461 / 21461
candidate exact-linear top-two gap           1.7764701843
native exact-linear top-two gap              1.3662271500
actual candidate/native pre-norm distance    3.5124788056
effective winner-row norm                    2.6583087516
competitors                                  248,319
unresolved using actual distance             248,319
unresolved using frozen radius               248,319
model forwards                               0
```

Thus the obvious hybrid -- compute a few exact dual rows and use an effective
LM-row norm for every remaining competitor -- selects every competitor even
when illegally granted the observed actual candidate/native distance instead
of the larger frozen radius.  Its full vocabulary composite scan is not an
artifact of the conservative EXP-083B RMS implementation envelope.

This diagnostic is not a preregistered population Gate and does not reject all
correlated exact MIPS indexes.  It rejects the proposed few-directions-plus-
norm continuation on the already observed row.

## Other screened sources

| Source | E0 disposition |
|---|---|
| More committed forward pairs | Atlas normal form; no new cross-residual information |
| Reverse/adjoint trace from a dense target pass | New information, but the local proposer/build is the dense work being avoided |
| CROWN/branch-and-bound proof state | Sound certificate machinery; prompt-dependent backward propagation remains charged |
| Freivalds/GKR/SNARK trace | Verifies a supplied result; F-048 still lacks the result source |
| Approximate MIPS or randomized coordinate sampling | Does not meet the frozen deterministic exact contract without a separate sound fallback equation |
| Static point-location over all reachable Transformer states | Enumerative advice/decision-diagram regime; build and state are not finite-target-feasible |

The vector-matrix-vector query model is studied directly by
[Rashtchian, Woodruff, and Zhu](https://arxiv.org/abs/2006.14015), and links
between systematic linear data structures and rigidity are developed by
[Ramamoorthy and Rashtchian](https://arxiv.org/abs/1910.11921).  These works
help define the query interface and restricted lower-bound models.  They do
not provide the finite exact numerical Transformer source required here, and
this audit does not promote them into a universal impossibility theorem.

## E0 scorecard

1. Target-scale upside: a truly sub-dense signed cross-residual source could
   avoid full intermediate reconstruction, so the abstract class has enough
   upside.
2. Novelty: paid dual images are materially different from an Atlas rank or
   bound change.
3. Evidence basis: exact bilinear identity and indistinguishability controls
   pass; no favorable population coverage exists.
4. Scaling: dynamic build cost remains one dense-equivalent pass per fresh
   direction; static vocabulary composites grow as `V*I*L`.
5. Universality: prompt-dependent nonlinear pullbacks prevent a fixed finite
   direction list from covering arbitrary checkpoints.
6. Correctness: the cross residual must be exact, soundly certified, or sent to
   unchanged fallback.
7. Resource closure: both currently constructible variants fail before full
   state, verification, miss, and fallback costs.
8. Cheap falsification: the conservation equation and frozen first-row
   direction screen reject the two concrete constructors before an experiment.

## Decision and remaining frontier

```text
REJECT_FORWARD_TRACE_ONLY_POST_ATLAS_SOURCE_AS_ATLAS_NORMAL_FORM
REJECT_DYNAMIC_CAUSAL_DECISION_DUAL_CODE_UNDER_REGISTERED_64_TOKEN_CONSTRUCTOR
REJECT_STATIC_FULL_VOCABULARY_DUAL_SCAN_AS_CORE
KEEP_LOSSLESS_SUBDENSE_BILINEAR_CROSS_RESIDUAL_SOURCE_OPEN_WITH_NO_CONSTRUCTION
RETURN_TO_NO_SURVIVING_CANDIDATE
```

No new experiment number is assigned.  No model, runtime, kernel, larger
checkpoint, target server, or hardware stage is authorized.

The next admissible invention must answer the exact remaining question: how can
`r^T W u` be obtained losslessly for prompt-dependent causal residual pairs
without a dense transpose/forward construction, a full static vocabulary scan,
training, target-future leakage, or hidden dense discovery?  Until that source
has a finite operation/traffic/state/build/fallback equation, the honest status
is `NO_SURVIVING_CANDIDATE`.
