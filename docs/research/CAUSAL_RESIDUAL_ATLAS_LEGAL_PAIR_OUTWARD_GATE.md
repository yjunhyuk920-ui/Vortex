# Causal Residual Atlas Legal Pair and Outward-Bound Gate

- Date: 2026-08-09 Asia/Seoul
- Status: PREREGISTERED, NO CHECKPOINT RESULT, NO EXPERIMENT NUMBER
- Evidence ceiling if executed: E1 leakage-safe small-real-checkpoint certificate falsification

## Question

Can the favorable page-existence result from EXP-083A survive the first slice
that an executor could actually use, when all of the following are mandatory?

- the rank-16 basis and its image are constructed from committed prefix
  input/image pairs only;
- page selection receives only the current causal input residual;
- the selected page plus every unread contribution is enclosed by outward
  numerical bounds;
- the bound reaches the pinned native BF16 greedy token, not merely one local
  projection norm; and
- capsule state, proof metadata, construction, selection, static norm
  compilation, verification, exact completion, and fallback are charged.

This document freezes the smallest such Gate before another checkpoint call or
experiment number. It does not claim that the Gate will pass.

## Frozen decision vocabulary

```text
PROMOTE_CAUSAL_RESIDUAL_ATLAS_TO_BACKWARD_LAYER_AND_POSITION_EXPANSION_GATE
REJECT_CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_PATH
INVALID_LEGAL_PAIR_OUTWARD_CONTROL_FAILURE
INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION
```

Promotion would mean only that one legal last-layer slice survived. It would
not admit an E2 executor, cold scheduler, hardware path, larger checkpoint, or
405B feasibility claim.

## Observation quarantine

The 18 EXP-083A evaluation prompts, all 1,296 candidate logits, oracle page
choices, and the post-hoc minimum-radius failure are now observed. None may:

- choose or tune the new selector;
- choose the bound family or numerical slack;
- choose layer, projection, token position, rank, page width, or thresholds;
- determine promotion or rejection in this Gate.

They remain provenance evidence only. The scientific population is a newly
written set of 24 prompts that had not been executed against the checkpoint
when this contract was committed:

```text
path    docs/research/inputs/causal_residual_atlas_legal_gate_prompts.json
SHA-256 67bd16d4f1e63a6f4e4a9131119335aff8d4d676549ce55287bf69b8fb879fb5
rows    24, four in each of six frozen families
```

The prompt text itself is public. “Unseen” means that no checkpoint output,
activation, page result, or certificate result for these rows existed before
preregistration. A hash mismatch invalidates the run.

## Why the last `down_proj` is the cheapest honest certificate

The first Gate uses only:

```text
checkpoint                 Qwen/Qwen3.5-0.8B, unchanged pinned BF16 payload
revision                   2fc06364715b967f1860aea9cf38778875588b17
weight SHA-256             04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
evaluation prompts         24 new rows
teacher position           1, first genuine post-prefill module call
layer                      23, final full-attention decoder layer
projection                 down_proj only
rank                       16
page width                 64 native input columns
pages                      56
selected pages             exactly one
```

After layer-23 `down_proj`, the declared token depends only on the residual
addition, final RMSNorm, and tied LM head. This avoids pretending that a
product of loose global Lipschitz constants through twelve mixed
attention/DeltaNet layers is useful. It is also a necessary slice: if a legal
one-page source cannot certify the final `down_proj`, adding earlier nonlinear
uncertainty cannot repair it.

Layer 23 was selected analytically from topology before its page behavior was
observed. A pass permits a new Gate to move backward through layers and across
the eight frozen decode positions. A failure stops the Atlas primary path.

## Causal generation boundary

For each prompt:

1. run unchanged prompt prefill and retain the immutable cache;
2. record committed prefix projection input/image pairs in prefix order;
3. take the unchanged prompt-last greedy token as the first committed decode
   input;
4. construct the Atlas before processing that token at layer 23;
5. choose one page without a native current `down_proj` result or any logits;
6. run the selected candidate to its own final logits;
7. commit only if the outward certificate proves the native greedy winner;
8. otherwise execute the unchanged dense `down_proj` and suffix from the
   immutable branch point, or abort without a token.

The native baseline logits are evaluator-only evidence. They may check the
certificate and calculate KL after the selector and certificate verdict are
immutable; they may never enter either computation.

## Frozen legal pair compiler

### Pair source

For one unchanged matrix `W`, committed rows are pairs `(x_s, y_s)`. The exact
real relation is `y_s = W x_s`; native BF16 rounding is not silently treated
as linear and is enclosed separately below.

The compiler scans pairs in original causal order and performs two-pass
modified Gram-Schmidt. For every candidate pair:

```text
r <- x_s
z <- y_s
for pass in (1, 2):
    c <- Q^T r
    r <- r - Q c
    z <- z - Z c
h <- ||r||_2
if h clears the frozen numerical-independence threshold:
    append q = r/h to Q
    append z = z/h to Z
stop after 16 accepted columns
```

The threshold is

```text
tau_s = max(input_width, prefix_rows) * eps_float64 * max(1, ||x_s||_2).
```

An exact lower norm enclosure must satisfy `h_lower > tau_s`; equality or an
interval crossing the threshold is not accepted. Pair order and both passes
are fixed. If fewer than 16 columns are certified, the row is not excluded: it
falls back and therefore fails the zero-fallback population Gate.

`Z` receives the identical coefficients and normalizer as `Q`; its constructor
has no `W` parameter. Thus, in exact arithmetic, `Z = WQ` follows from the
committed pairs. The Gate stores BF16 `Q_hat` and `Z_hat`, plus outward scalar
defect metadata. It does not claim that native rounded prefix images obey exact
linearity.

### Native numerical enclosure

The pinned numerical contract is BF16 inputs and weights, FP32 dot
accumulation, and BF16 output rounding. For width `n`,

```text
gamma_n = n*u32 / (1 - n*u32),  u32 = 2^-24.
```

For a prefix input `x`, verified `beta_W >= ||W||_2`, and verified Frobenius
bound `F_W >= ||W||_F`, the registered L2 image error is

```text
e_native(x)
  >= gamma_n F_W ||x||_2
   + u_bf16 (beta_W ||x||_2 + gamma_n F_W ||x||_2)
   + sqrt(output_rows) * 2^-134,

u_bf16 = 2^-8.
```

All operations round outward. The evaluator recomputes exact-real float64
prefix images only to test this envelope. Any native row outside it is a
control failure; the bound may not be widened after observation.

Let `E` collect the prefix-image errors, `C` be the outward-enclosed pair
coefficient map created by Gram-Schmidt, `delta_Q` bound stored-basis error,
and `delta_Zstore` bound stored-image error. The compiled image defect is

```text
delta_Z
  >= beta_W * delta_Q
   + ||E||_F * ||C||_2
   + delta_Zstore
  >= ||W Q_hat - Z_hat||_2.
```

Only this scalar and pair-derived `Q_hat/Z_hat` enter the current query.

## Frozen verified operator bound

For every checkpoint matrix, the static compiler proposes `beta_W` and proves

```text
beta_W^2 I - G_W > 0,

G_W = W^T W when columns <= rows,
G_W = W W^T when rows < columns.
```

The two choices have the same largest eigenvalue and the compiler uses the
smaller Gram dimension. It applies outward inclusion and a
positive-definiteness verifier. If
verification fails, `beta_W` is increased before any prompt is evaluated; a
failed or missing certificate mandates dense fallback. The method follows the
verified spectral-norm construction in Siegfried Rump, “Verified Bounds for
Singular Values, in Particular for the Spectral Norm of a Matrix and Its
Inverse,” BIT 51 (2011):
<https://www.tuhh.de/ti3/paper/rump/Ru10a.pdf>.

The paper's full-matrix route includes an outward Gram matrix and two Cholesky
verifications. The target equation charges, for each `large x small` matrix,

```text
4 * large * small^2
+ ceil(2 * small^3 / 3)
+ 10 * small^2
+ 10 * small
```

logical scalar operations. A future implementation must report a larger
actual count when applicable; it may not replace this charge with a plain
non-verified SVD call. The exact-dyadic LDL reference is intentionally bounded
to small synthetic controls.

## Frozen target-free selector

For current input `x`:

```text
a = Q_hat^T x
u = x - Q_hat a
```

For each contiguous 64-column page `G`, compute only
`e_G = sum_(i in G) u_i^2`. Select

```text
G* = argmax_G e_G,
```

breaking an exact tie by the lowest page index. The selector function accepts
only `u` and `page_columns`; weights, native current outputs, candidate
outputs, target logits, and candidate logits are absent from its interface.

This rule is not a heuristic fitted to EXP-083A. It follows from the common
operator certificate:

```text
||W[:, unread] u[unread]||_2
  <= ||W[:, unread]||_2 ||u[unread]||_2
  <= beta_W ||u[unread]||_2.
```

Deleting columns is right multiplication by an orthogonal coordinate
projection, so its operator norm cannot exceed one. With the same `beta_W`
for every page, removing the largest residual energy minimizes this registered
upper bound. This replaces the failed sum of per-page Frobenius products; it
was chosen from the inequality, not from the observed page winners.

## Legal selected-page output and local radius

Only after `G*` is immutable does the executor read that weight page and form

```text
y_hat = Z_hat a + W[:,G*] u[G*].
```

Let `r_page_arith`, `r_candidate_cast`, and `r_native_dense` be outward
arithmetic/cast radii. The projection-output comparison radius is

```text
r_down
  = delta_Z ||a||_2
  + beta_W ||u[unread]||_2
  + r_page_arith
  + r_candidate_cast
  + r_native_dense.
```

Every addition, multiplication, square, sum, square root, and norm endpoint is
directed outward. NaN, infinity, negative radius, malformed page state,
unverified `beta_W`, or a missing defect term fails closed.

## Declared-output certificate

### Residual addition and final RMSNorm

The selected BF16 `down_proj` result is added at the immutable native residual
branch. Both candidate and target addition-rounding errors are added to
`r_down`. Let the resulting candidate center be `h_hat` and radius `r_h`.

For RMSNorm

```text
f(h) = D h / sqrt(mean(h^2) + epsilon),
```

the Jacobian norm of the unweighted normalization is at most the reciprocal
denominator. On the complete L2 ball,

```text
s_min = sqrt(max(||h_hat||_2 - r_h, 0)^2 / d + epsilon)
L_rms  = max_i |D_ii| / s_min
r_norm = L_rms * r_h + two_path_BF16_cast_radius.
```

The lower denominator and upper radius are outward. This local bound is used
only while the ball remains finite; otherwise the row falls back.

### Tied LM-head margin

The candidate executes the ordinary tied LM head once. Static outward row
norms `n_j >= ||w_j||_2` and per-logit native dot/cast radii `d_j` give

```text
E_j = n_j * r_norm + d_j.
```

If candidate winner `k` satisfies the strict condition

```text
candidate_logit[k] - E_k
  > candidate_logit[j] + E_j  for every j != k,
```

then and only then the Gate certifies `k`. A tie, zero lower margin, non-finite
value, or incomplete vocabulary scan is unresolved. The evaluator compares
the certified winner with the unchanged native winner; any mismatch is an
invalid control and a false accept count greater than zero rejects promotion.

This is a certificate for the pinned greedy decoding contract. It does not
claim exact logit equality or sampled-output equivalence.

## Fully charged favorable 405B component equation

The deterministic calculator is:

```powershell
$env:PYTHONPATH = ".;.deps"
python scripts/derive_causal_residual_atlas_legal_gate.py
```

It retains the registered rank-16, one-page, 64-token Atlas point and adds:

- two-pass pair-only Gram-Schmidt construction;
- one read of the 16 selected committed pairs and one capsule write;
- three outward scalars per matrix, two per activation site, and one LM-head
  row norm per vocabulary row;
- a full vocabulary bound scan;
- verified spectral compilation amortized over an explicitly favorable
  `20,000,000` checkpoint service tokens; and
- exact dense fallback through the existing coverage equation.

Frozen derived values are:

```text
proof metadata scalars / bytes          131,915 / 1,055,320
pair build operations                 12,053,860,352
pair build traffic bytes               2,106,671,104
dynamic traffic fraction                 1.093701271771%
dynamic operation fraction               0.581348512634%
verified spectral compile operations  28,052,251,108,674,513
compile dense-token equivalents          69,479.6215490220
registered model service tokens          20,000,000
charged traffic fraction                  1.093706271797%
charged operation fraction                0.928746620379%
target fraction                            1.185185185185%
traffic-governed minimum coverage         99.908521086612%
operation-governed minimum coverage       99.743561435194%
minimum service tokens from operations    11,506,361
capsule plus registered metadata           0.983090482652 GiB
unallocated part of 8 GiB                  7.016909517348 GiB
```

The `20,000,000` lifetime is a favorable, explicit amortization assumption,
not measured workload behavior. The compile formula is logical work, not
physical compiler time. The `7.0169 GiB` remainder still excludes KV, runtime
workspaces, selected-page buffers, fallback overlap, allocator headroom, and
the baseline runtime itself. Therefore this component equation prevents hidden
costs but is not a complete 8 GiB peak-memory pass.

For operations and traffic independently, unresolved tokens pay exact dense
fallback:

```text
R = charged_common + (1 - rho) * (miss + 1).
```

The favorable Gate grants `miss = 0`; even then traffic requires
`rho >= 0.9990852108661195`. This finite 24-row Gate consequently permits no
fallback. A larger miss cost can only make the threshold stricter.

## Frozen population and thresholds

A row succeeds only when all of the following hold:

- rank 16 is constructed from that prompt's committed pairs;
- the spectral and pair-image defect certificates validate;
- the target-free selector returns one legal page;
- the final strict native-BF16 greedy certificate succeeds;
- the certified winner equals the evaluator's unchanged winner;
- exact completion from the same branch point reproduces the unchanged
  winner; and
- no control, leakage, malformed-state, or arithmetic-model failure occurs.

Promotion requires:

```text
token successes                       24 / 24
successes in each family               4 / 4
certified branches                    24 / 24
false accepts                               0
fallbacks                                  0
mean target-to-candidate KL            <= 0.02
nearest-rank p95 target-to-candidate KL <= 0.05
```

KL is evaluator-only and is calculated after the immutable certificate
verdict. It cannot rescue an uncertified row. The first valid unresolved or
wrong row is a scientific rejection; the runner still freezes already
computed evidence and checksums.

The corpus cannot establish 99.9085% population coverage statistically. Zero
failures only allows the mechanism to survive this E1 falsification.

## Controls and failure classification

Control failure, with no scientific interpretation:

- checkpoint, prompt, code, dependency, or contract hash mismatch;
- any access by the pair compiler to `W` or the current decode pair;
- any access by the selector to a weight, native output, or logit;
- a native dense prefix image outside the frozen arithmetic enclosure;
- a spectral claim without a valid positive-definiteness proof;
- basis rank or page state changing under deterministic replay;
- dense completion or immutable-branch replay changing the baseline;
- a certified winner differing from the unchanged winner;
- corrupt/non-finite state not forcing fallback or abort.

Infrastructure failure, with no scientific interpretation:

- missing pinned payload or dependency;
- timeout, memory exhaustion, filesystem failure, or incomplete bundle;
- verified spectral compiler unavailable or unable to finish before a model
  row exists.

Scientific rejection after valid controls:

- any one of 24 rows cannot construct rank 16;
- the pair defect or unread radius is non-finite or cannot certify;
- any row falls back;
- any family or KL threshold fails; or
- the fully charged registered-shape equation no longer fits after actual
  costs replace favorable formulas.

## Cheapest-failure order

The future experiment must stop in this order:

1. verify hashes, untouched-population provenance, and native arithmetic;
2. verify the layer-23 `down_proj` spectral certificate;
3. construct pair-only `Q_hat/Z_hat` and their defect bounds;
4. choose the page from residual energy only;
5. run one selected-page candidate and final outward certificate;
6. stop on the first valid unresolved row;
7. only after 24/24 success compute aggregate KL and the decision.

It must not enumerate 56 candidate logits, inspect a better page, tune a
radius, change the rank/page/layer, or move earlier after a failure.

## Core twelve-question audit

1. **Operation replaced:** one real final-layer dense `down_proj` at the first
   post-prefill decode call.
2. **New information:** committed prefix input/image pairs and a static
   verified operator bound.
3. **Selector cost:** one residual-energy scan, fully present in the target
   equation; no oracle variants.
4. **Wrong-skip detection:** strict final greedy margin with outward RMSNorm
   and LM-head error.
5. **Fallback:** immutable-branch unchanged dense `down_proj` plus suffix.
6. **Silent corruption:** prohibited; every uncertainty falls back or aborts.
7. **Scaling:** model-wide component accounting fits favorably only with the
   explicit coverage and 20M-token compile amortization assumptions.
8. **Sequential work:** selection uses the current committed activation and no
   future token or dense current result.
9. **Movement:** pair/capsule reads, proof metadata, selected page, and
   fallback are all separate terms.
10. **405B minimum:** charged favorable traffic/operations are
    `1.093706%/0.928747%` before actual physical overhead.
11. **Remaining gap:** only `0.091479%` traffic headroom remains, demanding
    `99.908521%` coverage.
12. **Strongest falsification:** one untouched row whose strict final margin
    cannot be certified after the analytically optimal common-beta page.

## Stop rule and authorized continuation

On any valid scientific failure:

```text
REJECT_CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_PATH
```

Do not rescue it with another selector score, spectral slack, rank, page size,
layer, prompt subset, certificate threshold, or observed-page training. A new
mechanism would need a different causal information source and E0 equation.

On a complete pass:

```text
PROMOTE_CAUSAL_RESIDUAL_ATLAS_TO_BACKWARD_LAYER_AND_POSITION_EXPANSION_GATE
```

Only then may a separately preregistered Gate move backward to `q_proj`,
earlier full-attention/DeltaNet layers, all eight positions, and simultaneous
composition. E2 still requires actual complete-generation operation
replacement, full target state, held-out prompts, exact fallback, and measured
call/byte/work counts. No cold scheduler, CUDA kernel, target Ubuntu command,
35B/122B/405B download, or hardware benchmark is authorized here.

## Reference reproduction

```powershell
$env:PYTHONPATH = ".;.deps"
python scripts/derive_causal_residual_atlas_legal_gate.py
python -m pytest tests/test_causal_residual_atlas_legal_gate.py -q
```

The ten focused tests cover the frozen population and resource equation,
selector interface and tie rule, randomized no-false spectral residual bounds,
an exact-dyadic spectral proof, complete pair/radius term charging, randomized
RMSNorm balls, strict LM-head certification, and malformed arithmetic state.

## Claim boundary

This commit provides a leakage-safe preregistration and synthetic/reference
certificate machinery. It contains no result from the 24 new prompts. Legal
pair behavior on real weights, native arithmetic-envelope success, final
certificate coverage, simultaneous operation replacement, full 8 GiB peak,
physical speed, 122B/405B scaling, Phase D, and E2-E7 remain `NOT TESTED`.
