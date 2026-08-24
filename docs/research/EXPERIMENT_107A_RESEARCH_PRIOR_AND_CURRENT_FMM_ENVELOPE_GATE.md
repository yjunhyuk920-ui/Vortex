# EXP-107A — Research-Prior Batches and Current Explicit-FMM Envelope Gate

## Purpose

Apply the new no-homework research contract before implementing another executor. Three ideas are only the minimum search batch. A failed batch is followed by a new batch that reverses its common failure premise until one candidate is `GO` or `CHEAP_KILL_ONLY`.

This round asks whether the latest public small-format exact matrix-multiplication catalog already supplies enough arithmetic reduction for the registered 405B projection inventory, and whether the nearest border-rank route has a finite exactification with enough headroom.

## PRE_RESULT_PRIOR

The prior table was frozen before the local calculation.

### Batch 1 — preserve distinct branches and share one weight sweep

| Principle | Prior | Strongest reason to work | Strongest failure argument | Decision |
|---|---|---|---|---|
| coded exact branch superposition | LOW | one physical weight-tile read serves several branch rows | invertible coding preserves distinct outputs; without a new bilinear rank source, MACs, outputs, KV and `N` still scale with branches | `NO_GO` |
| finite-word nonlinear-cell coalescence | LOW | byte-identical control cells could merge branch work | an arbitrary legal checkpoint can place every branch in a distinct RMSNorm/SiLU/rounding cell | `NO_GO` |
| symbolic token-embedding contribution DAG | LOW | token-static work can be checkpoint-compiled | embedding/static token work is subdominant and prefix nonlinearities destroy separability | `NO_GO` |

Common failure premise:

```text
Distinct branches remain information-distinct.
Sharing the weight read does not share their value-changing arithmetic or state.
```

### Batch 2 — abandon branch sharing and attack exact arithmetic itself

| Principle | Prior | Strongest reason to work | Strongest failure argument | Decision |
|---|---|---|---|---|
| current 680-scheme small-format FMM catalog | MEDIUM | exact matrix-independent bilinear identities directly reduce `r` | the best normalized exponent may still miss 10x before transforms and native-ABI costs | `CHEAP_KILL_ONLY` |
| Lupanov/bitplane constant linear circuit | LOW | arbitrary fixed Boolean matrices admit shared XOR circuits | BF16 multiplication is not 16 independent linear maps; cross-bit products, exponent alignment, carries, rounding and artifact bytes remain | `NO_GO` |
| Smirnov rank-46 4x4 APA exactification | LOW | zero-overhead normalized exponent is below the required line | border rank is not a finite exact program; exactification overhead may erase the small margin | `CHEAP_KILL_ONLY` |

### Batch 3 — invert the finite-exactification failure

| Principle | Prior | Strongest reason to work | Strongest failure argument | Decision |
|---|---|---|---|---|
| finite Strassen-calculus direct-sum extraction | LOW | 2026 speedup theorems show nontrivial direct-sum degenerations can beat naive border-rank use | current theorem is asymptotic and gives no target-size exact schedule or constants | `CHEAP_KILL_ONLY` |
| cold-backed whole-layer decision-bit circuit | LOW | query-adaptive cold probes might avoid materializing all vectors | without a new cold primitive it is the closed static DAG/table family and does not generate successor state | `NO_GO` |
| cross-layer polynomial elimination | LOW | one polynomial program could share residual/projection work | arbitrary RMSNorm, softmax, SiLU and native rounding do not admit a universal low-degree closure | `NO_GO` |

The third batch contains the surviving moonshot candidate, so the principle-generation loop may proceed to its cheapest Gate rather than manufacturing more ideas in this round.

## Pinned current catalog

```text
repository: dronperminov/FastMatrixMultiplication
commit:     98ba522db92b74f1f8c561a78038ff3091356d73
schemes:    680
omega < log2(7): 52
best displayed normalized omega: 2.792481250
```

The minimum displayed value is realized by the exact rational rank-48 `4x4x4` scheme and the rank-2304 `16x16x16` scheme.

## Favorable composition envelope

For one target multiplication tensor of volume

\[
V=K\,d_{in}\,d_{out},
\]

grant the cost

\[
C_\omega(V)=V^{\omega/3}
\]

with `omega=2.792481250` and unit constant. This grant ignores shape/aspect-ratio mismatch, finite divisibility and padding, every transform, additions and coefficient scales, packing and workspace, finite-word coefficient growth, BF16 native-order repair, checkpoint bytes and causal candidate construction.

This is not an implementation estimate. It is a lower cost envelope for pure compositions of the registered entries:

1. every entry has `omega_i >= omega_min`, hence `rank_i >= V_i^(omega_min/3)`;
2. Kronecker composition multiplies both ranks and volumes, so its normalized exponent cannot fall below the minimum input exponent;
3. independent partition/direct-sum evaluation sums ranks, and for `p=omega_min/3<1`, `sum(V_i^p) >= (sum V_i)^p`;
4. padding and positive transform work cannot reduce the envelope.

A recombination that discovers genuinely new shared bilinear products and a new lower-exponent scheme is outside this rejection; it is a new algorithm, not a composition of the current entries.

## Complete registered 405B projection result

| K | ideal unit-constant arithmetic fraction |
|---:|---:|
| 32 | 19.268236953% |
| 64 | 18.366180815% |
| 128 | 17.506355074% |
| 256 | 16.686782683% |
| 512 | 15.905579153% |
| 1,024 | 15.160948218% |
| 2,048 | 14.451177707% |
| 4,096 | 13.774635605% |
| 8,192 | 13.129766299% |
| 16,384 | **12.515087006%** |

The first core line is `10%`. The best block still retains

\[
\boxed{12.515087006\%}
\]

of classical scalar multiplication before any positive executor cost.

Solving the complete model-weighted equation at `K=16,384` gives

\[
\boxed{\omega_{required}=2.7700683089152998}.
\]

The current catalog minimum is higher by `0.0224129410847000`.

```text
REJECT_CURRENT_680_EXPLICIT_FMM_CATALOG_COMPOSITIONS_AS_10X_CORE
```

## Border-rank near miss

Smirnov's `4x4x4` APA algorithm has border rank 46 and reported polynomial order 3. Its zero-overhead normalized exponent is

\[
\omega_{46}=2.7617809780285065,
\]

which would give a favorable model-weighted fraction `9.203979637%` at `K=16,384`. Thus the algebraic source has genuine headroom, but only

\[
10\%/9.203979637\%=1.086486541
\]

multiplicative exactification overhead is available.

The registered naive plan takes the seventh tensor power (`4^7=16,384`), grows order 3 to degree 21, and performs 22 independent exact evaluations for generic interpolation. Its favorable model-weighted indicator becomes

\[
22\times9.203979637\%=202.487552017\%.
\]

```text
REJECT_REGISTERED_NAIVE_RANK46_APA_INTERPOLATION_EXACTIFICATION
```

This is not a lower bound on every border-rank exactification. It rejects the specified generic independent-evaluation construction only.

## Surviving next principle

```text
FINITE_STRASSEN_CALCULUS_DIRECT_SUM_EXTRACTION_CHEAP_KILL_ONLY
```

Alman--Li's 2026 asymptotic-rank speedup framework gives a real structural reason that nontrivial direct-sum degenerations may beat naive interpolation. It does not yet provide the finite exact program required here.

EXP-108A must instantiate one target-size schedule and charge all scalar multiplications and additions, degeneration/extraction/interpolation operations, coefficient and word growth, transforms, packing, writes, workspace, native finite-word exactness and repair, plus the independent causal `A`, `N/A`, verification and successor-state path.

If no finite exact schedule crosses `10%` before positive runtime costs, no kernel or public-checkpoint experiment is authorized. The common failure premise must then be inverted and another three-principle batch generated.

## Claim boundary

This is a locally executed E0/E1 arithmetic/prior Gate. Complete 405B execution, physical 8-GiB allocation, causal future blocks, CUDA/SASS, target bandwidth and same-machine native-4B-Q4 p50/p95 remain `NOT_TESTED`.
