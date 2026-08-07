# E0 Query-Adaptive Cold-Backed Feasibility Equation

## Scope and evidence boundary

This is the equation requested after the static synthetic-circuit closure. It
does not assume that a causal information source exists, and it does not assign
an experiment number. Its purpose is to state the most favorable complete
resource frontier that any such source must cross before implementation.

All claims in this note are `DERIVED` at E0. No checkpoint, Transformer, GPU,
Ubuntu host, storage device, CUDA kernel, or latency baseline was run.

## Frozen normalization

The authoritative logical denominator is the EXP-080A non-embedding dense
population:

```text
non-embedding coefficient uses D                 403,747,897,344
packed-Q4 bytes B_D = D/2                        201,873,948,672
packed-Q4 GiB                                    188.009765625
non-embedding matrix instances                   883
layers                                           126
complete registered checkpoint Q4 GiB            188.988281250
p50 proxy alpha_50 = 1.2 * 4 / 405               1.185185185%
p95 proxy alpha_95 = 1.5 * 4 / 405               1.481481481%
p50 non-embedding byte allowance                 2.228263889 GiB/token
p95 non-embedding byte allowance                 2.785329861 GiB/token
```

The dimensionless fractions are authoritative for this E0 Gate. The byte
figures are their exact packed-Q4 translations, not measured traffic. The
complete checkpoint size is used only for cold-capacity and index calculations;
mixing it into the non-embedding operation denominator would change the Gate.
The older `2.235174 GiB` proxy elsewhere in the repository starts from a
literal four-billion-parameter Q4 baseline; the `2.228264 GiB` value here
starts from the frozen non-embedding population. They encode the same
dimensionless `1.185185185%` Gate under different byte denominators and must
not be silently interchanged.

## One equation for operations and traffic

Let `j` be either logical operations `C` or logical bytes `B`, normalized to
one complete dense token evaluation. For a causal query, define:

- `rho`: exact fast-path coverage or hit probability on the declared held-out
  population;
- `g_j`: common selection, probing, metadata, verification, and other work paid
  on every query;
- `h_j`: successful fast-path execution work paid only on a hit;
- `m_j`: wasted detection or partial-execution work paid only on a miss;
- `kappa_j`: one-time compile/build cost in dense-token equivalents;
- `N`: number of logical service tokens over which that build is amortized;
- `1`: the unchanged dense operation or traffic paid after a miss.

The fully charged mean is therefore

```text
R_j = g_j + kappa_j/N + rho*h_j + (1-rho)*(m_j + 1).       (1)
```

This form is deliberately branch-aware. A verifier used only on hits belongs
in `h_j`; a selector or verifier run before the hit decision belongs in `g_j`.
Miss discovery after partial work belongs in `m_j`. Moving a term between these
positions without matching the actual control flow is forbidden.

For `h_j < 1 + m_j`, solving (1) gives the exact favorable coverage frontier:

```text
rho_j >= (g_j + kappa_j/N + m_j + 1 - alpha_j)
         / (m_j + 1 - h_j).                              (2)
```

The candidate must satisfy both resources:

```text
rho >= max(rho_C, rho_B),
R_C <= alpha, and R_B <= alpha.                           (3)
```

If the right-hand side exceeds one, perfect coverage cannot rescue the path.
If `h_j >= 1 + m_j`, a hit is no cheaper than miss plus fallback and increasing
coverage cannot create a speedup.

## Fallback is already nearly forbidden

The following table grants zero hit work, zero miss work, and zero build cost.
The listed fast-path traffic is treated as common work paid on every query, so
the remaining allowance is given entirely to dense fallback.

| Common fast-path traffic | Minimum exact coverage at `alpha_50` |
|---:|---:|
| `0.000000000%` | `98.814814815%` |
| `0.100000000%` | `98.914814815%` |
| known six-check verifier `0.266838924%` | `99.081653739%` |
| `0.500000000%` | `99.314814815%` |
| `0.900000000%` | `99.714814815%` |
| EXP-081A pre-fallback `0.926657941%` | `99.741472756%` |
| `1.000000000%` | `99.814814815%` |
| `1.100000000%` | `99.914814815%` |

The EXP-081A row independently reproduces its preregistered
`99.7414727557%` coverage requirement. Positive miss work or build
amortization makes every row stricter.

At the looser `alpha_95`, even a literally zero-cost fast path still needs
`98.518518519%` coverage under the same mean-resource conservation equation.
This is distinct from a latency quantile condition.

## Latency quantiles are separate, additional Gates

The fixed mission requires measured same-machine latency:

```text
Q_0.50(T_candidate) <= 1.2 * Q_0.50(T_native_4B_Q4)
Q_0.95(T_candidate) <= 1.5 * Q_0.95(T_native_4B_Q4).       (4)
```

If a dense fallback is slower than the respective threshold, a necessary
distributional condition is at least 50% non-fallback tokens for p50 and at
least 95% for p95. The mean logical conservation Gate above is normally much
stricter: it allows only `1.185185185%` fallback even when the successful path
is free. Passing (1)-(3) does not prove (4); it only avoids a conservation-level
rejection before hardware work.

For `k` cold requests, maximum useful request parallelism `d`, per-request
latency `tau`, effective compute rate `Gamma`, and storage/H2D bandwidths, a
favorable physical lower bound is

```text
T_token >= max(C_token/Gamma,
               B_storage/beta_storage,
               B_H2D/beta_H2D,
               ceil(k/d)*tau,
               T_serial_dependencies).                   (5)
```

EXP-073 measured none of the required storage, transfer, native-4B, or kernel
rates. Its favorable PCIe Gen2 x16 signaling ceiling is about
`7.450580597 GiB/s`. If the entire p50 logical byte allowance crossed that link,
serialization alone would take at least `0.299072517 s/token`, before protocol,
storage, compute, and dependency overhead. This is not a latency rejection
because the native 4B baseline remains unmeasured.

## Cold-page traffic and granularity

For equal payload pages of `L` bytes, per-page request/index traffic `e` bytes,
and all other traffic `g_B`, the maximum successful-path page count is

```text
k <= floor((alpha_50 - g_B) * B_D / (L + e)).             (6)
```

With `e=0`, the registered p50 frontier is:

| Payload page | Max pages, all other costs free | Max pages after known verifier | Full-checkpoint pages | 32-byte resident metadata |
|---:|---:|---:|---:|---:|
| `4 KiB` | `584,126` | `452,612` | `49,542,144` | `1.476471 GiB` |
| `64 KiB` | `36,507` | `28,288` | `3,096,384` | `0.092279 GiB` |
| `1 MiB` | `2,281` | `1,768` | `193,524` | `0.005767 GiB` |
| `2 MiB` | `1,140` | `884` | `96,762` | `0.002884 GiB` |

These are byte ceilings, not recommended page counts. Small pages pay more
metadata and request latency; large pages waste payload when a query needs only
a small region.

Conditionally, if an executor must issue at least one equal cold call for each
of the 883 non-embedding matrix instances, the average payload ceiling is only
`2,709,603 bytes` (`2.584080 MiB`) with every other cost free, or
`2,099,549 bytes` (`2.002286 MiB`) after the known verifier. A mechanism that
certifiably skips a whole matrix need not make that call, so this is a
conditional frontier rather than a universal lower bound.

Any selector scan is charged as well. An index retained hot consumes VRAM; an
index scanned from host or cold storage contributes traffic and request latency.
The page identifiers cannot be supplied by an uncharged oracle.

## Hot, intermediate, and cold state

The tight branch-aware peak equation is

```text
M_common = M_KV + M_runtime + M_index + M_verifier
           + M_resident_cache

M_peak = M_common
         + max(M_hit_pages + M_hit_intermediates,
               M_miss_pages + M_miss_intermediates + M_dense_fallback)
         <= 8 GiB.                                        (7)
```

Mutually exclusive buffers may be reused, which is more favorable than adding
both branches. Double buffers, concurrent requests, decompression state,
selector scratch, output accumulators, allocator reserve, and KV cache must
still appear in the branch where they are live.

Cold capacity is a separate equation:

```text
S_checkpoint + S_code_pages + S_manifest + S_cold_index
    <= available cold storage.                            (8)
```

The unmodified Q4 checkpoint alone is `188.988281250 GiB`. The EXP-073 snapshot
had `97.6183 GiB` root free, a `91.36998125 GiB` shortfall before sidecars. A
later storage upgrade can remove this capacity blocker but cannot improve the
per-token equations.

## Compile amortization

If automatic preprocessing costs `kappa` full dense-token equivalents and the
runtime retains only `r` fraction of headroom, then

```text
N >= ceil(kappa / r).                                     (9)
```

Examples:

| Build cost | Runtime headroom | Minimum service life |
|---:|---:|---:|
| one dense scan | `1.0%` | `100` tokens |
| one dense scan | `0.1%` | `1,000` tokens |
| ten dense scans | `0.1%` | `10,000` tokens |
| one hundred dense scans | `0.01%` | `1,000,000` tokens |

Build operations, build I/O, persistent bytes, and service-life assumptions
must be reported separately. Preprocessing may depend on the unchanged public
checkpoint but not on future evaluation activations or target outputs.

## Verification cannot be the missing source

A successful result must be exact under the selected branch or satisfy a
preregistered output-quality contract. A failed check must abort or enter the
charged dense branch. An exact verifier that recomputes `W*x` costs another
dense equivalent and therefore fails (1). The retained six-check trace verifier
is much cheaper (`0.058120260%` operations, `0.266838924%` traffic, and
`0.427185 GiB` sidecar), but it only checks a supplied trace; it does not
produce the trace.

Probabilistic verification additionally needs a declared per-check soundness
error and a union bound over matrices, tokens, and the service lifetime. Its
sidecar and challenges remain in (1) and (7).

## What the equation says about the missing invention

With all overhead and fallback set to zero, the cold read or computation must
already be at least

```text
1 / alpha_50 = 84.375x
```

more informative per charged unit than a raw dense sweep. Reserving only the
known verifier traffic raises the required payload amplification to

```text
1 / (alpha_50 - 0.266838924%) = 108.891389x.
```

A raw Q4 page merely supplies the coefficients it contains. Under a page-only
interface with no checkpoint-derived information about an unread coefficient,
exact arbitrary `W*x` is impossible: two matrices can agree on every observed
page and differ at one unread coefficient; choosing an activation with a
nonzero value at that column changes the exact output while the executor sees
identical information. This is an elementary indistinguishability argument,
not a universal data-structure lower bound.

Preprocessed coded pages, a hot checkpoint-derived index, or a causal
certificate can escape that narrow argument because they carry information
about omitted weights. They must then pay their build, storage, selector,
probe, decode, verification, and miss costs in (1), (7), and (8). Static
synthetic aggregates alone remain inside the already closed linear-DAG class.

Therefore the next search is sharply constrained. A surviving source must:

1. derive page/operation identifiers from the current committed activation and
   unchanged checkpoint without a dense-equivalent discovery scan;
2. reconstruct or certify omitted contributions, rather than merely omit raw
   Q4 pages;
3. demonstrate at least `98.814815%` exact held-out coverage in the physically
   impossible zero-cost limit, and at least `99.081654%` if it uses the known
   verifier before any other cost;
4. provide more than `84.375x` useful information amplification, or more than
   `108.891x` after that verifier;
5. close both operation and byte equations, the 8-GiB branch peak, cold
   capacity, build amortization, and eventually the measured quantile Gate.

## Decision

```text
DERIVE_QUERY_ADAPTIVE_COLD_BACKED_FEASIBILITY_FRONTIER
REJECT_RAW_Q4_PAGE_SELECTION_WITHOUT_OMITTED_CONTRIBUTION_SOURCE
KEEP_CODED_CAUSAL_COLD_SOURCE_OPEN_FOR_INFORMATION_SOURCE_SEARCH
```

The equation is a useful E0 result: it converts "read only a few pages" into a
falsifiable `>=98.8%-99.1%` coverage and `>=84.4x-108.9x` information-amplification
requirement. It is not a Core Candidate and supplies no positive execution
mechanism. No EXP-083, checkpoint download, Ubuntu command, runtime, kernel,
hardware test, E1, or E2 is authorized yet.

## Reproduction

The calculator is intentionally separate from runtime admission:

```powershell
python scripts/derive_query_adaptive_cold_equation.py
python -m pytest tests/test_query_adaptive_cold_equation.py -q
```

It reproduces the registered shape, coverage frontier, page table, metadata,
compile examples, and favorable PCIe floor. The tests also reproduce the
independent EXP-081A fallback threshold.
