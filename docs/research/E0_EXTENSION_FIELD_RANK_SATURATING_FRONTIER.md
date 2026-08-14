# E0 Extension-Field Rank-Saturating Frontier

## Verdict

```text
REJECT_RANK_PARAMETER_AS_PROBE_COUNT
REJECT_FULL_AMBIENT_RHO_ONE_DIRECT_MATERIALIZATION
REJECT_QUERY_SPECIALIZED_IDENTITY_DIRECT_LAYOUT
KEEP_QUERY_DEPENDENT_NONLINEAR_COSET_DECODER OPEN
REQUIRE SMALL 32-QUERY PHYSICAL UNION, NOT ONLY SHORT SINGLE-QUERY LEADERS
NO SURVIVING CANDIDATE
```

This is a cheap novelty and resource Gate.  No model, checkpoint, backend,
server, download, or hardware action occurred.

## Elementary explanation

An `m x k` binary matrix can be repacked so that each column becomes one
letter with `m` binary digits.  A rank-one question then looks like

```text
same big letter alpha * a different on/off switch u_j in every column.
```

This makes its **rank** equal to one.  It does not mean that only one stored
piece is used.  If half the switches `u_j` are on, half the stored letters are
still read.  Thirty-two independent questions turn almost every switch on at
least once.

The finite-geometry construction has a second trap.  Its one-dimensional
rank certificate can use a dense binary combination of hundreds of millions
of basis vectors.  Storing a reusable `m`-bit answer for every such basis
vector expands the registered one-bit checkpoint to about `748.55 TiB`.
Compressing those redundant answers back to their real information content
returns to the original checkpoint and removes the promised local access.

The useful remainder is narrower: choose a different short representative
from each query's code coset, and choose the representatives for all 32
queries **jointly** so their physical-address union is small.  No such large
construction is currently known here.

## 1. Exact extension-field repacking

Choose a basis `e_1,...,e_m` of `F_(2^m)` over `F_2` and its trace-dual basis
`e_1*,...,e_m*`.  Pack a binary row direction and checkpoint column as

```text
alpha  = sum_i r_i e_i
beta_j = sum_i W_ij e_i*.
```

Duality gives the exact identity

```text
r^T W[:,j] = Trace(alpha * beta_j),
```

so for a binary column direction `u`,

```text
<r u^T, W> = Trace(alpha * sum_j u_j beta_j).             (1)
```

Thus the mask becomes `(u_j alpha)_j` in `F_(2^m)^k` and has rank weight one.
Equation (1) is an exact GF(2) identity.  It is not yet a native Q4/BF16/FP32
accumulation rule.

## 2. Why published rank saturation is not a probe bound

Bonini, Borello, and Byrne define an `n`-dimensional binary q-system `U` with
basis `g_1,...,g_n`.  Their characterization says every ambient vector has

```text
v = sum_j lambda_j g_j,
rank_weight(lambda) <= rho.                              (2)
```

At `rho=1`, equation (2) means `lambda_j=gamma*mu_j` for one field element
`gamma` and binary `mu_j`.  The support of `mu` can still be all `n`
coordinates.  Therefore `rho` is the dimension of the coefficient values'
binary span, not the number of nonzero coefficients or memory cells.

For full ambient coverage, the paper proves the exact minimum

```text
s_(2^m/2)(k,1) = m(k-1)+1.                               (3)
```

If one checkpoint field summary is materialized per q-system basis vector,
the registered shape population costs

```text
one-bit source                         405,849,243,648 bits
direct field summaries               6,584,324,197,576,704 bits
summary/source ratio                      16,223.57143848x
direct summaries                              748.55099656 TiB.
```

One `16,384 x 16,384` matrix alone uses `268,419,073` q-system basis vectors
and `0.4999694843 TiB` of direct summaries for one source bit plane.

These summaries are highly redundant functions of the raw matrix, so this is
not an information lower bound.  It is a rejection of the direct
materialization that would turn coefficient rank into local reads.

Primary source: Matteo Bonini, Martino Borello, and Eimear Byrne,
*Saturating systems and the rank-metric covering radius*:
<https://arxiv.org/abs/2206.14740>.

## 3. The rank-one-query-specific system still scans the source

Full ambient saturation is stronger than VORTEX needs.  Every required
rank-one vector is already a scalar multiple of a binary vector, so the
query-specific system

```text
U = F_2^k
```

uses only the `k` coordinate basis vectors.  Storing an `m`-bit field summary
for each coordinate is exactly the original `m*k` source bits.

For `K` independently uniform binary right directions, a coordinate is used
at least once with probability

```text
1 - 2^-K.
```

At `K=32`, after granting the entire 8 GiB as arbitrary hot source bits and
perfect bit packing, the direct model-wide one-bit layout still has

```text
expected active cold bytes ceiling          42,141,220,855
complete block allowance                     13,780,500,000
allowance ratio                                  3.058032789x
zero-compute time at 32 GB/s                    41.153536 ms/token.
```

This is already above 20 ms/token before the other checkpoint bit planes,
native arithmetic, metadata, addresses, state, KV, or verification.  A
worst-case single direction with every switch set reads the complete direct
layout, so the expectation is not the only failure mode.

## 4. Directly invented nonlinear-coset screen

The extension-field failure does not reject nonlinear syndrome decoding.  A
throwaway exact search therefore tested a materially different rule:

```text
for each query q, choose the minimum-weight a(q) satisfying G a(q)=q.
```

The prototype is preserved off the main branch at commit `2f68300` on
`prototype/rank-one-dictionary-small-search`.  The `2 x 2` result at five
atoms is exhaustive over all 2,688 full-rank dictionaries; larger witnesses
are explicit but their optimality is not claimed.

| shape | atoms | redundancy | worst single-query leader | expected 32-query canonical union |
|---|---:|---:|---:|---:|
| `2 x 2` | 5 | 25.00% | 2 | 4.9769 / 5 |
| `2 x 3` | 7 | 16.67% | 3 | 6.9987 / 7 |
| `3 x 3` | 11 | 22.22% | 4 | 10.9344 / 11 |

The experiment establishes two scoped facts.

1. Nonlinear coset leaders genuinely beat a fixed raw-coordinate
   representation on a single tiny query, so F-076 cannot be generalized to
   nonlinear decoders by assertion.
2. Independently minimizing each query does not produce batch locality in
   these controls.  Almost every atom appears somewhere in a 32-query union.

This is not an asymptotic lower bound.  A joint batch decoder could choose
slightly longer individual representatives with a smaller common support.
That creates the next exact interface:

```text
given Q=[q_1,...,q_K], find one small T such that
q_i is in span{g_j : j in T} for every i.                 (4)
```

Equation (4), finite physical addresses, and native summaries must all be
constructed before this becomes a Core Candidate.

## 5. Nearby code results do not fill the gap

The 2025 stream-decodable-code construction can compute a private linear
function with near-linear encoded length, but its decoder processes the
encoded word as a stream and succeeds with high probability.  It is not a
zero-error sparse-probe layout.  The authors also note that their known local
code for arbitrary linear functions is a Hadamard extension with exponential
length.  This matches the literal-table boundary already rejected by F-075,
not equation (4).

Primary source: Meghal Gupta, Venkatesan Guruswami, and Mihir Singhal,
*Tight Bounds for Stream Decodable Error-Correcting Codes*:
<https://arxiv.org/abs/2407.06446>.

## 6. Promotion obligations

A future batch coset code must provide all of the following at once:

1. a near-source-size implicit dictionary or nonlinear encoding;
2. a proof for every legal causal batch, or a fail-closed causal restriction;
3. a joint decoder producing a small union of 64-bit/page addresses without a
   metadata scan;
4. exact native Q4/BF16/FP32 answer and accumulation semantics;
5. complete construction, persistent-state, selector, KV/state, correction,
   fallback, and physical shared-link costs;
6. a favorable 405B equation below the registered block allowance.

No current object meets these obligations.  The extension-field route is
therefore rejected as a constructor, while query-dependent joint nonlinear
coset decoding remains an unconstructed research interface.

## 7. Reproduction

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_extension_field_rank_saturating_frontier.py `
  --output-dir results\e0_extension_field_rank_saturating_frontier

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_extension_field_rank_saturating_frontier -v
```

Authority:

```text
results/e0_extension_field_rank_saturating_frontier/summary.json
results/e0_extension_field_rank_saturating_frontier/checksums.sha256
```
