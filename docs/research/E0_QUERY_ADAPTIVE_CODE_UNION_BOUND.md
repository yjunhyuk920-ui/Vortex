# E0 Query-Adaptive Exact Code-Union Bound

Status: `REJECT_TRACE_BUILT_QUERY_ADAPTIVE_LINEAR_CODE_UNION_AS_CORE`

This note tests whether nonlinear routing over many small exact Bilinear Cross
Residual ledgers can escape EXP-084A's single-span failure. It gives the
candidate its strongest useful interpretation: a perfect causal router and
leaf lookup are free, only the selected leaf is scanned on a query, and every
leaf may be chosen with future-aware optimality. Every answer-bearing basis
direction still has to be constructed and retained.

The result closes only unions of materialized, trace-built linear leaves. It
does not prove a lower bound for an implicit nonlinear data structure derived
directly from the checkpoint.

## Candidate mechanism

For a Bilinear Cross Residual query `q = r tensor u`, leaf `l` stores a linear
space

```text
U_l = span(q_l,1, ..., q_l,d_l)
```

and the exact cached scalar answers to those basis queries. A perfect router
chooses one leaf. The fast path is legal only when an exact rational witness
proves `q in U_l`; otherwise the unchanged dense operation runs or the
executor aborts.

The union `U_1 union ... union U_L` is generally not a linear space, so this
is genuinely more expressive than scanning one ledger. The router changes
which small code is read. It does not create a new cached answer direction.

## Union-information theorem

Let `Q={q_1,...,q_N}` be a globally linearly independent finite query
population and let leaf `l` have dimension `d_l`. Then

```text
number of exact fast-path hits <= sum_l d_l = A.
```

Proof: any subset of `Q` lying in `U_l` remains independent and therefore has
at most `d_l` members. Summing that bound over all leaves proves the claim;
duplicate membership can only make the sum looser.

Equivalently, if leaf bases share storage, the global span of all independent
answer-bearing generators must still have rank at least the number of
independent hits. Sharing can remove redundant builds, but cannot encode a
new independent answer for free.

A second containment result handles the immediate EXP-084A rescue proposal.
If every leaf generator is drawn from one calibration span `V`, every
`U_l` is a subspace of `V`; therefore their union contains no query outside
`V`. Partitioning or routing a failed calibration span cannot add coverage.

These are finite linear-algebra statements, not claims that all causal
Transformer queries are independent.

## Registered 405B resource equation

Use the existing Causal Bilinear Span Ledger accounting:

```text
service population N                  20,000,000 tokens
p50 target alpha                      8/675 = 1.185185185%
build per independent cached answer   2 dense equivalents
largest selected leaf                 23 directions
```

For selected-leaf dimension `b`, resource `j` in `{traffic, operations}`, and
union-wide independent direction count `A`:

```text
q_j(b)       = charged selected-leaf query fraction
build(A)     = 2A/N
A_max(b)     = floor((N/2) * min_j(alpha - q_j(b)))
c_required   = max_j(1 + q_j(b) + 2A/N - alpha)
```

`c_required` is the exact-hit coverage required when every miss executes one
full dense fallback. Selector computation, indexing, address/request latency,
pair extraction, local result production, native numerical repair, and
fallback overlap are all granted free. Only one leaf is hot, but construction
and persistent representation of all leaves are charged.

| leaf `b` | `A_max` | independent coverage over 20M | required coverage with fallback | explicit union state | independent-stream target multiple |
|---:|---:|---:|---:|---:|---:|
| 1 | 87,958 | 0.439790% | 99.999992826% | 6.259769 TiB | 85.0039x |
| 2 | 84,082 | 0.420410% | 99.999995412% | 5.984211 TiB | 85.0203x |
| 4 | 76,328 | 0.381640% | 99.999991083% | 5.432879 TiB | 85.0530x |
| 8 | 60,817 | 0.304085% | 99.999994415% | 4.329702 TiB | 85.1184x |
| 16 | 29,778 | 0.148890% | 99.999999041% | 2.120943 TiB | 85.2494x |
| 23 | 2,600 | 0.013000% | 99.999998047% | 0.185625 TiB | 85.3640x |

Dimension 1 is the most favorable direction-count ceiling. Its selected-leaf
traffic is only `0.305598011%`, yet its 87,958 cached directions cover at most
`0.43979%` of a 20M-member independent population. The fully charged fraction
on that population is `1.0074538801`, or `85.0039x` the complete p50 target.

Even an online grant that turns each miss's unavoidable dense answer directly
into a cache entry and removes the extra registered build passes cannot help a
never-repeating independent stream: each new query still performs one dense
execution, giving at least `1 + q_j(b)`. At `b=1` this is `84.6328x` target.

The theorem therefore rules out the union as a universal primary mechanism
without a proved low-union-dimension causal query population. It remains a
valid auxiliary cache for repeated or tightly clustered queries.

## Frozen EXP-084A zero-forward diagnostic

The calculator replays only the already frozen factor arrays. It performs no
Transformer forward and does not create a new experiment.

All 24 build rows have rank 24 modulo each of `65521`, `65519`, and `65497`.
Each of the five stored evaluation rows raises the complete 24-row build span
to rank 25 under every prime. Thus every evaluation row is rationally outside
the full calibration span, not merely outside the first-23 ledger used by the
registered run:

```text
complete build span rank             24 / 24 / 24
evaluation rank after insertion      25 / 25 / 25 for every row
hits by any partition of that span   0 / 5
certified misses                     5 / 5
model forward calls                  0
```

This proves that no partition, routing rule, or union of leaves generated only
from those 24 build rows can rescue any stored evaluation row. It does not
measure the union complexity of the full causal population or prove that
future queries are globally independent.

## Decision and claim boundary

```text
REJECT_TRACE_BUILT_QUERY_ADAPTIVE_LINEAR_CODE_UNION_AS_CORE
RETAIN_REPEATED_QUERY_CODE_UNIONS_AS_AUXILIARY_ONLY
DO_NOT_CLAIM_A_GENERAL_NONLINEAR_OR_CELL_PROBE_LOWER_BOUND
REQUIRE_AN_IMPLICIT_CHECKPOINT_DERIVED_NONLINEAR_SOURCE_NEXT
KEEP_NO_SURVIVING_CANDIDATE
```

The next admissible proposal must provide a concrete automatic checkpoint
constructor and exact query equation whose useful answer information is not a
materialized collection of trace basis directions/scalar answers. It must
charge state, build, selector, probes, traffic, operations, verification,
misses, and fallback before any new model execution.

Not established here:

- a universal lower bound for arbitrary nonlinear or implicit data structures;
- independence of all actual causal Transformer queries;
- a paid general `(r,u)` extractor or native BF16/Q4 reconstruction path;
- actual dense-operation replacement, model quality, physical latency, VRAM,
  target-server behavior, or 122B/405B execution.

## Reproduction

```powershell
$env:PYTHONPATH = ".;.deps"
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_query_adaptive_code_union.py `
  --exp084-dir results\exp_084a `
  --output-dir results\e0_query_adaptive_code_union
.deps\exp076-venv\Scripts\python.exe -m pytest -q `
  tests\test_query_adaptive_code_union.py
.deps\exp076-venv\Scripts\python.exe -m pytest -q
.deps\exp076-venv\Scripts\python.exe scripts\run_validation.py
```

Observed validation: `9` focused tests, all `508` repository tests, and the
standard validation runner passed. A second output directory reproduced the
summary byte-for-byte. Canonical summary SHA-256:
`761b857dda9521a41a3b5b93bf32c6429379a76d4d72192704ee6f03aacb656e`.

Authority:

```text
vortex_runtime/query_adaptive_code_union.py
scripts/derive_query_adaptive_code_union.py
tests/test_query_adaptive_code_union.py
results/e0_query_adaptive_code_union/summary.json
results/e0_query_adaptive_code_union/checksums.sha256
```
