# Next Experiment

## Closed Gate — EXP-070

EXP-070 tested exact short-block Q4 table circuits across every frozen real dense projection. Each plan grouped identical coefficient tuples, reconstructed the complete Q4 matrix exactly, and charged dictionaries, row IDs, block offsets, favorable fused gather-adds, row scales, and non-natural permutation costs.

Authoritative coverage and integrity:

```text
3 pinned models
144 dense projections
3,024 preregistered plans
7 block widths
3 deterministic order families
checksum mismatches: 0
reconstruction mismatches: 0
hash-collision mismatches: 0
control failures: 0
```

Best joint plan per matrix, without per-axis cherry-picking:

```text
operation p50/p90:            88.4856% / 91.4423%
query-byte p50/p90:          111.0294% / 112.7907%
static representation p50/p90: 111.0294% / 112.7907%
minimum joint fraction:      105.4244%
maximum joint fraction:      113.4191%
```

Decision:

```text
REJECT_EXACT_Q4_LOCAL_PATTERN_TABLE_AS_CORE
RETAIN_BLOCK_PATTERN_ANALYZER_AUXILIARY
```

The registered short-block table family is closed. It may not be reopened by adding block widths after observation, hiding dictionaries/IDs/routing/scales, reporting arithmetic without bytes, or using learned/approximate pattern merging while claiming exactness.

Authority:

```text
results/exp_070/summary.json
results/exp_070/raw/plan_rows.jsonl
results/exp_070/raw/selected_rows.jsonl
results/exp_070/checksums.sha256
workflow 30930542616
artifact 8901017649
artifact ZIP SHA-256 0e3e60f959af852759b9aac8dd6af1a28524cdcbb6c736cd8e32ad00d6c29987
```

## EXP-071 — Universal Exact Dense Runtime Lower-Bound Applicability Audit

### Research-class change

Do not implement another weight decomposition or cache variant next. First determine whether the fixed objective is compatible with known unconditional online matrix-vector data-structure lower bounds under a clearly stated conventional execution model.

The audit starts from primary results including:

- Clifford, Grønlund, and Larsen, *New Unconditional Hardness Results for Dynamic and Online Problems*, FOCS 2015, DOI `10.1109/FOCS.2015.71`;
- Chakraborty, Kamma, and Larsen, *Tight Cell Probe Bounds for Succinct Boolean Matrix-Vector Multiplication*, STOC 2018, arXiv `1711.04467`.

These papers must not be cited as a 405B impossibility proof until every hypothesis and reduction below is checked.

### Formal execution model

Define the candidate runtime explicitly:

```text
immutable arbitrary dense matrix/model
unbounded offline preprocessing time
read-only cold representation
at most 8 GiB total hot/side information
online causal query vector unavailable during preprocessing
exact output required for every supported matrix and query
standard word-RAM/cell-probe memory accesses
no future queries, learned target modification, or external prover hardware
```

Record word size, cell size, randomization/error allowance, side-information size, cold representation size, preprocessing dependence, and whether computation between probes is free.

### Required applicability checks

1. Verify theorem statements directly from the primary papers.
2. Build a machine-readable hypothesis matrix for:
   - Boolean and `F2` arithmetic versus row-scaled Q4/float execution;
   - square versus rectangular projections;
   - static versus dynamic preprocessing;
   - exact versus bounded-error queries;
   - per-matrix versus model-wide side information;
   - theorem word/cell-size and space ranges.
3. Construct an exact small-domain reduction control:
   - embed arbitrary `0/1` matrices and `0/1` vectors in a dense float projection;
   - prove that exact integer output permits recovery of the corresponding `F2` product;
   - exhaustively verify the reduction on the registered small domains.
4. Determine whether square padding and a direct-sum argument across independent layer matrices are actually justified. Do not assume direct-sum composition.
5. Apply only certified formulas to the registered Llama-3.1-405B tensor plan and the 8 GiB hot-state budget.
6. Separate theorem-backed lower bounds from heuristic projections and hardware estimates.

### Required outputs

```text
source theorem and exact statement
hypothesis-by-hypothesis applicability table
verified reduction controls
per-matrix parameter rows
model-wide side-information allocation audit
certified probe lower bound, if derivable
fraction of dense Q4 traffic/operations implied by that bound
remaining loopholes and execution models not covered
```

### Promotion outcomes

A strong closure requires a rigorous reduction and model-wide/direct-sum bound whose favorable lower limit already exceeds the fixed `1.185185%` whole-execution budget. Only then may the repository state that the registered conventional exact online-runtime model is ruled out.

```text
CERTIFY_CONVENTIONAL_EXACT_ONLINE_DENSE_RUNTIME_LOWER_BOUND
```

If the source theorems do not cover Q4/float semantics, rectangular/model-wide composition, the 8 GiB side-information regime, or the required direct sum, record:

```text
INSUFFICIENT_LOWER_BOUND_DO_NOT_CLAIM_IMPOSSIBILITY
```

An insufficient theorem is not evidence that the objective is feasible or impossible.

### Stop rule

Before this audit completes, prohibit:

```text
another classical exact matrix decomposition variant
another pattern width/order sweep
claims that all software executors are impossible
numerical 405B lower-bound projections without checked theorem hypotheses
CUDA or target-hardware implementation based only on an asymptotic theorem
```

### Claim boundary

Phase A/B theorem and reduction audit, ceiling E1. A paper theorem is not a measurement. Actual 405B execution, 8 GiB runtime behavior, CUDA, PCIe, SSD, TTFT, tokens/second, and physical latency remain **NOT TESTED**.
