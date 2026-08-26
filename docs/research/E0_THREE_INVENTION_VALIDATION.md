# E0 Validation — Three Invented Executor Principles

Date: 2026-08-26 Asia/Seoul  
Base evidence ref: `d4b44e6f2ccfb35231c852e66de4434755ff17c4`  
Deterministic core: `9024a1ccb8da797dbf2567dcfd5c059507debe7733b41a090fdbfe9cb4e11e27`

## Question

Validate the three proposed mechanisms before assigning a numbered experiment:

1. **CSPE — Causal Syndrome-Peeling Executor**
2. **STC — Spacetime-Transpose Contraction**
3. **NRM — Native Reduction-Tree Macro**

The fixed p50 target fraction is:

```text
8 / 675 = 1.185185185185%
```

Therefore a candidate must eliminate at least:

```text
98.814814814815%
```

before selector, metadata, verification, fallback, KV, workspace, and physical
traffic costs.

This audit runs no new checkpoint and no hardware. It combines finite controls
with already committed real-checkpoint evidence. It may reject the declared
mechanisms; it does not establish a universal impossibility theorem for every
future nonlinear data structure.

---

## 1. CSPE

### Identity

For a fixed code dictionary `P`, exact decomposition

```text
x = P a + e
Z = W P
W x = Z a + W e
```

is mathematically valid. If `a` and `e` are sparse and native reduction
semantics can be repaired cheaply, the mechanism could calculate a complete
projection output rather than only a token decision.

### Universal finite counting Gate

The target hidden width is `n=16,384`; the zero-overhead query budget is only:

```text
floor(n * 8/675) = 194
```

dictionary/residual terms.

Restrict the activation domain to binary vectors. For fixed supports of sizes
`s` and `r`, the corresponding real subspace has dimension at most `s+r` and
therefore intersects the binary hypercube in at most `2^(s+r)` points. Summing
over dictionary and residual supports gives the favorable upper bound used by
the reference script.

Even if all 8 GiB are given to one square projection and every stored output
atom costs only **one bit per output coordinate**, at most `2^25` atoms fit.
The best support split covers at most:

```text
3849.499747
of 16,384 binary information bits
= 23.495482%
```

To cover every binary activation with at most 194 queried terms would require
at least:

```text
2^89.610826
```

fixed dictionary atoms under the same favorable bound. Precomputing `W P` for
that population is outside any finite VORTEX resource envelope.

This is a universal counterexample to the stated fixed-dictionary core. It
does not reject a checkpoint-specific implicit nonlinear code that is not a
finite `P` dictionary.

### Restricted real-checkpoint evidence

The closest existing favorable exact state-axis realization used future target
activations and coordinate-wise parent selection. At `K=128` it reconstructed
exactly but measured:

```text
input residual nonzero p50     80.063430%
down residual nonzero p50      89.354700%
whole-model operation p50      20.239988%
target                         1.185185%
miss factor                    17.077490x
reconstruction mismatches      0
```

Separate causal exact temporal-span evidence had p50/p90 mandatory operation
fractions of 100%/100% and zero verified exact replay hits.

### Decision

```text
REJECT_CURRENT_CSPE_AS_UNIVERSAL_CORE
KEEP_ONLY_CONCRETE_DISTRIBUTION_SPECIFIC_NONLINEAR_CODE_GATE_OPEN
```

A reopening must define the actual implicit code, decoder, persistent state,
`W`-image construction, finite-word semantics, and complete byte/operation
equation before another model run.

---

## 2. STC

### Relation versus algorithm

Writing the `K`-step autoregressive process as one relation is valid. However,
a relation is not yet a subdense executor. The current proposal gives no
finite representation that computes the relation below dense arithmetic.

### Causal adversarial control

For an arbitrary transition `H`, computing `H^K(s0)` through black-box state
queries has a pointer-chasing barrier. After only `K-1` path queries, two exact
transition completions can share the entire observed transcript and disagree
on the `K`th state. The deterministic control verifies this construction for:

```text
K = 2, 4, 8, 16, 85
```

Therefore a universal STC implementation must do at least one of:

```text
follow the causal path sequentially;
materialize a branch/jump representation for unseen states;
exploit an additional checkpoint-specific structure.
```

The third item is the missing new information source.

The committed real-model fixed-point evidence also observed both the
one-position-per-round triangular barrier and transcript indistinguishability.
After four favorable passes, the p50 matching prefix was only 4.5 and the
maximum was 6.

### Perfect-future arithmetic Gate

Even when all future activations are granted for free, the authoritative
constructive Hyperblock/Strassen calculation retained:

```text
best constructive arithmetic   36.111580%
target                          1.185185%
miss factor                     30.469145x
joint passing block lengths     0
```

Thus transposing the time/layer execution order can amortize weight traffic,
but the current mechanism does not remove enough arithmetic.

### Decision

```text
REJECT_CURRENT_STC_AS_REORDERING_NOT_SUBDENSE_EXECUTOR
RETAIN_FACTOR_GRAPH_REFERENCE_ONLY
```

A reopening requires a concrete finite-word contraction whose node count,
bytes, workspace, causal construction, exact terminal KV, and measured
arithmetic all close the target equation. Merely changing variable-elimination
order is not sufficient.

---

## 3. NRM

### Exact-hit semantics

A native reduction-tree macro is exact when all of the following are fixed:

```text
same projection row/subtree;
same activation words at the same coordinates;
same multiply and native partial-reduction ABI.
```

On a hit, replacing the subtree with its previously stored native partial word
is valid.

### Coverage lemma

If a fixed-position subtree pattern matches a previous registered pattern,
then every covered leaf value must previously have occurred at that same
coordinate. Therefore:

```text
macro-covered leaves ⊆ same-coordinate scalar-history hits
```

The focused synthetic control verifies the subset property.

On the committed real Qwen3.5 activation trace, the unlimited
same-coordinate history hit fraction was only:

```text
2.985491%
```

Even with macro storage and lookup granted free, this leaves at least:

```text
97.014509%
```

of work, or:

```text
81.855992x
```

the target.

Granting duplicate products, perfect opposite-pair cancellation, and other
local exact reuse together still eliminated at most:

```text
8.677455%
```

leaving a `77.053397x`
target miss.

### Favorable storage screen

For output width 16,384, one BF16 partial-output vector costs 32 KiB. Giving
the complete 8 GiB to one matrix stores at most:

```text
262,144
```

macros, an average of only:

```text
16.0
```

size-one values per input coordinate, versus 65,536 possible BF16 words.
Larger subtree-pattern spaces grow exponentially.

### Decision

```text
REJECT_NRM_EXACT_PATTERN_MACROS_AS_CORE
```

A cross-position algebraic macro is not NRM anymore; it becomes a static
transform/circuit mechanism and must pass those existing Gates separately.

---

## Overall decision

```text
NO_THREE_INVENTION_CORE_SURVIVES_E0_EVIDENCE_AUDIT
```

| Candidate | Mathematical core | Target path | Verdict |
|---|---|---:|---|
| CSPE | identity valid | fixed sparse dictionary fails counting and real restricted Gates | rejected as universal core |
| STC | relation valid | no subdense contraction; causal and arithmetic Gates fail | rejected as standalone core |
| NRM | exact on a hit | exact hit coverage and storage are far below requirement | rejected |

No numbered experiment, checkpoint download, CUDA work, target-host action, or
405B claim is authorized by this audit.

## Reproducibility

```bash
python experiments/e0_three_invention_validation/run_validation.py \
  > results/e0_three_invention_validation/summary.json
pytest -q tests/e0_three_invention_validation/test_validation.py
```

Expected:

```text
7 passed
deterministic core 9024a1ccb8da797dbf2567dcfd5c059507debe7733b41a090fdbfe9cb4e11e27
```

## README_UNCHANGED_REASON

The fixed mission, public quick start, repository map, and currently advertised
runtime capabilities are not changed by an E0 rejection audit. This audit adds
no user-facing executor or benchmark, so README modification is not warranted.
