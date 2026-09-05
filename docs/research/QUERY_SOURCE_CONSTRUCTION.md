# Query-state partition source: explicit constructor, not a universal cheap executor

Date: 2026-09-05. Parent: `d4ae2ce79de49b90c1fc0188f212ba7d8b92550c` (PR #122).

**The requested universal low-cost information generator has NOT been constructed.**
This bounded reference removes an undefined selector from one examined method:
it actually builds a checkpoint-derived index, consumes only the current input
and prefix states, and computes every row result. Its index/selection costs and
ordinary/adversarial results disqualify it as the VORTEX core. No mission goal,
quantifier or acceptance threshold is changed.

```
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
```

## 1. Final theorem and current obligation boundary

The intended mission theorem still quantifies over arbitrary public unmodified
HF dense checkpoints, legal inputs, contexts, RNG states and original interfaces.
It requires automatic construction, causal prefill/decode, original output and
inductive successor-state preservation, completely paid resource upper bounds,
405B/total-8-GiB feasibility and the same-machine native-4B-Q4 p50/p95/TTFT goals.
This note proves only a serial numerical subroutine. It does not assert the
mission theorem, even conditionally on target hardware becoming available.

| Obligation | Actual result | Complete-mission status |
|---|---|---|
| O1 | Finite index constructor for finite binary32 matrices | OPEN: no bounded universal HF lowering or target-size representation |
| O2 | Explicit causal all-row serial-FMA generator | OPEN: no full prefill/decode program |
| O3 | Prefix partition invariant below | OPEN: no original HF/CUDA graph, RNG, logits or KV continuation proof |
| O4 | Exact serialized size and logical counters; polynomial worst-case work | OPEN: no favorable complete executor/machine upper bound |
| O5 | Ordinary/adversarial screen fails even this core's cheapness requirement | OPEN: no target-memory/latency/TTFT closure |
| O6 | Reproducible source, proof argument and bounded tests | OPEN: these do not verify the missing mission obligations |

The decisive missing construction remains a target-wide, economical way of
obtaining the otherwise omitted exact information. A correct reference procedure
which usually visits all information has not solved that problem.

## 2. Comparison with the existing frontier, before backend work

This continuation revisits the existing alternatives; it does NOT invent three
new names and claim three new high-upside execution principles.

1. Static native expression sharing: canonicalize identical native reduction
   paths at construction time. Sharing preserves the original expression but
   needs enough exact structure to remove work. F-040/F-050 already reject its
   promotion as a universal hot/cold core. This route was considered, then stopped
   before implementation; no new constructor or evidence defeats those failures.
2. Query-state partition source (implemented here): reverse the order of finding
   reusable numerical work. Start with actual current native accumulator classes,
   then refine them using checkpoint-derived weight membership, or certify a
   whole class using rounding monotonicity. The finite source and selector are
   explicit. A >=10x route exists only if many actual classes collapse; it is NOT
   a proved population property. This extends the scoped native-transfer tools,
   not a claim of a previously unknown universal technique. Its unfavorable
   worst-case bounds were registered before execution, and the small reference
   makes the candidate explicit rather than authorizing a backend.
3. Non-entrywise nonlinear cold encoding / exact continuation quotient: a query
   could in principle read a small code or evolve a smaller equivalent state.
   The existing work does not supply the bounded universal constructor, decoder
   or cheap state transition. Finite exhaustive function tables move the missing
   cost into preparation/storage and do not meet the unchanged mission. No
   undefined encoder, free proof generator or free state materializer is used.

None is promoted merely because the others were rejected. No new kernel,
model-wide compiler, CUDA work or Actions run is authorized by this result.

## 3. Executable specification and index

The reference is the pure word-valued operation

    s[i, j+1] = RN32_FMA(W[i,j], x[j], s[i,j]), j = 0,...,n-1.

`W` and `x` contain finite IEEE binary32 words. `s[i,0]` is supplied or +0.
The pinned `experiments/native_transfer/native_transfer.py` wrapper uses Linux
libm `fmaf`; rounding is nearest-even and floating-point exception flags are
not observed. Every original column is processed in order. This is NOT an
assertion that a Torch/cuBLAS kernel uses that serial reduction graph. Nonfinite
checkpoint/input words are rejected by this bounded prototype, not silently
removed from the mission domain. Index integrity is a precondition: `query`
uses an `Index` produced by `build`, not an untrusted forged object.

`build(W)` scans each original coefficient once. For column j it sorts exact
weight words, creates one row-membership bitmap M[j,w] for each distinct word,
and records the minimum and maximum numerical weight. No query or future output
is used. The index retains all coefficient information; a reconstruction test
recovers every original word from the serialized membership records.

An accumulator partition P contains pairs (a,R), where a is a binary32 word
and R is the bitmap of rows whose exact current accumulator is a.

```
P = group_rows_by_exact_initial_word(initial)
for j in original_column_order:
    next = []
    for (a, R) in P:
        u = native_fma(min_weight[j], x[j], a)
        v = native_fma(max_weight[j], x[j], a)
        if a is finite and u and v are identical finite NONZERO words:
            next.append((u, R))
        else:
            for (w, M) in all_weight_buckets[j]:
                B = R AND M
                if B is nonempty:
                    next.append((native_fma(w, x[j], a), B))
    P = sort_and_union_masks_for_identical_words(next)
return materialize_every_row(P)
```

The selector is bitmap intersection and deterministic sorting, not an oracle.
A failed range certificate refines the same pass; it does not obtain a free
reference answer or restart an uncharged dense execution. The index, input and
prefix partition are the ONLY information source. The native dense reference is
used exclusively by the audit/tests, never by `query`.

## 4. Scoped exactness proof

Inductive invariant before column j: partition masks are disjoint, their union
is every row, and each row's partition word equals that row's native dense prefix
accumulator after exactly j ordered FMAs.

Initialization groups the supplied exact initial words, establishing the
invariant. For a fixed finite a and x, multiplication by x and addition of a
are monotone in the numerical weight (possibly decreasing when x is negative).
Correct round-to-nearest is monotone too. Thus every result for a weight between
the stored extrema lies numerically between the two endpoint results. When both
endpoints have the same finite nonzero word, every intermediate result has that
same word. Nonzero is required here to avoid treating signed zero's two words
as a single numerical value. Ambiguous zeros and nonfinite endpoint cases do NOT
use this shortcut.

Otherwise R intersect M[j,w] partitions R by actual weight bits. All members
of a nonempty intersection have exactly the same operands (w, x[j], a), so one
call to the original word-valued native operation gives every member's correct
successor word. Merging only identical result words preserves the invariant.
Finite loops advance j, establishing termination and every final row result.

This proves a conditional substitution lemma: where an original program node
has precisely this ABI and consumes these operands, replacing that node returns
the same roots. Repeating it from supplied exact accumulators preserves those
accumulators. It is NOT a proof that an arbitrary model's needed successor state
is just these accumulators, nor a substitute for implementing and proving the
original Transformer numerical graph, RNG and all future KV/state transitions.
The current tests include accumulator continuations, not generated LLM tokens.

## 5. Complete accounting boundary

Let m rows, n columns, L=ceil(m/64), sigma[j] distinct weight words, a[j] current
accumulator classes, and e[j] nonempty intersections for unresolved classes.
Then a[j]<=m, sigma[j]<=m and e[j]<=m. Row masks are disjoint within each partition.

**Construction.** All mn checkpoint words are read. Sorting column records costs
O(n*m*log(m)) word comparisons; constructing, shifting and merging row bitmaps
must also be paid, conservatively O(n*m*L) word work. Parsing/checking weights,
choosing extrema, allocating objects and encoding the index are additional paid
linear/bitmap work. Original checkpoint retention is extra storage, not replaced
by a claim of zero-cost semantic modification. Preparation is not amortized over
an assumed infinite session, and no short-session/TTFT target is established.

**Actual encoded index size:**

    S = 32 + 12*n + sum_j sigma[j]*(4 + 8*L) bytes.

This includes a 32-byte header, 12-byte column records, 4-byte weight words and
padded membership masks. The original FP32 matrix has 4mn bytes. The Python
prototype's tuples, integers, lists, sort workspace, input matrix, object headers
and allocator behavior require additional resident memory. Encoded size is NOT
measured process memory or total GPU memory. Encoding creates another temporary
byte buffer. The encoded file's roundtrip test is not a production mmap loader.

**Query arithmetic.** At most 2*sum(a[j]) endpoint FMAs plus sum(e[j]) bucket FMAs,
hence at most 3mn native FMAs. This is an upper bound which FAILS to prove the
needed speedup, not a claim that the algorithm does fewer than mn operations.

**Selection and state.** Unresolved classes can test every bucket, paying
sum_j a_fail[j]*sigma[j]*L bitmap-word intersections. Per column, at most m
next records are sorted/merged; include O(m*log(m)) word comparisons and
O(mL) mask copies/ORs. Initial partition construction, every cutoff popcount,
final row materialization, input/initial-state/output words and loop/address
work must also be paid. Integer bitmap extraction can cost O(mL), not merely
m free output stores. Worst-case selector work is O(n*m^2*L); auxiliary live
partition masks can require O(mL) words plus sorting records.

For equal-cost word primitives, one conservative asymptotic upper bound is

    O(nm*c_fma + nm^2*L + nm*log(m) + nmL + n + mL).

It is a poor bound for the target, not a machine-service latency guarantee.
The declared FMA counters alone omit substantial work. No favorable asymptotic
constant, overlap schedule, bandwidth or target timing is assumed.

**Movement.** `logical_cold_index_bytes` counts the encoded header/column records
and each required column bucket payload once. The in-memory Python index is
already resident; this counter is NOT a measured SSD/PCIe/HBM trace. It assumes
a needed column payload can be held while its state classes are processed.
Repeated mask/record access is counted separately, and spills/reloads would add
cost. `bitmap_AND_operand_bytes_upper` and `bitmap_merge_bytes_upper` are logical
fixed-width models, not measured traffic. No peak bandwidth is used as achieved
throughput. Full physical RAM/GPU allocation and latency remain unmeasured.

A unique-state/unique-weight family retains every bucket FMA and essentially
all indexed information. Accordingly this specific algorithm cannot supply a
universal >=10x elimination theorem. This is NOT a lower bound ruling out every
possible executor or nonlinear cold data structure.

## 6. Frozen experiment and observed outcome

Preregistration precedes execution: `experiments/query_source/preregistration.json`.
SHA-256: `7b36b4f54436087d71236d335b91b2beeb570e92e3b618e349be620d329e1d51`.
The bounded sizes are 16 and 64, with four current-input queries per population.
The xorshift32 generator and all formulas are in `audit.py`; matrix/input/output
hashes use canonical sorted compact JSON. Full output words and logical counts
are retained in lossless `results/query_source/audit.json.gz`.

At 64x64 the reference executes 4096 FMAs. The following counts hold for all
four recorded queries in each population; no percentile latency is measured.

| Population | Bucket FMAs | Total FMAs including range tests | Encoded index | Logical index bytes/query | Mask intersections/query |
|---|---:|---:|---:|---:|---:|
| Engineered absorption control | 0 | 128 (3.125%) | 48296 B | 800 B | 0 |
| Ordinary finite dyadic inputs | 4095 (99.975586%) | 12159 (296.850586%) | 49040 B | 49040 B | 253249 |
| Unique-column adversary | 4096 (100%) | 12162 (296.923828%) | 49952 B | 49952 B | 258112 |

The original FP32 matrix is 16384 B. The engineered control deliberately starts
with a huge equal accumulator contribution whose later small updates round away.
It is a correctness/shortcut positive control, NOT evidence about trained models.
Ordinary and adversarial populations have no successful range cuts; the latter
retains ALL dense bucket FMAs before selector and endpoint overhead.

Across 24 complete queries, every row word agrees with the direct serial
reference. Five focused unittest tests cover signed zeros/subnormals, rectangular
outputs, exact index reconstruction, validation errors and 16 accumulator
continuations. The direct serial oracle is independently scheduled, but shares
the already-existing native FMA primitive: this is not a new independent IEEE
hardware validation. Deterministic result regeneration matches byte-for-byte.
No actual checkpoint forward, HF layer replacement, full repository suite,
GPU/405B run, physical memory/traffic/latency test or GitHub Actions ran.

## 7. Additive decision, assumptions and next-work ledger

- Decision: retain this bounded explicit source as an unpromoted reference;
  reject it as the universal low-cost core. No performance capability is added.
- Unproved premise: enough native state/range collapses on the unchanged model
  and input domain. The ordinary/adversarial screen defeats claiming it here.
- Unproved lift: actual reference graph, complete model state and original RNG.
- Unclosed budget: index storage, initialization, selector work and target machine
  service/latency, including short sessions. More hardware alone does not close it.
- Do not optimize this index, choose a larger initial accumulator, sweep seeds,
  bucket/range widths or try more tiny matrices and call that core progress.
- A new direction needs a genuinely different information source with a bounded
  constructor and paid decoder. It must address the existing anti-repetition
  register and the original O1-O6, not merely attach a faster verifier.

This local research boundary is not mission completion. The broader source
construction remains open without asserting universal impossibility.

## 8. Reproduction and sources

From the repository root (Python standard library, pinned Linux libm wrapper):

```
python -m experiments.query_source.audit > /tmp/query_source_audit.json
python -m unittest discover -s tests/query_source -v
python -c "import gzip,pathlib; assert gzip.decompress(pathlib.Path('results/query_source/audit.json.gz').read_bytes()) == pathlib.Path('/tmp/query_source_audit.json').read_bytes()"
sha256sum -c results/query_source/checksums.sha256
```

Source dependency Git blob: `43be63155ae3be6496e2838c536b4bd09c99d41e` for
`experiments/native_transfer/native_transfer.py`, unchanged from the parent.
The user-facing bundle includes it for standalone execution. The PR only adds
new source/tests/evidence/docs and current root annotations. Old evidence is not
reclassified; preceding root text is preserved verbatim by original Git blobs under
`docs/research/history/pre_query_source_20260905/`.

- Governing [constructive contract](../CONSTRUCTIVE_THEORY_CONTRACT.md).
- Prior [native transfer construction](NATIVE_TRANSFER_CONSTRUCTION.md).
- [Permanent recent failure register](../../FAILED_APPROACHES_RECENT.md), F-040/F-050
  and related source/rounding/history failures. No broad impossibility conclusion.
- NVIDIA, Floating Point and IEEE 754: reduction order and FMA semantics matter.
  https://docs.nvidia.com/cuda/floating-point/index.html
- Anand et al., The Structural Complexity of Matrix-Vector Multiplication:
  structure-dependent data structures do not provide a universal native-ABI bound.
  https://arxiv.org/html/2502.21240v3
- Chakraborty, Kamma, Larsen, Tight Cell Probe Bounds for Succinct Boolean
  Matrix-Vector Multiplication: its probe model is not a GPU/PCIe time model.
  https://arxiv.org/abs/1711.04467
