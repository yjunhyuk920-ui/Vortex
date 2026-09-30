# Ordered tree grammars: paid native-source accounting

## Result and scope

A finite syntax constructor is available, and the existing scalar constructor can
represent supported native addend streams exactly. Neither supplies a cheap
heterogeneous dense-matrix effect source. This is a source/theorem-scope audit,
not an impossibility result, new core principle, model experiment or speed claim.
It clarifies O1/O3/O4 dependencies; full mission O1–O6 remain open under the
[Constructive Theory Contract](../../docs/CONSTRUCTIVE_THEORY_CONTRACT.md).

## Primary-source statement

[Ganardi, Hucke, Jeż, Lohrey and Noeth, *Constructing small tree grammars and small
circuits for formulas*](https://ii.uni.wroc.pl/~aje/prace/JCSS2017.pdf) was inspected
on 2026-09-30. Theorem 20 (printed p.24) gives an O(n)-time constructor of a
Chomsky-normal-form TSLP for an ordered n-node tree with σ labels of constant
maximum rank: size O(n log σ / log n), depth O(log n), and bounded nonterminal
rank. Section 2 uses a RAM with O(log n)-bit arithmetic at unit cost.

Lemma 22 (printed pp.26–28) instead converts a particular monadic TSLP into an
O(|G|)-size, O(depth(G))-depth arithmetic circuit preserving its polynomial over
every semiring. Its context representation A0+A1·x·A2 uses semiring laws. Lemma 21
is the intervening monadic-grammar construction. Thus Theorem 20 supplies the
syntax bound; Lemma 22 does not independently supply a native floating-point
evaluation bound. The paper also notes (printed p.3) that encoding nonterminal
references incurs logarithmic bits per reference.

These are paraphrases, not quotations. No hidden big-O constant is set to one.

## Native interpretation, without semiring reassociation

The [complete native-chart report](../binade_transition_20260930/native_charts/REPORT.md)
and [constructor](../binade_transition_20260930/native_charts/constructor.py)
give exact ordered scalar successor maps for non-NaN accumulators, finite exact
dyadic addends, gradual underflow and RNE, including signed zeros and overflow.
The BF16 wrapper obtains each addend as the exact product with its zero sign.

Grammar substitution can preserve the original ordered syntax. For any supported
scalar sequence, interpreting its steps with these maps preserves that sequence's
words without assuming floating-point associativity or distributivity. This is
a semantic bridge, not Lemma 22's constant-size context theorem. Arbitrary
grammar contexts, multiplication contexts, map composition, guards and dynamic
side expressions still need supported constructors and paid representations.
Expanding the grammar and running the original scalar constructor is a finite
fallback construction, but provides no speedup.

The chart API is not a verified arbitrary GEMM/HF ABI. Actual instruction order,
reduction trees, registers, FTZ/DAZ, nonfinite products, logits, KV/layout and RNG
remain separate obligations. Dynamic activation changes also change the addend
streams; model-once syntax preparation cannot make query-time maps free.

## Distinct leaves retain their coordinate source

For W with M rows and N input coordinates, an exact product-source key is
(coordinate j, weight word w_ij, ABI). An ABI identifier includes the prescribed
product/rounding interpretation; it is not permission to change that operation.
Equal coefficient words at different coordinates generally multiply different
inputs. Counting only BF16 values therefore omits source identity.

For a fixed ABI, let D = sum_j |{w_ij : 0 <= i < M}|. Finite BF16 has
2·255·128 = 65,280 bit patterns, including both signed zeros. Consequently
D <= N·min(M, 65,280), and D = MN is possible for M = 16,384 by choosing distinct
finite words within each column. This is an allowed counting witness, not a claim
about measured weights or reachable HF activations. Semantic coincidences on a
particular input do not prove an all-input sharing rule.

With M = N = 16,384 and P = MN = 268,435,456:

- Paired leaves: σ approximately D = P and n approximately 2P give
  log2(σ)/log2(n) = 28/29 = 0.96551724; the reciprocal is 1.03571429
- Split weight/coordinate leaves: 65,280 + 16,384 + 2 = 81,666 nominal labels
  and n approximately 4P give log2(σ)/log2(n) = 0.54391493, with reciprocal
  1.83852279, about 1.84

These are formal logarithmic factors, not measured compression, guaranteed
finite-instance savings, speedups or lower bounds. The split encoding itself
roughly doubles syntax nodes. Its 1.84 factor is relative to that expanded tree,
not to raw weights or paired-leaf syntax. The two operation labels are schematic;
zero seeds, ABI tags and a bounded-rank output collector add labels/nodes and
must be counted in a concrete tree. Adding one collector label gives σ = 81,667
and leaves the rounded illustration unchanged. A high-rank M-output root would
not meet the theorem's constant-rank premise.

Splitting leaves does not remove the D distinct coordinate/word product requests
of direct dictionary evaluation. That evaluation pays D exact products per input
plus routing and uses P products when D = P. This is accounting for this proposed
evaluation, not a lower bound on every exact algorithm. A different shared source
must be constructed and costed rather than presumed absent or free.

## Paid resource ledger and materialization examples

Charge syntax/source scans, checkpoint storage, label and nonterminal-reference
bits, coordinates, ABI/layout metadata, product reads and arithmetic, dictionary
lookups, addresses, grammar construction, map creation/composition, chart cuts,
guards, dispatch, scratch storage, transfers, checks and any exact repair.
Include initialization/TTFT and a justified finite reuse horizon. For a grammar
with g records, even an ordinary indexed encoding needs explicit reference widths
on the order of log2(g); O(g) grammar symbols are not O(g) bits or bytes.

The existing chart wrapper specifically pays MN coefficient accesses, MN input
accesses and MN exact products, then its chart-construction overhead. These are
logical operation/access counts; they are not asserted to be distinct physical
memory transactions. Streaming and caching may change residency or traffic, but
their schedule and costs need proof. The source theorem does not eliminate these
terms or establish a target-memory/latency bound.

If all P records were materialized simultaneously, chosen payload allowances give:

- 24 bytes per literal record: 6,442,450,944 bytes = 6 GiB
- 256 bytes per chart record: 68,719,476,736 bytes = 64 GiB

These are illustrative allocations, not minimal representations or mandatory
peak-memory lower bounds. The 256-byte allowance is a derived packed-chart
allowance from the native report, not measured Python object size. Actual chart
counts, old/new lists, integer temporaries, indexing and allocator costs are extra.

The [registered parameter count](../../gate0_budget.json) is 405,849,243,648;
at two bytes each the complete BF16 coefficient payload is 811,698,487,296 bytes.
This arithmetic is neither a claim that every parameter is read each token nor
that all bytes must reside on the GPU. It is the full registered BF16 payload,
and the cost of a complete payload sweep before other work.

## Disposition and provenance

Retain the grammar constructor and exact native-map bridge as auxiliary tools.
The cited theorem's guarantees do not establish a paid >=10x source or target
budget. This is insufficiency of the supplied guarantee, not evidence that no
better grammar, source or executor can exist. The decisive missing deliverable
is a causal heterogeneous effect constructor with sufficient full-cost bounds.

Local and remote branch `research/cloud-continuation-20260930` were read at
[5e5b5bf6ec87677d652d44f8840f68e1ea2fbf1c](https://github.com/yjunhyuk920-ui/Vortex/commit/5e5b5bf6ec87677d652d44f8840f68e1ea2fbf1c).
The branch-scoped PR search returned no PR. The native-chart package was present
as local untracked evidence and was not represented as already published.
Root README was reviewed; its native-constructor statement predates that local
package. Root-ledger integration belongs to the parent and was not edited here.

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
HANDOFF_STATUS=IN_PROGRESS
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN

Only source inspection and text/math/link checks were performed. No model run,
GPU work, new checkpoint, runtime benchmark, commit, push or publication occurred.
See [local receipt](LOCAL_CHECKS.md) and [derived arithmetic](accounting.json).
