# Native two-support BF16 producer — executed, restricted result

The surviving E1 producer now accepts arbitrary **finite BF16 matrix words**
and up to two nonzero BF16 input coordinates, instead of binary coefficients
and a single basis coordinate. A finite column compiler and paid zero-sign
metadata produce the exact declared balanced FP32 effect without scanning all
matrix entries at query time. The source/input restriction is explicit.

A separate, deliberately restricted CPU Llama fixture actually replaces its V
projections with this producer. The final packaged run preserved 40 steps of
logits and serialized K/V state, full-prefill logits and K/V values, and eight
native `generate` samples with each sample's probabilities and RNG states.
The compiled runtime cannot call dense Linear/model forwards. This is a real
small operation replacement, not an arbitrary unchanged HF executor.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`,
`CORE_ADMISSION=false`, O1-O5 OPEN and O6 PARTIAL. The persistent goal remains
unachieved. The user has no 405B-capable hardware; no target run was attempted.

## Selection and scope

The existing A/B/C frontier is preserved. A is the surviving restricted causal
producer; B needs a direct globally nonlinear dense decoder; C needs a native
dynamic summary whose update is cheap even when every input coordinate changes.
Only A currently supplies the bounded object expanded here. B and C remain
OPEN without an invented decoder. This is **not** three new qualifying core
principles or a reset of the >=10x mission. It follows NEXT_EXPERIMENT's explicit
source-alphabet/right-factor expansion task.

The new scalar theorem handles every finite BF16 matrix, but only admitted
inputs with support <=2 and a fixed inactive-zero sign vector. The HF source
is much narrower: a generated 14,496-parameter Llama checkpoint with two-support
embeddings, bounded BF16 V weights, unit norm weights, and all other dense paths
zero. Its logits are consequently a simple zero interface; the nontrivial new
output is the generated V cache. This source is not a public 405B checkpoint.
No source alphabet expansion here makes arbitrary dense hidden states sparse.

## Finite Compile / Address / Read / Decode

For m rows, n columns, let N be the next power of two >=n. Operands are finite
BF16 words. The scalar ABI decodes BF16 exactly, performs separate FP32 RNE
products, places them at logical leaves 0..n-1, pads leaves n..N-1 with +0,
then performs adjacent-pair balanced FP32 RNE additions until one word remains.
Every NaN-producing or NaN-consuming operation emits FP32 0x7fc00000; BF16
store emits 0x7fc0. Gradual underflow and repair-before-BF16-store are required.
This is an explicit scalar ABI, not an assumption about arbitrary BLAS/CUDA.

1. **Compile(W,z).** Validate all finite BF16 source words, transpose them into
   actual stored columns, and count, in each row, positive-zero products with
   the fixed signed-zero vector z. Include N-n positive padding leaves in this
   count. The source scan, sign comparisons, output copy and allocation are paid.
2. **Address(G,x).** Scan all n input words. Reject nonfinite inputs, mismatched
   inactive-zero signs, support above two, or max_support outside 0..2. Record
   the increasing list of active logical columns. Their source-dependent
   coefficient addresses are ordinary column-base plus row-offset addresses.
3. **Read.** Read each selected column's m actual BF16 words and up to m row
   counts. Read the input, fixed sign pattern and dimension/control metadata.
   There is no source-dependent free address descriptor or result catalog.
4. **Decode.** Compute zero, one or two FP32 products per row. For two, perform
   one FP32 addition in logical order. Subtract from the precomputed row count
   the active columns' hypothetical positive-zero contributions. If the FP32
   result is zero, restore -0 iff the remaining count is zero and every selected
   actual product is -0. Otherwise return +0. Store the repaired word to BF16.

The actual implementation is `scalar_producer.py`. Python integers make row
count arithmetic nonwrapping. Four-byte counts in numerical tables apply at
n=16384; a packed general format would need ceil(log2(N+1)/8) bytes per count.
No general packed format or GPU implementation is asserted to exist.

The fixed inactive-zero sign vector and row counts are charged metadata.
The implemented extra column copy is a source-sized cold sidecar if not resident.
At 405B BF16 parameters this copy alone would be 810 GB decimal; it does not
fit a canonical-source-plus-8-GiB-cell-pool model. This representation belongs
to the explicitly larger cold-sidecar setting, and that storage remains paid.

All loops terminate for finite admitted dimensions. Compile is O(mn+m+n)
source/metadata work and O(mn+m+n) live representation/scratch words. Address
and Decode are O(n+m+mk), k<=2, with O(n+m+k) non-source working objects in this
Python implementation. There is no remaining per-row n-element initialization.
These finite primitive/work bounds do not bound an arbitrary host allocator,
device transfer schedule, or target wall time.

## Correctness

Changing a zero sign preserves addition modulo zero sign: nonzero finite
words, infinities and the canonical NaN remain identical. Induct over the fixed
tree. Every omitted zero-only branch can therefore be carried through without
changing the eventual nonzero/NaN result. With at most two selected leaves,
there is at most one ancestor that combines two selected products. Every other
ancestor adds only zeros. Thus direct zero/one/two-product evaluation is the
same as the original tree modulo final zero sign, regardless of leaf positions.
This is not associativity and does not extend to three arbitrary products.

An FP32 RNE sum is -0 iff both children are -0. A nonzero exact sum of finite
FP32 operands is an integer multiple of the minimum subnormal, so it cannot
round to zero. Exact nonzero cancellation is +0. Infinity/NaN never gives zero.
Induction proves that a -0 root requires every original leaf, including padding,
to be -0. The paid row count plus selected product signs tests exactly this
condition. Actual underflowed products are inspected; nonzero operands alone
do not determine the product's zero sign. Repair occurs before BF16 store,
because a negative nonzero FP32 result can subsequently round to BF16 -0.

For k>2, the producer refuses. The separate `full_reference` routine is finite
and pays mn source reads/products and m(N-1) adds. It is not an automatically
integrated fast fallback and does not meet the mission by being exact.
The independent review contains a three-term BF16 rounding witness; silently
extending the direct sum rule would be incorrect. All-coordinate legal changes
remain the existing decisive boundary against sparse-delta generalization.

## Actual native bridge and its boundary

`hf_bridge.py` constructs the frozen ordinary Llama tensors before hashing.
Compilation validates the complete checkpoint, verifies all 32 normalized
embedding rows using the native RMSNorm implementation, compiles V columns,
and builds a paid 64-position native RoPE signed-zero key template. Query state
contains only these stores/scales/templates and native DynamicCache; it holds
no model. The query derives its two active coordinates from the current token,
produces V, constructs K, appends the cache and returns the fixed zero logits.

The native control is torch 2.8.0+cpu / transformers 4.55.4, BF16, one compute
thread, deterministic algorithms, SDPA as recorded in `hf_results_v5/manifest.json`.
All model configuration/defaults, build information, direct token-ID convention,
seed and cache class are recorded there. No CUDA path ran.

The 40-step trace is all 32 token IDs then [31,0,17,17,3,25,0,31]. Each step
compares logits dtype/raw words, each cache object's instance fields, and tensor
dtype/device/shape/stride/requires_grad/layout/raw words, storage offsets and
backing-storage sizes, conjugate/negative-view bits, container types, and the
internal cache storage-alias graph. Actual storage addresses are
not compared. Native full-prefill logits and K/V values also agree with the
incremental result; this is not a claim that prefill and incremental cache
strides are identical. Eight native `generate` samples use the frozen prefix,
sampling options and identical RNG start. The canonical v5 additionally checks
every generation logit word, probability word, sampled token, and complete
CPU RNG state before/after each sample, plus final serialized cache state.

This is measured agreement on these traces. It is not an inductive native-HF
proof for every sequence up to 64 positions, nor for longer legal continuations,
other backends/configurations, arbitrary checkpoints, arbitrary input embeddings,
cache edits, RNG devices or sampler options. The 64-position template is an
executor restriction, not a claim that the native HF API forbids longer input.
The scalar theorem cannot close this native ABI/continuation gap by substitution.
In particular, prior KV values are carried/appended but do not affect the zero
logits. Nontrivial cache-dependent attention/output preservation is not tested.
No aliases to arbitrary caller-owned tensors or mutation API are in this
comparison; internal cache alias equality is narrower than every HF API state.

## Paid cost and failed universal admission

For k<=2, per query, the scalar producer scans n input words, reads mk BF16
coefficients, charges up to m count reads and sign/control metadata, performs mk products
and at most m max(k-1,0) additions, and emits m result words. Sign comparison,
subtraction, branches, address arithmetic, allocations/conversion and output
are additional O(n+m+mk) work. Preprocessing reads mn words once and writes mn
column entries, with mn sign/count operations. The `compile_matrix_reads` and
`compile_validation_reads` counters describe the SAME fused pass and must not
be double-counted. The legacy `positive_control_index_terms=mn` field describes
the compile-time sign/count inspection, not a surviving runtime full scan.

At m=n=16384,k=2: 32,768 coefficient reads/products, 16,384 adds/count reads,
16,384 input checks and 16,384 outputs. The logical BF16 selected-column payload
is 65,536 bytes; 32-bit counts add 65,536 bytes. Original plus compiled BF16
matrix payload is 1,073,741,824 bytes, before 65,536 count bytes and 2,048 zero-sign bytes. Input/output/addresses and physical Python object traffic are
additional. The two-source-payload formula 4mn is not actual Python peak RSS:
tuple/list references, integer objects, over-allocation, transient list-to-tuple
copies, interpreter/runtime/allocator and page traffic are all paid, unclosed
host terms. A compact BF16 storage implementation was not measured here.

The HF fixture reads 64 BF16 V weights per token rather than 1,024, a 93.75%
coefficient-payload reduction. Adding its 32 four-byte counts gives 256 bytes
against 2,048 original V bytes: only 87.5% reduction even before input, output,
key template, cache and physical object traffic. Therefore it has NOT passed
a >=90% total-traffic gate at this shape. There is no timing win claim.

Compilation retains/validates all 14,496 source parameters (28,992 BF16 payload
bytes), transposes 1,024 V words, verifies 2,048 normalized words and retains a
256-byte key template plus scales/sign/count metadata. Source creation, model
initialization, copies, three complete checkpoint hashing passes in the final
runner, norm/RoPE calls, reference execution and verification are paid. The
original/generated matrix objects, temporary tensors and full reference cache
also occupy the validation process; its memory is not compiled-only memory.

Each new token appends 128 logical KV bytes. `DynamicCache.update` copies
128*T existing bytes at prior length T, then writes 128*(T+1) bytes into the
new arrays; these are read and write traffic, not one combined 128*T transfer.
Across a T-token run this copying is quadratic in T. The discarded/native
attention cost is not evidence that arbitrary source attention can be skipped.
An allocation/validation exception may leave a partially updated cache.
This experimental adapter does not promise transactional rollback or a reusable
post-failure state. It must abort that execution; any restart requires paid
initialization and prefix reconstruction. There is no implemented fast recovery.

Logit allocation, softmax, multinomial, probability/RNG control and sampler state
are O(vocabulary) plus implementation/RNG terms. CPU, RAM/cache/page movement,
SSD source/output I/O, initialization, verification, error/refusal, repair and
any separately chosen full dense fallback remain charged. GPU/HBM/PCIe execution
was not measured or assumed free. No remote computation was used.

For 405B retained original+compiled BF16 weights, matrix payload alone would
be 1.62 TB decimal. This is a storage component, not a claim that it must all
reside in GPU memory or an impossibility proof. With dense input, the concrete
fallback returns to all source reads and products. No arbitrary-checkpoint
sparsity guarantee, source compression, compiler amortization horizon, bounded
runtime allocator, SSD/PCIe/HBM schedule, coupled 8-GiB peak, native4BQ4 latency
ratio or TTFT upper bound has been constructed. O4/O5 stay OPEN.

## Verification and preserved failures

The agent executed the frozen scalar sweep over all 65,280 finite BF16 source
encodings and six dyadic input words, in four zero-pattern/width controls:
1,566,720 sweep cases plus four fixed cases; 1,566,724 total and five rejection
controls. This is NOT exhaustive over BF16 operand pairs or two-support inputs.
The primary independently ran 124,423 integer-dyadic oracle cases, including
2,592 canonical-NaN roots, 1,068 negative FP32 zeros, and 1,441 nonzero FP32
results storing to BF16 -0. No host-float arithmetic is used in that oracle's
exact significand operations. Neither finite grid substitutes for the proof.

Scalar v1 had a per-row n-element zero allocation and accepted max_support=3
while summing only two terms. Primary static review found these before HF ran;
v2 removed the allocation, fused compilation, and rejected unsupported support.
Historical scalar v1 output and source hashes survive, but its source bytes and
an actual pre-fix red execution were NOT retained. The v2 test is a post-fix
refusal regression. Do not describe it as a preserved v1 runtime reproduction.

HF v1 passed the narrower raw-value check. Expanded v2 stopped at step zero:
native first V stride was [16,2,16,1], candidate [16,2,2,1]. The failing state,
manifest and executed source snapshot are retained. The fix uses native
view(batch,time,heads,head_dim).transpose(1,2). V3 passed the expanded check.
V4 added generation-logit/full-prefill-logit checks. Canonical v5 additionally
binds the executed sources and invocation/environment, records storage offsets,
sizes and internal cache aliases, replaces the removable norm assert with an
explicit checked failure, and narrows machine-readable state claims to the
serialized fields. It passes the unchanged source/trace/thresholds. The checkpoint hash remains
4ebef0122256beb7bb571c3cabb8d65a4fdb870cdc3d6288330a70be5d8db23b.

Requested delegation: Terra/medium for the scalar implementation, Sol/high for
independent local proof and implementation review, with no recursion/GPU/Git.
Actual selected model metadata was unavailable; these are requested settings.
Primary owns the theorem/cost/native integration/O1-O6 judgments and separately
checked the source, integer oracle and actual HF execution.

The scalar metadata-read counters are conservative row-count budgets, not an
instrumented count of every physical read; nonzero rows may skip the count.
See `VALIDATION.json`, `obligations.json`, `SCALAR_INDEPENDENT_REVIEW.md`,
`HF_IMPLEMENTATION_REVIEW.md` and the immutable versioned results. No GitHub
Actions, 405B, CUDA, <=8-GiB peak, native4BQ4 timing or TTFT was executed.

## Next executable research obligation

Do not rerun this fixture or merely add a third sparse coordinate as the next
core discovery. Return to three distinct candidate mechanisms for a finite
native **arbitrary dense** global producer or dynamic state whose paid update
handles all-coordinate changes. Specify its checkpoint compiler, causal
address rule, actual read words, decoder and exact successor-state relation
before testing. The signed-zero metadata lemma is reusable only when its input
admission premise is proved; it does not supply such a premise for HF405B.
No general impossibility or general target-feasible algorithm was established.
