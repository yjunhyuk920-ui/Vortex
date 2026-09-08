# Independent scalar proof and cost review

Date: 2026-09-09
Reviewed inputs: `native_sparse_extension/PREREGISTRATION.md` and
`vortex-independent-audit/docs/CONSTRUCTIVE_THEORY_CONTRACT.md`
Method: static proof and accounting review only. No code, native execution,
Hugging Face run, GPU run, Git operation, or configuration change was used.

## Verdict

The proposed scalar sparse projection has a valid restricted theorem. For the
declared scalar ABI, replacing inactive signed-zero leaves by `+0` preserves
every nonzero, infinity, and canonical-NaN FP32 root word. The original root is
`-0` exactly when every original leaf product, including padding leaves, is
`-0`. Therefore one per-row positive-zero count, adjusted for active logical
slots, is sufficient to repair the only information lost by pruning.

That conclusion is conditional on the exact tree, rounding, NaN, metadata, and
store definitions below. It is not a theorem about a general native BLAS,
PyTorch CPU linear layer, CUDA kernel, or Hugging Face model. In particular,
the optional BF16 store must happen after FP32 signed-zero repair; a small
negative nonzero FP32 root can round to BF16 `-0`, even though not all FP32 leaf
products were `-0`.

The runtime bound is finite for admitted support `k <= 2`: `mk` selected BF16
coefficient reads, `mk` products, zero additions for `k <= 1` or exactly one
addition per row for `k = 2`, one shared `n`-word input scan, and `m` row-count
reads. No per-row `n`-element initialization or tree walk is required. The logical retained
matrix payload is `4mn` bytes when both row-major original BF16 weights and a
column-major BF16 copy are kept. This result supplies a restricted E1 producer
lemma. It does not close arbitrary dense inputs, O1-O5, the 405B/8-GiB mission,
or native HF/CUDA equivalence.

## The scalar theorem that is actually supported

Let `W` be an `m x n` matrix of finite BF16 words. Let `x` be a finite BF16
vector whose support `S = {j : |x_j| != 0}` has size `k <= 2`. For every
inactive slot `j`, the zero word of `x_j` must have the sign of a fixed declared
zero pattern `z_j`. Inputs with a nonfinite word, more than two nonzero-magnitude
coordinates, or a different inactive-zero sign are rejected.

Let `N = 2^ceil(log2(n))` for `n >= 1`. Logical leaves `0..n-1` are

```text
p_ij = RN32(BF16_to_FP32(W_ij) * BF16_to_FP32(x_j)),
```

with a separate FP32 multiply. Leaves `n..N-1` are `+0`. The root `R_i` is the
specified balanced binary tree over these `N` leaves, with a separate FP32
round-to-nearest-even addition at each internal node. The scalar ABI must map
an invalid `+infinity + -infinity` operation to one declared canonical quiet
NaN and must map every later operation receiving a NaN to that same canonical
quiet NaN.

Under those definitions, the compiler may retain W by columns and, for every
row, store

```text
C_i = count over logical j of positive-zero Mul32(W_ij, z_j).
```

At runtime it computes

```text
A_i = count over j in S of positive-zero Mul32(W_ij, z_j)
I_i = C_i - A_i.
```

`I_i` is exactly the number of positive-zero products among the inactive
logical leaves. The selected `W_ij` words already reveal the signs needed to
compute `A_i`; no free per-query oracle is needed. Padding is handled separately
by the public fact `N != n`. Equivalently, padding may be included in `C_i`, but
the implementation must freeze one convention and use it everywhere. Mixing
the two conventions would double-count padding.

For the admitted `k <= 2` domain, the optimized evaluator need not materialize
the original leaf positions. Let the selected actual product words be `p` and,
when present, `q`. It computes `Q_i = +0` for `k = 0`, `Q_i = p` for `k = 1`,
and `Q_i = Add32(p,q)` for `k = 2`, regardless of the two logical positions.
It then emits:

```text
R_i = Q_i,                         if Q_i is nonzero, infinity, or NaN
R_i = -0,                          if Q_i is zero, N == n, I_i == 0,
                                      and every selected actual product is -0
R_i = +0,                          otherwise
```

For `k = 0`, the last condition over selected products is vacuously true. This
is needed so an all-`-0` unpadded row returns `-0`.

## Proof that zero-sign replacement preserves nonzero and NaN roots

Define `a ~ b` for FP32 words when they are bit-identical or when both are
zeros and only their sign bits may differ. Treat the single declared canonical
quiet NaN as one exact word.

The fixed addition is compatible with this relation. Changing only an input
zero sign cannot change a finite nonzero input, an infinity, or the condition
for invalid opposite infinities. Adding either signed zero to a nonzero finite
word or to an infinity returns the same nonzero word or infinity. If an input
is the canonical NaN, the canonicalizing add returns the same canonical NaN.
Only an output that is itself zero can acquire a different sign.

Each inactive original leaf and its `+0` replacement are related by `~`.
Induction from leaves to root therefore gives `R_i ~ Q_i`. Replacing an entire
zero-only subtree by one `+0` is the same induction compressed into one step.
Consequently, if either root is nonzero, infinite, or NaN, both roots are the
same FP32 word. Intermediate product overflow does not break the proof: the
same selected product overflows to the same signed infinity in both trees.
Opposite infinities meet at the same selected-path ancestor and produce the
same canonical NaN.

## Proof of the position-independent `k <= 2` reduction

The direct two-product optimization is valid for this restricted domain.

For `k = 1`, every node between the selected product `p` and the root combines
the branch containing `p` with a zero-only sibling branch. A zero-only branch
evaluates to a signed zero. By the `~` congruence above, each such addition
leaves the branch bit-identical when it is nonzero, infinite, or canonical NaN,
and can change only a zero sign. Thus the full root is related by `~` to `p`.

For `k = 2`, let `L` be the lowest common ancestor of the selected leaves.
Below `L`, each selected product travels through only zero-only sibling
branches, so the two inputs arriving at `L` are respectively related by `~` to
the original product words `p` and `q`. The operation at `L` is the only
addition in the full tree whose two branches contain selected products.
Congruence gives

```text
value_at_L ~ Add32(p, q).
```

Above `L`, the result again meets only zero-only sibling branches, so the full
root remains related by `~` to the single direct addition. The locations of
`p`, `q`, and `L` therefore cannot affect any nonzero, infinity, or canonical
NaN result. If a finite product addition overflows, it overflows in the same
single `Add32(p,q)` operation. If product overflow made `p` and `q` opposite
infinities, the original operation at `L` and the direct operation both return
the declared canonical NaN; all later NaN propagation stays canonical. If the
direct result is a zero, the global signed-zero rule reconstructs its exact
original sign from the padding, inactive-positive count, and actual product
words.

This argument also proves `k = 0`: the numerical root is some signed zero and
the same global rule determines its exact sign. Therefore the producer may use
zero adds for `k <= 1` and one add per row for `k = 2`; it does not need leaf
arrays, per-row `n`-element initialization, lowest-common-ancestor lookup, or
`log n` path traversal.

The proof does not extend to arbitrary `k`. With three active products, the
balanced grouping can change overflow and cancellation. A valid BF16-product
example is `p = 2^127`, `q = 2^127`, and `r = -2^127`, obtained from finite
BF16 weights of those values and BF16 inputs `+1`. The grouping
`Add32(Add32(p,q),r)` returns `+infinity`, while
`Add32(p,Add32(q,r))` returns `2^127`. Inputs with `max_support > 2` must
therefore be rejected rather than sent through this optimization.

## Proof of the `-0` criterion

For FP32 RNE addition, an internal node is `-0` if and only if both children
are `-0`.

* `-0 + -0` is `-0`.
* Any other pair of zero signs returns `+0` under RNE.
* The exact sum of two finite FP32 numbers is an integer multiple of the
  minimum FP32 subnormal. If it is nonzero, its magnitude is at least that
  minimum and cannot round to zero. Exact cancellation of nonzero operands is
  `+0` under RNE.
* An infinity operation returns an infinity or NaN, never zero. A NaN operation
  returns the canonical NaN.

Induction over the balanced tree now proves that its root is `-0` if and only
if every leaf is `-0`. A single padded `+0` therefore makes a `-0` root
impossible. On an unpadded tree, `I_i == 0` says that all inactive logical
products are `-0`, while the selected-product test says that all active leaf
products are `-0`. Together they are exactly the leaf condition.

This also covers underflowed selected products. Finite BF16 operands can
produce a signed FP32 zero after product underflow, and the selected-product
test must inspect the actual FP32 product word rather than infer nonzeroness
from the BF16 operands. Finite BF16 operands can also overflow to signed FP32
infinity; those rows follow the nonzero/infinity/NaN proof above.

## Required word-level clarifications

These are proof preconditions, not implementation preferences.

1. **Canonical NaN must be a total rule.** Naming a canonical quiet NaN only
   for the first invalid add is insufficient. State its exact FP32 word, make
   every add with a NaN input return that word, and state the canonical BF16
   NaN word for a stored result. RNE does not select a NaN payload.
2. **Repair precedes store.** First reconstruct the exact FP32 root sign. Then
   convert that FP32 word to BF16. Do not force every zero *after* BF16
   conversion according to the all-leaves rule. A negative nonzero FP32
   subnormal can legitimately round to BF16 `-0` while the original FP32 root
   was not zero.
3. **Freeze the reference tree completely.** Specify `N`, left/right pairing at
   every level, logical leaf numbering, and padding location. The optimized
   query does not walk this tree for `k <= 2`; it uses the direct cases proved
   above. This shortcut and the hard rejection of `max_support > 2` must be
   explicit in the declared algorithm and operation count.
4. **Freeze FP semantics.** The executable scalar artifact must ensure BF16
   decode, separate FP32 multiply, separate FP32 add, RNE, gradual underflow,
   and no FMA contraction, FTZ/DAZ, extended accumulator, or host-dependent NaN
   propagation. A bit-level software reference is the cleanest proof object.
5. **Define the count type.** A 32-bit row count is enough at `n = 16384`, but
   the general constructor must reject larger dimensions or use a count width
   that represents `0..n`. Subtraction must be checked, not allowed to wrap.
6. **Validate the whole input.** A caller-supplied support list alone does not
   prove the other `n-k` words are zeros with the fixed signs. The claimed
   `n`-word scan is necessary unless a separately proved trusted input type is
   introduced and paid for.

## Cost review

The following is the logical payload and work count for one `m x n` matrix. It
does not include object headers, alignment, allocator state, indexes, buffers,
thread pools, page tables, verification, or device/runtime state.

### Compilation

* Read and validate exactly `mn` BF16 source words.
* Inspect `mn` source signs against the fixed zero-sign vector while building
  `m` row counts. This can share the validation pass but is still `mn` work.
* Write exactly `mn` BF16 words into the column-major compiled copy.
* Write `m` row counts and the declared zero-sign bit vector.
* A practical transpose needs a finite tiling/buffering schedule. Its scratch,
  cache traffic, synchronization, checksums, and read-back verification are
  additional paid terms.

When the original and compiled BF16 copies are both retained, their payload is

```text
2mn + 2mn = 4mn bytes.
```

For 32-bit counts and one shared sign bit per logical input coordinate, the
additional logical metadata is

```text
4m + ceil(n/8) + dimension/tree headers bytes.
```

The `4mn` expression is therefore a matrix-payload count, not total resident or
peak compile memory. At 405B parameters, retaining two BF16 copies alone is
about 1.62 TB decimal before any metadata, model state, SSD staging, host RAM,
PCIe/HBM movement, allocator, verification, or fallback cost.

### Admitted sparse query

For one vector and all `m` rows:

* Scan `n` BF16 input words to validate finiteness, support, and every inactive
  zero sign.
* Read `mk` selected BF16 weights, exactly, from the compiled columns.
* Read `m` row counts. Read the `k` fixed zero-sign bits and dimensions/tree
  description; do not count these as free.
* Perform `mk` FP32 products.
* Perform no additions for `k = 0` or `k = 1`, and exactly `m` FP32 additions
  for `k = 2`. Per-row active-sign subtraction and final zero repair are
  `O(mk + m)` integer/control work. No `n`-element leaf buffer, tree-path walk,
  or lowest-common-ancestor computation is needed per row.
* Write `m` FP32 results, or `m` BF16 stored results, plus any status/refusal
  output.

Thus the optimized admitted query is `O(n + m + mk)`, with exactly `mk`
coefficient reads and at most `m` FP32 additions. Since `k <= 2`, this is
`O(n+m)`, but it is not constant-cost end to end: the single whole-input scan
and all-row metadata/output terms remain. In particular, the `n` scan occurs
once per query, not once per row.

For `m = n = 16384`, the minimum logical payload figures are:

| Item | Bytes |
|---|---:|
| Original BF16 matrix | 536,870,912 (512 MiB) |
| Compiled BF16 columns | 536,870,912 (512 MiB) |
| Both matrix payloads | 1,073,741,824 (1 GiB) |
| 32-bit row counts | 65,536 (64 KiB) |
| Fixed zero-sign vector | 2,048 (2 KiB) |
| Input scan | 32,768 (32 KiB) |
| Selected columns, `k=1` | 32,768 (32 KiB) |
| Selected columns, `k=2` | 65,536 (64 KiB) |
| FP32 output | 65,536 (64 KiB) |
| BF16 output | 32,768 (32 KiB) |

These numbers are payload lower components, not a measured latency or peak
working set. CPU control, selected-index construction, transpose tiles,
allocator fragmentation, state, initialization, checksum/verification,
recovery, and fallback must be added to any end-to-end bound.

### Dense boundary

Setting `k = n` is outside the optimized producer's admitted domain whenever
`n > 2`; it must be rejected or sent to a separately declared paid fallback.
A separate full balanced reduction can use
`mn` products and `m(n-1)` additions, but it reads all `mn` weights and needs a
paid row-major or explicitly scheduled column-major implementation. Retaining
the original row-major copy permits a conventional dense fallback; that
fallback is full reconstruction and its traffic, latency, workspace, and tail
frequency all count. It is not evidence for the <=8-GiB or same-machine latency
mission.

The local compiler also does a full `mn` scan and a full `mn` transpose write.
Those finite costs are acceptable for the restricted lemma, but no amortization
horizon has yet been proved for the fixed mission.

## Hugging Face fixture review

The planned fixture is useful as a bounded bridge test, not as a lifting proof.
Its checkpoint is constructed to keep hidden states two-sparse and to make the
attention/MLP output and logits structurally zero: token embeddings have two
`+/-1` entries, RMSNorm weights are one, q/k/o/MLP/lm-head tensors are zero, and
only bounded BF16 `v_proj` weights vary. This can exercise selected-column V
production, K/V cache append, position tracking, logits, sampling, and RNG
state without claiming a public target model has sparse activations.

Before a native run, one immutable metadata manifest must confirm at least the
Transformers and PyTorch versions, model class/config, hidden/intermediate/head
dimensions, dtype, device, thread and backend settings, RMSNorm epsilon, RoPE
variant/scaling, cache class/layout, attention implementation, tokenizer or
direct token-ID convention, sampling API/options, seed 260909, exact weight
generator and exponent/mantissa domain `[-8,8]`, context bound, and checkpoint
hashes. A later run is evidence only for that manifest.

Separate proof or native word-by-word measurement is required for each of the
following:

1. **RMSNorm words and support.** Standard Llama RMSNorm with two `+/-1`
   coordinates does not generally have scale one; it divides by the RMS with
   epsilon and casts according to the actual implementation. Prove or measure
   the two active BF16 output words and every inactive zero sign. “Unit norms”
   must mean an exact frozen parameter/configuration, not an assumed arithmetic
   identity.
2. **CPU linear ABI.** PyTorch CPU GEMV/GEMM is not known to use the declared
   padded balanced, separate-product FP32 tree. The scalar theorem cannot be
   cited as native parity. For this bounded two-support fixture, a separate
   lemma may show order independence for nonzero words when every BF16 product
   is exactly representable and only one two-term sum is relevant, but signed
   zero, backend accumulation, output casting, and any exact cancellation still
   require proof or word-level comparison. Do not silently substitute the
   scalar reference for the native model.
3. **Zero-state transitions.** A zero q/k/o/MLP tensor does not by itself fix
   every zero sign. Linear kernels, SiLU, multiplication, residual adds, and
   casts can change inactive zero signs. Establish the fixed pattern at each
   layer and each use of `v_proj`; do not assume the embedding pattern persists.
4. **RoPE and K cache.** Zero keys passing through RoPE can acquire
   position-dependent signed-zero words through multiply/add/subtract rules.
   Define a finite context bound, construct the template with the exact native
   implementation, pay its storage/construction, and compare every K word for
   prefill and incremental paths separately.
5. **V cache and DynamicCache state.** Compare every V word, dtype, shape,
   layer ordering, sequence position, append/copy behavior, and all successor
   metadata after each token. Freeze the exact `DynamicCache.update` contract.
   Tensor values alone are insufficient if the declared successor state
   includes other fields.
6. **Residual and attention legality.** Show bounded sources keep every
   RMSNorm, RoPE, score, softmax, V aggregation, residual, and cache word finite.
   Zero `o_proj` makes the attention contribution semantically zero, but the
   native path must remain legal and its zero signs must not change later
   admitted inputs unnoticed.
7. **Full logits.** A zero lm-head should yield zero logits, but exact signed
   zero words remain a native-kernel fact. Compare all logits, not just argmax,
   probability equality, or sampled tokens.
8. **RNG and sampling.** Snapshot native and compiled RNG states before and
   after each sample, call the same frozen native sampling procedure with the
   same logits/options/generator, compare the sampled token, and compare the
   complete next RNG state. Forward execution must also be checked for any RNG
   consumption. Equal zero-valued logits alone do not prove successor RNG
   equality.
9. **Prefill versus incremental.** Run and compare these as distinct native
   paths. Agreement in one is not evidence for the other, because attention,
   RoPE, and cache kernels can differ.
10. **No dense escape.** Instrument or structurally audit the compiled path so
    it cannot call the original dense `v_proj` or full model forward. Reference
    execution for comparison is separate and paid. Hash the checkpoint before
    and after both paths.

Passing all of these checks would establish a reproducible result for one
small, bounded, deliberately sparse CPU fixture. It would not establish native
CUDA behavior, arbitrary Hugging Face linear order, a public unchanged 405B
checkpoint, arbitrary legal continuations, or target memory/latency.

## Contract status and precise next obligations

This work is a continuation of the already selected restricted E1 producer. It
is not three new qualifying execution principles and does not reset the
architecture comparison. The earlier rejected full-reconstruction,
free-witness, row-replay, full-scan, catalog, free-decoder,
global-advice-per-matrix, sparse-delta, and arbitrary dense-change assumptions
remain rejected. The count is paid row metadata, and the fixed zero pattern is
an explicit admission restriction; neither is a free selector for the fixed
mission.

The obligation status remains:

| Obligation | Review result | Next required artifact |
|---|---|---|
| O1 uniform construction | OPEN | A constructor and admission proof for every arbitrary public unchanged 405B checkpoint and all legal mission inputs; the fixed two-support domain does not qualify. |
| O2 causal program | OPEN | A complete load, prefill, token step, output, KV/state, refusal/fallback, and termination algorithm for the unchanged mission. The fixture is only a bounded sketch. |
| O3 exact outputs/RNG/state | OPEN | First close the scalar word spec above; then provide a separate native HF proof or immutable word-by-word fixture record for full logits, K/V, positions, RNG, and continuation at every step. |
| O4 complete cost | OPEN | Add compiler wall schedule, scratch/peak memory, SSD/RAM/PCIe/HBM traffic, indexes, CPU/GPU work, validation, verification, cache, initialization, allocator, recovery, and fallback bounds. The payload formulas here are only components. |
| O5 target memory/latency | OPEN | A target-scale <=8-GiB allocation schedule and coupled same-machine p50/p95 and TTFT proof/measurement against native 4B Q4. No current hardware evidence exists. |
| O6 independent artifacts | PARTIAL | Publish the total scalar ABI, canonical NaN/store rules, tree/count format, executable bit-level reference, compiler format, proof tests, immutable HF manifest, and first-failure record. This review supplies only the independent static proof audit. |

The next decisive order is:

1. Freeze the total scalar ABI and metadata format, including canonical FP32 and
   BF16 NaN words, tree topology, count convention, checked count width, and
   repair-before-store rule.
2. Produce a small bit-level reference plus a formal or exhaustive finite-word
   check of the addition congruence and `-0` induction. Adversarial cases must
   include product overflow, opposite infinities at different lowest common
   ancestors, product underflow to both zero signs, exact finite cancellation,
   non-power-of-two padding, `k=0/1/2`, and BF16 store underflow.
3. Finish the compiler/query artifact and report exact counters for coefficient
   reads, metadata reads, products, additions, support/sign checks, input/output,
   compile reads/writes, peak scratch, and retained storage at 16384 square for
   `k=1`, `k=2`, and a separately labeled rejected-or-paid dense fallback.
4. Only after the immutable HF metadata manifest confirms the intended fixture,
   run the native CPU bridge comparison. Preserve the first failing word and do
   not alter the source range, seed, checkpoint, or thresholds after observing
   it.
5. Treat a fixture pass as O6 evidence for the restricted bridge and return to
   the unresolved arbitrary-dense constructor and complete 405B cost schedule.
   No fixture result promotes `CORE_ADMISSION`, theory status, or hardware
   status by itself.

Accordingly, the honest status remains:

```text
CORE_ADMISSION=false
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
O1-O5=OPEN
O6=PARTIAL
```
