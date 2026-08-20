# EXP-087A — Lossless Entropy-Stationary Perfect-Block Gate

## Status

```text
Phase: A/B plus small-real-checkpoint observation
Evidence ceiling: E1
Scientific result: PENDING OFFICIAL CHECKPOINT RUN
Real complete-layer operation replacement: false
Target hardware: NOT TESTED
```

## Question

After EXP-086A showed that a perfect 128-position block amortizes one complete
MLP weight stream below the logical traffic target but leaves exact residual
arithmetic at about twenty percent of the full model, can an execution class
that performs **no weight or state skipping** still approach the fixed mission
by combining:

1. exact tile-local lossless checkpoint coding;
2. one decode/materialization per perfect future block; and
3. weight-stationary block GEMM rather than one GEMV per token?

This Gate separates three questions that previous experiments mixed:

```text
lossless storage/transfer reduction
native finite-precision block equivalence
physical block-vs-sequential execution efficiency
```

It does not assume that dense arithmetic disappears. The 405B arithmetic floor
is reported explicitly and may reject the path once the same-machine native 4B
baseline is measured.

## Execution-dependency change

EXP-085A tried to select a subset of nonlinear SwiGLU pages and failed even with
an exact-contribution oracle. EXP-086A tried to derive future activation states
from exact low-order recurrence programs and failed even with future target
activations and free coordinate-wise parents.

EXP-087A does neither. It retains every coefficient and every dense product.
The checkpoint tile is read and decoded once while `K` exact future activations
are resident:

```text
compressed W tile -> exact W tile -> W @ X[:, 0:K]
```

The possible benefit comes only from:

- fewer physical checkpoint bytes through lossless coding;
- one checkpoint stream amortized across `K` positions; and
- improved hardware utilization from matrix-matrix execution.

Because this is a different execution dependency, a failure of EXP-086A does
not decide this Gate.

## Frozen public checkpoint and population

```text
model       HuggingFaceTB/SmolLM2-135M
revision    93efa2f097d58c2a74874c7e644dbc9b0cee75a2
loader      transformers.LlamaForCausalLM.from_pretrained
runtime     BF16, eager attention, pinned repository requirements
layer       0 complete SwiGLU MLP
tensors     gate_proj, up_proj, down_proj
families    English, Korean, code, structured JSON, mathematics, repeated pattern
positions   128 exact greedy causal MLP inputs per prompt
blocks      13, 32, 64, 128
codec       independent 1 MiB tiles; zlib level 9 or raw fallback
```

`K=13` is retained because current tree-speculation research has reported
acceptance in that range. The final partial block is excluded deterministically;
no padding or selected-state substitution is allowed.

## Lossless compiler

For every frozen weight tensor:

1. serialize the exact BF16 words in native order;
2. divide the payload into fixed 1 MiB tiles;
3. encode each tile with zlib level 9;
4. retain raw bytes whenever compression would expand a tile;
5. store SHA-256 for encoded and decoded payloads; and
6. reconstruct the complete tensor and require byte-for-byte equality.

The Gate also measures the empirical Shannon entropy of complete BF16 words.
This is a favorable information floor, not a realizable format unless a
separate exact entropy coder is constructed.

No quantization, weight modification, retraining, tolerance, or approximate
codec is permitted.

## Native block ABI Gate

For every captured block and each projection, compare:

```text
sequential reference:
    concatenate F.linear(x_i, W) for i=0..K-1

block candidate:
    F.linear(X, W)
```

The same comparison is performed across the complete SwiGLU MLP:

```text
W_down[SiLU(W_gate X) * (W_up X)].
```

One CPU thread and the non-MKLDNN PyTorch path are frozen to reduce unrelated
backend variation. Any BF16 word mismatch means ordinary block GEMM does not
preserve the declared reference ABI and the path must be revised before a
physical speed claim.

This Gate does not prove GPU ABI equivalence. A pass only establishes the
pinned CPU reference boundary.

## Accounting

For observed raw bytes `B`, encoded bytes `E`, empirical Shannon bytes `H`, and
perfect block length `K`:

```text
zlib ratio       = B/E
Shannon ratio    = B/H
logical traffic  = 1/(ratio*K)
```

The registered p50/p95 fractions remain:

```text
p50 = 1.2*4/405 = 1.185185185%
p95 = 1.5*4/405 = 1.481481481%
```

The target-surface diagnostic uses the already recorded private target
inventory:

```text
favorable PCIe Gen2 x16 ceiling = 7.4506 GiB/s
root free space                 = 97.6183 GiB
M5000 published FP32 peak       = 4.3 TFLOP/s (external projection)
```

For source size `S`, lossless ratio `c`, and block length `K`:

```text
T_io/token = S/(c*B_ext*K)
T_compute/token >= 2*405,849,243,648/F_peak
T_roofline >= max(T_io, T_compute)
required native 4B p50 >= T_roofline/1.2
```

The compute term deliberately remains independent of `K`: block GEMM can raise
utilization toward the hardware peak, but it does not remove coefficient
multiplications. This prevents I/O amortization from being mislabeled as an
operation-count reduction.

Both an ideal packed-Q4 source and the original BF16 source are reported.
Neither is a measured 405B artifact.

## Frozen controls

Before a checkpoint verdict:

- constant byte and BF16-word entropy must be zero;
- every encoded tile and complete tensor must round-trip bit-exactly;
- corruption must fail closed through the stored hashes;
- synthetic block linear output must equal sequential output;
- synthetic complete SwiGLU block output must equal sequential output;
- I/O time must decrease monotonically with compression ratio and `K`;
- dense compute floor must remain independent of compression and `K`;
- deterministic result regeneration must preserve the evidence core.

## Frozen decisions

### Native ABI failure

```text
REVISE_LOSSLESS_ENTROPY_STATIONARY_NATIVE_BLOCK_ABI_MISMATCH
```

This permits only an explicitly reference-order-preserving block kernel. It
does not authorize changing the output contract.

### Codec failure

```text
REJECT_ZLIB_TILE_CODEC_RETAIN_ENTROPY_LOWER_BOUND
```

This closes zlib as the actual weight codec when the registered tensors do not
shrink, while retaining the measured entropy floor as an input to a separate
coder Gate.

### Conditional survival

```text
CONDITIONAL_LOSSLESS_ENTROPY_STATIONARY_PATH_REQUIRES_TARGET_BASELINE_AND_TARGET_ENTROPY
```

This requires all of:

```text
zero weight round-trip mismatch
zero projection/full-MLP block mismatch
encoded bytes < raw bytes
K=128 zlib logical traffic <= 1.185185185%
```

A conditional result is not target success. It authorizes only:

1. exact entropy measurement on accessible target tensors or a representative
   larger public checkpoint;
2. a reference-order-preserving fused decoder/GEMM prototype; and
3. the separately authorized same-machine Stage-2 native 4B and H2D baseline.

### Insufficient block/codec

```text
REVISE_LOSSLESS_ENTROPY_STATIONARY_REQUIRES_LONGER_BLOCK_OR_STRONGER_CODEC
```

A larger speculative block is not authorized unless an independent causal
proposal Gate supplies the needed accepted length.

## Strongest falsification

Even a perfect lossless ratio and perfect future block cannot beat the dense
compute floor. When the same-machine native 4B p50 is measured, reject this
execution class if:

```text
2*405,849,243,648/F_effective > 1.2*T_4B_p50
```

or if the exact reference-order kernel cannot approach the required effective
throughput. This is a physical Gate, not inferred from the small CPU run.

## Stop rule

Do not use a favorable CPU block speedup to claim 405B performance. Do not
multiply zlib, Shannon, speculative, and GEMM ratios when they reduce the same
resource. Do not hide decompression, materialization, KV, draft, rejected
branches, synchronization, or source storage.

## Claim boundary

Even a conditional pass remains:

```text
future exact activations used             true
causal accepted block                     NOT TESTED
405B exact lossless entropy                NOT TESTED
reference-order GPU fused kernel           NOT TESTED
same-machine native 4B baseline            NOT TESTED
target H2D/storage bandwidth               NOT TESTED
405B execution / 8 GiB / E6 / E7           NOT TESTED
```

Structurally valid conditions were established. Large-model performance remains
unverified.
