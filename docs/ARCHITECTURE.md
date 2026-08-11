# Architecture

## Current executable path

```text
Hugging Face model directory
        |
        v
HuggingFaceLayout
  - reads config.json
  - reads safetensors index or scans shards
  - maps tensor name -> shard
        |
        v
TensorLocator
  - loads one tensor, one layer, or one slice
        |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
StreamingLlama           transcode_hf_linear     OnlineAtlasLinear
  - row-sliced embedding   - low-bit base         - input basis U
  - byte-budget cache      - lossless residual    - operator image WU
  - operator routing       - per-tile norms       - exact cold fallback
  - KV cache               - disk manifest        - persistent capsule
        |                      |                      |
        v                      v                      v
internal hidden state    DiskProgressiveLinear   base-free span execution
        |                  - base logits
        +----------------->- residual bounds
                           - selective refinement
                           - exact argmax certificate
```

## Module responsibilities

### `vortex_runtime/hf_loader.py`

`HuggingFaceLayout` reads model metadata and creates a tensor-to-shard map. `TensorLocator` supports exact tensor loading and safetensors slicing without instantiating a Transformers model.

### `vortex_runtime/tile_cache.py`

`ByteBudgetLRU` tracks actual tensor byte sizes, evicts least-recently-used entries, and records peak residency. It is a correctness-oriented model of a future VRAM tile cache.

### `vortex_runtime/llama.py`

`StreamingLlama` is a reference implementation for Llama-style checkpoints. It performs exact layer execution while retrieving tensors through the bounded cache. Selected projection suffixes can now be routed through `OnlineAtlasLinear`. The module also contains exact sequential generation and Jacobi-style block generation for equality testing.

This module is not yet an optimized CUDA backend.

### `vortex_runtime/atlas_linear.py`

`OnlineAtlasLinear` is the first base-free internal fast path. It stores an orthonormal input basis `U` and exact operator image `WU`. Inputs in the cached span execute as `(WU) @ (U.T @ x)` without loading the original matrix. Span misses invoke the exact weight loader and expand the atlas. Capsules are persistable through safetensors.

The validated milestone routes attention O and MLP down projections through this operator.

### `vortex_runtime/progressive.py`

`ProgressiveLinear` quantizes a dense matrix into a low-bit center plus exact residual. It stores multiple residual norms per row/tile and computes sound dot-product bounds. `certify_argmax` refines residual tiles until one output row's lower bound exceeds every competitor's upper bound.

### `vortex_runtime/vtx_linear.py`

Defines the current disk-backed VTX linear format:

- quantized base values;
- per-row/per-tile scales;
- lossless residual matrix;
- L1/L2/L-infinity residual metadata;
- row-block safetensors files;
- JSON manifest.

`DiskProgressiveLinear` computes base logits, bounds unread residuals, reads selected residual slices, and certifies the exact argmax.

### `vortex_runtime/planner.py`

Produces tensor-size and model-size estimates from architecture dimensions. The committed validation report includes a Llama 3.1 405B plan.

### `vortex_runtime/toy_model.py`

Creates deterministic tiny Hugging Face-compatible Llama checkpoints. Tests do not require downloading external models.

## Planned model-wide path

The preferred normal path is now AtlasLinear rather than full-base progressive evaluation:

```text
activation -> atlas gate -> cached U/WU execution
                       \-> exact cold stream on miss -> atlas expansion
```

Disk progressive execution remains useful for final decision certification and cold fallback formats. Additional projections are added only after real-model trace validation demonstrates bounded rank growth and declining cold streams.

A future model-wide path may combine:

```text
input state
  -> atlas or exact Q/K/V
  -> attention
  -> atlas or exact O
  -> residual merge
  -> atlas or exact gate/up
  -> SiLU and elementwise product
  -> atlas or exact down
  -> residual merge
  -> progressive LM-head proof
```

## Backend boundary

The Python implementation establishes semantics and metrics. A production backend will require:

- FP16/BF16 or packed atlas capsule storage;
- fused basis projection and operator-image kernels;
- asynchronous exact-weight fallback staging;
- pinned-memory and CUDA stream scheduling;
- compressed/offloaded KV policy;
- CUDA graph or persistent-kernel execution;
- real GPU memory accounting.

Backend optimization must not alter the exactness contract without explicit validation modes and recorded quality measurements.

## EXP-076 native-MTP reference boundary

`vortex_runtime/mtp_acceptance.py` contains pure longest-prefix, exact commit,
percentile, parameter-traffic, K-selection, and Gate helpers. The heavy runner
under `experiments/exp_076` maps the pinned Qwen3.5 text and native-MTP tensors
into a CPU BF16 reference, maintains independent committed/proposal/verification
caches, and reconstructs post-rejection state from the exact committed prefix.

This component is retained as falsification infrastructure only. EXP-076 found
held-out accepted-prefix p05/p50/p95 `0/4/4` at selected `K=4`, below required
`9/11` tail/median minima, so it is not a promoted decoding backend. It is not
vLLM equivalence, a CUDA implementation, a quantized path, or an operation-
replacement runtime.

## EXP-077A Fractal MLP reference boundary

`vortex_runtime/fractal_oracle.py` contains pure trace-validation, strict
fraction, aggregation, and Gate helpers. The runner under
`experiments/exp_077a` replays frozen EXP-076 prefix/verification caches and
temporarily replaces every SwiGLU MLP with a top-channel favorable oracle.

This is retained as a negative-test instrument, not a backend. At a realized
`9.988839%` MLP fraction it preserved only `71.5278%` of held-out top-1 target
decisions, despite seeing full current intermediates for free. It does not
sparsify attention, DeltaNet, or the LM head and has no deployable selector,
traffic, CUDA, VRAM, or speed claim.

## EXP-079A causal proof-state reference boundary

`vortex_runtime/causal_proof_state.py` contains the pure budget, DCT pilot,
correlated row-block enclosure, Gate, and fail-closed state transitions. The
throwaway terminal driver exposes hot-bound, refine, certify/commit, and exact
fallback states. The heavy runner temporarily replaces every Qwen3.5-0.8B MLP
down output with a DCT center plus oracle-selected exact row blocks.

This is not a backend. The selector sees the full residual for free, the local
L2 balls are not propagated through later nonlinear layers, ideal Q4 bytes are
logical accounting, and exact fallback is semantic only. A pass would authorize
only the nonlinear proof-propagation stage; no CUDA or target-server work is
authorized by EXP-079A.

EXP-079A did not pass that boundary. At the p50 logical budget the dual-oracle
reference preserved only `4.1667%` held-out top-1 and left a minimum local sound
radius roughly `49x` the signal. The DCT/block-zonotope components remain
negative-test infrastructure only; they must not be wired into the production
operator path or optimized with a backend.

## Extension-field and joint-coset research boundary

`vortex_runtime/extension_field_rank_saturating_frontier.py` is a pure finite
calculator. It records exact GF(2) repacking equations, direct q-system
storage, independent-batch union traffic, and tiny nonlinear dictionary
witnesses. It is not an operator, source format, decoder, or backend.

The unimplemented joint-coset interface would select one atom set `T` for a
whole query batch such that every query lies in `span(G_T)`. No persistent
layout, sub-dense selector, native arithmetic, causal guarantee, or physical
cost closure exists, so it must not appear in the runtime path or be described
as a surviving candidate.

The exact geometry is now narrower. A K-query batch lies in
`span(r_i) tensor span(u_i)`, of dimension at most `K^2`; at K=32 this is a
1,024-dimensional Joint Factor Envelope. Product-simplex generalized weights
give the exact largest rank-one intersection of any selected span. These facts
authorize neither a cached envelope catalog nor a restriction oracle: the
catalog has more than `2^1,046,528` registered names, and the strongest finite
support count obtained here forces only 20,218 bit atoms. Any future component
must generate the envelope restriction implicitly and charge every native
summary, address, probe, and decode operation.
