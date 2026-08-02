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
