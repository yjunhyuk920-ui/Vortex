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
        +----------------------+
        |                      |
        v                      v
StreamingLlama           transcode_hf_linear
  - row-sliced embedding   - low-bit base
  - byte-budget cache      - lossless residual
  - exact projections      - per-tile norms
  - KV cache               - disk manifest
        |                      |
        v                      v
exact hidden state       DiskProgressiveLinear
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

`StreamingLlama` is a reference implementation for Llama-style checkpoints. It performs exact layer execution while retrieving tensors through the bounded cache. It also contains exact sequential generation and Jacobi-style block generation for equality testing.

This module is not yet an optimized CUDA backend.

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

The next architecture extends VTX progressive execution into every projection:

```text
input state
  -> progressive RMSNorm input handling
  -> progressive Q/K/V
  -> attention with propagated uncertainty/correction
  -> progressive O
  -> residual merge
  -> progressive gate/up
  -> nonlinear uncertainty/refinement
  -> progressive down
  -> residual merge
  -> progressive LM-head proof
```

Three execution modes are expected:

1. **Base path:** low-bit, tile-streamed evaluation.
2. **Refinement path:** selectively reads lossless correction planes/tiles.
3. **Exact path:** streams all required residuals when certification cannot close.

## Backend boundary

The Python implementation establishes semantics and metrics. A production backend will require:

- packed 2–6 bit kernels;
- fused base-plus-refinement GEMM/GEMV;
- asynchronous CPU/RAM/NVMe staging;
- pinned-memory and CUDA stream scheduling;
- CUDA graph or persistent-kernel execution;
- compact residual metadata;
- real GPU memory accounting.

Backend optimization must not alter the exactness contract without explicit validation modes and recorded quality measurements.
