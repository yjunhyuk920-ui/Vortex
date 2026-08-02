# VORTEX implementation audit — 2026-08-02

This audit compares the repository's executable code, tests, and measured output against the fixed flagship goal:

> Run a 405B-class Hugging Face model under 8GiB VRAM, without user training or model-specific work, at native-4B-class wall-clock speed while preserving the declared original-model quality mode.

## Audit method

The repository was revalidated with:

```bash
python -m pytest -q
python scripts/run_validation.py
python scripts/audit_current_architecture.py
```

Observed result:

```text
7 passed
validation script completed successfully
```

The tests are valid for the behavior they cover. They do not yet test a production GPU path or a real pretrained model.

## What is genuinely implemented

- Local Hugging Face safetensors layout and shard discovery.
- Individual tensor and row-slice loading.
- A byte-budgeted tensor LRU.
- A CPU reference Llama decoder that loads internal dense matrices on demand.
- An automatic LM-head transcode into an INT8 base plus exact FP32 residual.
- Sound row/tile bounds for exact greedy argmax certification.
- Real selective residual slices for the disk-backed LM head.
- Exact causal fixed-point/Jacobi output equality on deterministic tiny random checkpoints.

## Actual validation scale

The generated test checkpoint contains:

```text
4 layers
hidden size 64
intermediate size 160
vocabulary 257
approximately 0.197 million parameters
```

The scale gap to the 405.85B planner model is approximately 2.06 million times.

The current tests therefore validate runtime plumbing and local mathematical properties. They do not establish scaling behavior.

## Critical implementation boundaries

### No real CUDA execution path has been validated

All committed tests run on CPU. `DiskProgressiveLinear` loads CPU safetensors and performs CPU matrix operations. Passing `device="cuda"` to `StreamingLlama` would leave the hidden state on CUDA while the progressive LM-head tensors remain on CPU, so the current end-to-end path is not a working CUDA runtime.

`ByteBudgetLRU` accounts only for tensors placed in that cache. It does not account for:

- attention/MLP activations;
- KV cache;
- CUDA allocator reserve and fragmentation;
- GEMM workspaces;
- temporary tensors;
- progressive proof buffers.

Therefore the 8GiB condition has not been measured or enforced on real hardware.

### The runtime is not universal yet

The CLI accepts a local directory, not a Hugging Face repository ID. `StreamingLlama` explicitly rejects every `model_type` except `llama`. There is no tokenizer integration, generic graph lowering, sampling path, MoE path, multimodal path, or custom-operator fallback.

### Internal Transformer execution is still exact dense streaming

Q, K, V, O, gate, up, and down projections load their complete matrices and call `torch.nn.functional.linear`. Progressive execution currently applies only to the LM head.

### Current in-memory residual metric is logical, not physical

`ProgressiveLinear.certify_argmax` precomputes every residual tile contribution before its refinement loop. Its reported residual fraction describes which contributions the proof logically consumed, not how much compute or memory traffic was physically avoided. The disk-backed implementation does perform selective residual reads and is the relevant path for I/O measurements.

### Current “4/5/6-bit base” is stored as INT8

`transcode_hf_linear` stores `quant` using `torch.int8` for every configured base precision. Therefore 4-, 5-, and 6-bit modes currently have identical base weight byte size. Bit packing is not implemented.

The exact residual is stored in FP32. Scale and three bound arrays are also FP32.

## 405B projection of the current format

The reproducible calculation is in `current_architecture_audit.json`.

If the existing VTX format were applied unchanged to all 126 internal Llama 3.1 405B linear projections and the LM head:

```text
INT8 base:                 376.02 GiB
FP32 exact residual:      1504.08 GiB
FP32 scales:                11.75 GiB
FP32 bound metadata:        35.25 GiB
------------------------------------------------
Total, excluding embedding: 1927.10 GiB
```

More importantly, even if **zero residual tiles** were needed, every target pass would first read the complete base and metadata:

```text
minimum base + metadata traffic: 423.02 GiB per target pass
```

A 4B Q4 model contains approximately:

```text
1.86 GiB of weights
```

Thus the current format requires at least:

```text
227.1 certified tokens per 405B weight stream
```

just to match the 4B model's weight bytes per token. This excludes computation, residual reads, KV traffic, transfer inefficiency, and proof overhead.

The measured tiny-checkpoint Jacobi result is:

```text
mean committed block: 1.275 tokens
```

The optimistic amortization gap is therefore approximately:

```text
178.1x
```

The current Jacobi implementation also resets the KV cache and evaluates the entire prefix plus guess block on every target pass, so target-pass count alone overstates its performance value for a real long prompt.

## KV-cache residency

For Llama 3.1 405B with 126 layers, 8 KV heads, 128-dimensional heads, and BF16 K/V:

```text
2K context:     0.98 GiB
4K context:     1.97 GiB
8K context:     3.94 GiB
16K context:    7.88 GiB
128K context:  63.00 GiB
```

The current implementation has no KV compression or offload policy capable of maintaining the full declared context under 8GiB.

## Definitive current verdict

### Proven now

- The repository contains a functioning CPU research prototype.
- Tiny Llama weights can be streamed under a simulated tensor budget.
- Exact greedy LM-head decisions can be certified using selective disk residual reads.
- Exact Jacobi output equivalence can be demonstrated on tiny deterministic random models.

### Not proven now

- A working CUDA end-to-end runtime.
- Real 8GiB peak VRAM.
- A real pretrained 1B, 7B, 70B, or 405B run.
- Progressive internal Transformer projections.
- Original-model benchmark quality.
- 4B-class wall-clock performance.
- Universal Hugging Face model support.

### Architecture verdict

The final target **cannot be reached by merely extending the existing full-base VTX format to every internal projection**. The complete base matrix traffic alone is too large.

The next architecture must make the normal path avoid reading the complete 405B base weights. A viable candidate must combine:

1. a small persistent operator representation that handles most real activations without full base reads;
2. an exact cold-weight fallback for failed gates;
3. multiple committed tokens per cold weight stream;
4. compressed/offloaded KV;
5. a fused CUDA/storage pipeline;
6. direct wall-clock validation on the scaling ladder.

## Revised next milestone

Before implementing progressive Q/K/V/O with the current full-base format as if it were the final solution, perform a falsification milestone:

1. Introduce an operator interface and retain exact dense streaming as baseline.
2. Measure one real pretrained 1B–3B model on actual activation traces.
3. Implement a cached subspace/operator-image fast path that does not read the full base matrix.
4. Measure gate hit rate, cold residual traffic, exact-token equality, and wall-clock.
5. Reject the design unless measured bytes per token fall on a scaling trajectory compatible with the final 4B gate.

No final feasibility claim is valid until this milestone produces real-model and real-hardware measurements.
