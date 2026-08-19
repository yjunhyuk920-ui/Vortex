# Fixed-Public Dynamic Executor — Output-Row Streaming Constructor

Status: **PREREGISTERED IMPLEMENTATION ARM**

Authority: `docs/research/FIXED_PUBLIC_DYNAMIC_EXECUTOR_DIRECTIVE.md`

Starting remote commit for this arm:

```text
2380bb74b7a713b4f200ff9c1a66427e650996a6
```

This document freezes the third implementation primitive before its hosted result is observed. It does not rename either previously executed primitive.

## 1. Measured failure term being attacked

The last hosted actual-checkpoint result recorded the complete DEV-W layer-0 reference parameter footprint as:

```text
7,080,192 B
```

For `cold_packed_projection_sequential_materialization` the measured terms were:

```text
artifact                    = 4,208,296 B
compiled layer resident     = 1,771,776 B
peak full projection hot    = 1,769,472 B
accounted footprint         = 7,749,544 B
reference layer             = 7,080,192 B
```

Therefore the already exact G2/G3 implementation failed G4 because the full projection was materialized at once. Its hosted latency was also worse than reference (`105.321/202.285 ms` reference p50/p95 versus `137.679/234.948 ms` candidate p50/p95).

The second primitive, `checkpoint_mlp_torchinductor_existing_isa`, failed exact transition and did not remove checkpoint residency. It is not the basis of this constructor.

## 2. Frozen constructor

Mechanism identifier:

```text
checkpoint_mlp_output_row_streamed_lossless_existing_isa
```

Frozen tile size before hosted execution:

```text
output_tile_rows = 128
```

For each real checkpoint MLP projection (`gate_proj`, `up_proj`, `down_proj`):

1. partition only the **output-row axis** into ascending contiguous 128-row tiles;
2. store each tile losslessly (`zlib-9` only when smaller, otherwise raw), with raw and artifact SHA-256;
3. at runtime materialize one tile, execute ordinary `torch.nn.functional.linear`, and append the output rows in original order;
4. do not partition the input/reduction dimension of any dot product;
5. preserve the original complete Llama MLP expression `down(act(gate(x)) * up(x))` and the complete Transformer-layer residual/normalization boundary.

This is a different runtime primitive from full-projection materialization: the live decoded weight extent is bounded by one output-row tile rather than one complete projection. It is also distinct from TorchInductor compilation; it uses ordinary existing PyTorch/host ISA operations and an explicit cold-data stream.

## 3. Pre-result physical prediction

For DEV-W dimensions `hidden_size=576`, `intermediate_size=1536`, BF16:

```text
gate/up max tile = 128 * 576  * 2 = 147,456 B
down max tile    = 128 * 1536 * 2 = 393,216 B
```

Thus the preregistered peak decoded weight tile is at most:

```text
393,216 B
```

Using the previous artifact size only as a planning reference, not as a result for the new tiled artifact:

```text
4,208,296 + 1,771,776 + 393,216 = 6,373,288 B
```

This is below the `7,080,192 B` DEV-W reference-layer footprint. The hosted run must measure the actual tiled artifact bytes; chunking can change compression ratio, so this calculation is not accepted as G4 evidence.

## 4. Fail-closed gates

The hosted workflow remains authoritative.

- **G1:** load the pinned public DEV-W through official `LlamaForCausalLM.from_pretrained()`.
- **G2:** compare next token **and successor cache/RNG state** continuously. Any mismatch rejects this primitive.
- **G3:** the replacement must remain a complete real Transformer-layer boundary with the checkpoint MLP actually replaced.
- **G4:** use measured `artifact + compiled-layer-resident + peak-hot` or measured p50+p95 latency improvement. No estimated compression result can pass G4.
- **128 steps:** the frozen workload requires at least 128 continuous decode transitions per workload; aborting on mismatch records the failed step.
- **Integrity:** every tile must fail closed on artifact or raw hash mismatch.
- **Existing ISA:** no virtual/hypothetical opcode is introduced.
- **TARGET-W:** DEV-W success is not extrapolated into a 405B claim. TARGET-W G5/G6 remain NOT TESTED until actual gated tensors and hardware accounting are available.

## 5. Required resource evidence

Each decode trace must continue recording at least:

```text
wall_ns
cold_bytes
hot_materialized_bytes
pcie_bytes
gpu_bytes
kv_bytes
artifact_bytes
resident_model_parameter_bytes
compiled_layer_resident_parameter_bytes
replaced_layer_reference_parameter_bytes
decompression_ns
materialization_ns
projection_calls
integrity_probes
algorithmic_flops
cpu_rss_bytes
peak_vram_bytes
isa_instruction_count/status
```

The new constructor additionally records `output_tile_rows` in the mechanism result and manifest. The real hosted result, whether pass or failure, must be committed to the canonical ledgers by the workflow.
