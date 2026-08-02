# AtlasLinear fast-path milestone — 2026-08-02

## Purpose

The previous VTX format still read a complete low-bit base matrix before selective residual refinement. That cannot scale to the 405B/4B wall-clock target because base traffic alone dominates.

This milestone introduces a different normal path:

```text
x -> project into cached input basis U -> evaluate cached operator image WU
```

The original weight is loaded only when the input has a component outside the current basis.

## Implemented operator

`OnlineAtlasLinear` stores:

- an orthonormal input basis `U`;
- its exact linear image `WU`;
- cold-read, fast-hit, rank, byte, and capsule metrics;
- safetensors persistence for the atlas capsule.

For an input in the cached span:

```text
x = U z
W x = (W U) z
```

No base or residual weight tensor is loaded on that path.

For a miss:

1. load the exact cold weight;
2. compute the exact result;
3. extract the orthogonal input residual;
4. add that direction and its exact image to the atlas;
5. retain the cold weight as the correctness fallback, not the steady-state path.

## Runtime integration

`StreamingLlama` now routes selected internal projections through an operator boundary. The current validated integration covers:

- `self_attn.o_proj.weight`;
- `mlp.down_proj.weight`.

Atlases can be saved and loaded independently from the model weights.

## Reproduced results

```bash
python -m pytest -q
python scripts/run_validation.py
```

Observed:

```text
10 passed
```

### Synthetic rank-8 trace

Matrix: `96 x 64`

- exact/allclose output: yes;
- learned rank: 8;
- cold weight reads: 8 over 128 calls;
- fast-path fraction: 93.75%;
- persistent capsule: 5,120 bytes.

### Internal tiny-Llama replay

Managed operators: all four layers' O and down projections.

- first run builds the atlas;
- atlas is persisted;
- a fresh runtime loads the atlas;
- generated token sequence matches the build run;
- replay cold weight reads for managed operators: 0;
- replay fast vectors: 120;
- capsule size: 84,480 bytes.

## What this proves

This is the first committed VORTEX path where an internal Transformer projection can execute without reading either the full base weight or its residual, while retaining an exact cold fallback and reproducible token equality on the validated trace.

## Immediate next gate

Run the same operator on a real pretrained 1B–3B Llama-family model and measure, per projection and layer:

- rank growth;
- hit rate on continued generation and neighboring prompts;
- cold streams per token;
- capsule bytes;
- exact token agreement;
- CPU/GPU wall-clock.

The design advances only if rank growth flattens and cold traffic falls sharply on unseen continuation traces.
