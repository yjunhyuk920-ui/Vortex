# Validation protocol

## Principle

VORTEX is evaluated by reproducible behavior and wall-clock results, not by architecture claims. Every benchmark must record raw configuration, hardware, model revision, prompt set, sampling settings, and generated outputs or output hashes.

## Correctness modes

### Exact greedy mode

For fixed model, prompt, tokenizer, and implementation-compatible numerics:

- token sequence must match the exact VORTEX streamed baseline;
- progressive certificates must never commit a token outside their proven bounds;
- any unresolved position must refine or fall back rather than guess.

### Sampling mode

For fixed seed and sampling algorithm, compare against the exact runtime implementation. When bitwise identity is not meaningful because kernels differ, record distributional and task-quality results separately and never label them exact.

## Unit and integration gates

Required on every change:

```bash
python -m pytest -q
```

Required when runtime behavior or metrics change:

```bash
python scripts/run_validation.py
```

The generated `validation_results.json` must be committed when intentional metric changes occur.

## Progressive operator metrics

Record per operator type and layer group:

- base bytes read;
- metadata bytes read;
- residual bytes read;
- residual fraction read;
- number of refined tiles;
- exact-fallback frequency;
- center/top-1 match before refinement;
- certified/final top-1 match;
- CPU and GPU wall-clock;
- peak host and device memory.

## Block execution metrics

- target passes per generated token;
- mean, p50, p95, and maximum committed block;
- accepted tokens per model-weight stream;
- repair frequency and repair scope;
- KV traffic per token;
- total storage traffic per token.

## Quality suite

The final evaluation must include at least:

- Korean and English conversation;
- C# and Python code generation;
- mathematical reasoning;
- long-form planning;
- JSON/schema-constrained generation;
- tool-call formatting;
- factual QA;
- translation;
- long-context retrieval and continuation.

Record exact-match/divergence, benchmark scores where licenses permit, and human-reviewed failure cases.

## Hardware gate

Flagship comparison uses the same machine, same GPU, same prompt set, same context length, and same generation settings.

Required target:

```text
GPU VRAM peak: <= 8 GiB
VORTEX 405B p50 time/token: <= 1.2x native 4B Q4 p50 time/token
VORTEX 405B p95 time/token: <= 1.5x native 4B Q4 p95 time/token
```

Also record:

- GPU model and clocks;
- PCIe generation;
- host RAM capacity and bandwidth;
- storage device and measured sequential/random throughput;
- software versions;
- cold-start and warm-start separately.

## Completion rule

No document, README, issue, or release may state that the final target is achieved until:

1. a real 405B-class dense model runs end-to-end;
2. peak GPU memory stays within 8GiB;
3. the quality/correctness suite passes the declared mode;
4. the wall-clock gates above pass against a native 4B baseline;
5. commands, logs, metrics, and code needed to reproduce the result are committed or released.
