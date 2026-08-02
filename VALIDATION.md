# Validation status

Date: 2026-08-02

## Reproduction

```bash
python -m pytest -q
python scripts/run_validation.py
```

Observed before repository upload:

```text
7 tests passed
validation completed successfully
```

Measured results are written to [`validation_results.json`](validation_results.json).

## Executable properties currently validated

1. Hugging Face `config.json` and safetensors index discovery without loading a complete model.
2. Individual tensor and tensor-slice retrieval.
3. Layer-streamed tiny-Llama execution under a byte-counted tensor cache.
4. Embedding row slicing instead of loading the complete embedding matrix.
5. Automatic conversion of the LM head into low-bit base plus lossless disk residual blocks.
6. Exact greedy argmax certification: a token is committed only when unread residual bounds cannot change the result.
7. Exact causal Jacobi decoding: committed token sequences are compared with sequential greedy generation.

## Recorded prototype metrics

### In-memory progressive LM head

| Base bits | Final exact certification | Coarse top-1 match | Mean residual fraction |
|---:|---:|---:|---:|
| 4 | 100% | 87.5% | 50.44% |
| 5 | 100% | 100% | 3.28% |
| 6 | 100% | 100% | 0.57% |

### Disk-backed progressive LM head

- Final exact certification: 100%
- Coarse top-1 match: 96.875%
- Mean residual fraction: approximately 5.39%

### Exact Jacobi generation

- Sequence match against sequential greedy: 100%
- Mean target passes per token: 1.65
- Mean maximum committed block: 3.5
- Mean committed block: approximately 1.275

These figures are from deterministic synthetic/tiny-checkpoint validation. They are not 405B performance claims.

## Remaining critical implementation

Progressive, decision-directed execution is still required for internal Q/K/V/O and gate/up/down projections. Until target weight traffic and compute are reduced or amortized at those layers, the runtime does not meet the final 405B-at-4B wall-clock objective.

See [`docs/VALIDATION_PROTOCOL.md`](docs/VALIDATION_PROTOCOL.md) for the fixed final gates.
