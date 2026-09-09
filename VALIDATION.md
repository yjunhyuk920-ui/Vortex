# Validation status

## Native residual lift validation

[Actual evidence](experiments/codex_native_residual_lift_20260909/REPORT.md):
106 ordered FP32 cases,2,097,280 first integer pairs/16,768 residual pairs,
nine domain refusals and24 small CPU torch linear/mv calls. The native output
words and CPU RNG state match their declared checks. Incomplete source defects
were actually reproduced and corrected. No full HF generation/KV/405B/CUDA/
8GiB/latency measurement or whole-repository test run occurred.

## EXP-100A current CPU validation

[Actual validation](experiments/codex_fmm_integrity_20260909/VALIDATION.md):
full red, first-corrected and hardened gates,50 cells/132 controls,11 existing functions plus6
unittest regressions; exact old-field comparison and independent workspace sum.
This was not pytest or a repository-wide suite. Windows RSS unavailable/null.
No native405B/GPU/baseline/TTFT run; O1-O5 OPEN/O6 PARTIAL.

## Restricted BF16 producer extension — 2026-09-09

Current focused evidence is [experiments/codex_native_sparse_extension_20260909/VALIDATION.json](experiments/codex_native_sparse_extension_20260909/VALIDATION.json).
The scalar agent ran 1,566,724 frozen controls and five refusals; the primary
ran an independent 124,423-case integer oracle and the actual final packaged
CPU HF comparison. All 40 incremental logits/serialized cache states and eight
sampled generation logits/probabilities/tokens/RNG states match. Full-prefill
logits/KV values also match. HF v2's first-stride failure remains preserved.
Scalar historical v1 source bytes were not retained; do not claim a v1 red run.

Scope is Phase B native synthetic control and a declared scalar theorem.
No full repository suite, real public checkpoint, CUDA, 405B, <=8-GiB peak,
native4BQ4 latency ratio, TTFT or GitHub Actions ran this round. O1-O5 OPEN,
O6 PARTIAL; THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED.

## Latest bounded record — 2026-09-08

experiments/native_global_transition_20260908/results/validation.json records
14unit checks and a3510science-file fresh real-model replay. Scope: guarded pinned
HF CPU logits/KV/layout/RNG, not the full repository suite or target hardware.
The historical commands and observations below are preserved, not claimed rerun.


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
