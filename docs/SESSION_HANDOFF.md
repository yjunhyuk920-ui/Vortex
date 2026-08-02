# Session handoff

Last updated: 2026-08-02 (Asia/Seoul)

## Fixed objective

Build a universal Hugging Face runtime that can execute an unmodified 405B-class dense model with:

- peak GPU VRAM at or below 8 GiB;
- no user training, distillation, fine-tuning, or manual model-specific conversion;
- declared original-model quality preserved;
- p50 warm-decode time/token at or below 1.2x a native 4B Q4 baseline on the same machine;
- one-command operation.

Do not replace this target with “the model fits,” a smaller model, or a slow streamed baseline.

## Mandatory first read

1. `docs/PROOF_FIRST_CONTRACT.md`
2. `AGENTS.md`
3. `docs/ARCHITECTURE_GATE0_CERTIFICATE.md`
4. `architecture_gate0_budget.json`
5. `docs/EXPERIMENT_006_BLOCK_COMBINED_GATE.md`
6. latest committed files under `results/`

## Current evidence level

Current evidence is **E0/E1**.

Working primitives include:

- safetensors discovery and slicing;
- bounded tensor caching;
- streamed tiny-Llama execution;
- progressive LM-head certification;
- exact Jacobi comparison;
- `OnlineAtlasLinear`;
- real TinyLlama `nn.Linear` replacement;
- multiple optimistic repair oracles;
- machine-readable Gate 0 calculations.

None of these proves the 405B target.

## Critical accounting correction completed

The Gate 0 generator now uses:

```text
traffic/token = B_hot + rho * B_cold / A
compute/token = C_hot + rho * C_cold
```

`A` is committed block tokens and `rho` is the selected exact repair fraction.

Only storage traffic is divided by `A`. Exact arithmetic is performed for every token and is not divided by block length.

The previous compute expression based on `C_cold / E` was removed from the active generator and tests.

## Current Gate 0 result

```text
memory estimate:                  4.30497 GiB
memory limit:                     8.00000 GiB

hot traffic:                      1.29249 GiB/token
traffic limit:                    2.83517 GiB/token
required traffic efficiency E: 491.29916

hot compute:                      3.53152 GFLOP/token
compute limit:                   12.13274 GFLOP/token
maximum exact repair fraction:    1.01727%
```

The original candidate assumed:

```text
repair fraction rho: 25%
committed block A:   160
```

Its traffic fits, but its corrected compute is:

```text
214.91185 GFLOP/token
```

Decision:

```text
REJECT the original VORTEX-WAVE-1 design point.
```

## Strongest observed repair result

TinyLlama 1.1B exact-target adjoint tile oracle:

```text
observed E:                  8.195999
required E:                491.299160
observed repair fraction:   12.201075%
maximum repair fraction:     1.017268%
```

It fails both constraints:

- traffic shortfall: about 59.94x;
- compute excess: about 11.99x.

## Rejected steady-state mechanisms

- exact-span Atlas warm decode;
- rank-32 exact layer-suffix repair;
- rank-32 output-row repair;
- residual-energy-ranked 2D tile repair;
- exact-target adjoint 2D tile repair per token;
- the 25%-repair VORTEX-WAVE-1 design point.

Do not add more features to these paths as though they remain the core solution.

## Active experiment

The only active continuation of this family is the **block-shared combined traffic/compute oracle** in:

```text
scripts/run_oracle_block_shared_gate.py
.github/workflows/block-shared-combined-gate.yml
docs/EXPERIMENT_006_BLOCK_COMBINED_GATE.md
```

A candidate advances only if the same exact subset:

```text
rho <= 0.01017268
A / rho >= 491.29916
committed exact causal prefix > 0
traffic pass = true
compute pass = true
```

## Exact next actions

1. Open a draft PR from `feat/architecture-gate0-certificate` to `main` so the pull-request workflows run.
2. Collect the `tinyllama-block-shared-combined-gate` artifact.
3. Commit its raw JSON and SHA-256 digest.
4. Feed the result into `architecture_gate0_budget.json`.
5. If no subset inside the combined budget commits a non-zero exact prefix, reject the rank-32 capsule/block-repair family.
6. Begin a new architecture only after writing its full 405B memory, traffic, and compute equations first.

## Files changed by the accounting correction

- `vortex_runtime/feasibility.py`
- `vortex_runtime/gate0_observations.py`
- `scripts/run_architecture_gate0.py`
- `tests/test_feasibility.py`
- `architecture_gate0_budget.json`
- `README.md`
- `docs/ARCHITECTURE_GATE0_CERTIFICATE.md`
- `docs/SESSION_HANDOFF.md`

## Validation commands

```bash
python -m pytest -q
python scripts/run_validation.py
python scripts/run_architecture_gate0.py
```

`tests/test_feasibility.py` explicitly verifies that increasing block length lowers storage traffic but does not lower selected exact arithmetic.

## Communication rule

Current status must be described as:

> E0/E1: the original VORTEX-WAVE-1 design point is rejected by corrected compute accounting. The strongest optimistic per-token repair also fails both traffic and compute gates. The block-shared combined oracle is the final active falsification test for this repair family.

Do not claim that 405B-at-4B-speed is feasible until the evidence reaches E4.
