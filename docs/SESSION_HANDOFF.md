# Session handoff

Last updated: 2026-08-02 (Asia/Seoul)

## Fixed objective

Build a universal Hugging Face runtime that executes an unmodified 405B-class dense model with:

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
5. `results/tinyllama_1_1b_block_shared_combined_gate.json`
6. `scripts/run_block_shared_residual_selector.py`

## Current evidence level

Current evidence is **E0/E1**. No E2 model-wide selector/certificate and no E4 405B completion exist.

## Correct accounting

```text
traffic/token = B_hot + rho * B_cold / A
compute/token = C_hot + rho * C_cold
```

Only storage traffic is divided by committed block tokens `A`. Exact arithmetic is charged every token.

Fixed envelope:

```text
memory estimate:                  4.30497 GiB / 8 GiB
hot traffic:                      1.29249 GiB/token
traffic limit:                    2.83517 GiB/token
required traffic efficiency E: 491.29916
hot compute:                      3.53152 GFLOP/token
compute limit:                   12.13274 GFLOP/token
maximum exact repair fraction:    1.01727%
```

## Rejected work

The following are not active steady-state solutions:

- exact-span Atlas warm decode;
- rank-32 exact layer-suffix repair;
- rank-32 output-row repair;
- residual-energy-ranked 2D repair per token;
- exact-target adjoint 2D repair per token;
- the original 25%-repair VORTEX-WAVE-1 point.

The original point projected 214.91 GFLOP/token and fails the compute gate.

## Block-shared combined oracle result

Draft PR: `#5 research: Gate 0 falsification and corrected compute accounting`

Workflow:

```text
run id:          30738817896
artifact id:     8830640577
artifact digest: sha256:26eb56a58911aec98e714d0433e5b078343acf54df2cd35b9f30fa33891e2832
raw JSON SHA:    df84dc0e1a5b343d36717b8e8fa6d4e692e1d32f04552e7f8d65f03d9afeabd6
```

Observed TinyLlama 1.1B result:

```text
zero-repair exact prefix:               1 token
selected tiles:                       128
selected exact bytes:                8 MiB
repair fraction:                   0.190642%
repaired exact prefix:                  2 tokens
incremental exact prefix:               1 token
traffic efficiency E:             1049.087891
projected traffic:                  2.014943 GiB/token
projected compute:                  5.143432 GFLOP/token
traffic gate:                       pass
compute gate:                       pass
```

Decision:

```text
The logical block-shared byte/compute mechanism survives E1 falsification.
```

This does not promote the architecture to E2. Exact target tokens and teacher gradients selected the 128 tiles. Only O/down projections and one evaluation prompt were covered. No sound commit certificate exists.

## Active engineering task

The next gate is a target-independent selector:

```text
scripts/run_block_shared_residual_selector.py
.github/workflows/block-shared-residual-selector.yml
```

Allowed selector inputs:

- approximate autoregressive activation residuals;
- precomputed weight-tile Frobenius norms.

Forbidden selector inputs:

- exact future target tokens;
- teacher-forced gradients;
- exact target logit margins.

Exact target output is used only after selection to measure the causal-prefix result.

## Advancement conditions

The selector advances only when all hold:

```text
incremental exact prefix > 0
rho <= 0.01017268
A / rho >= 491.29916
traffic pass = true
compute pass = true
```

A passing selector still requires:

1. repetition across Korean, English, code, mathematics, structured output, and long-form prompts;
2. multiple seeds and continuations;
3. a sound online causal-prefix certificate;
4. replacement of Q/K/V/O, gate/up/down, attention/KV, and LM head paths;
5. measured physical bytes, peak VRAM, and wall-clock;
6. scaling through 1B–3B, 7B–8B, 30B, 70B, and 405B.

## Files changed in the current session

- `vortex_runtime/feasibility.py`
- `vortex_runtime/gate0_observations.py`
- `vortex_runtime/gate0_corrected.py`
- `vortex_runtime/block_gate.py`
- `scripts/run_architecture_gate0.py`
- `scripts/run_oracle_block_shared_gate.py`
- `scripts/run_block_shared_residual_selector.py`
- `tests/test_feasibility.py`
- `tests/test_block_gate.py`
- `tests/test_gate0_corrected.py`
- `architecture_gate0_budget.json`
- `results/tinyllama_1_1b_block_shared_combined_gate.json`
- `.github/workflows/block-shared-residual-selector.yml`
- `README.md`
- `docs/ARCHITECTURE_GATE0_CERTIFICATE.md`
- `docs/SESSION_HANDOFF.md`

## Validation commands

```bash
python -m pytest -q
python scripts/run_validation.py
python scripts/run_architecture_gate0.py
```

## Communication rule

Current status must be stated as:

> E0/E1: the original VORTEX-WAVE-1 point and all per-token repair paths are rejected. An exact-target block-shared oracle found an 8 MiB repair set that extends the exact prefix from one to two tokens while passing analytic traffic and compute gates. A target-independent selector and sound commit certificate remain unproven.

Do not claim that 405B-at-4B-speed is feasible until the evidence reaches E4.
