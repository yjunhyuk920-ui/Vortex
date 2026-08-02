# Session handoff

Last updated: 2026-08-02 (Asia/Seoul)

## Mandatory first read

Read:

- `docs/PROOF_FIRST_CONTRACT.md`
- `AGENTS.md`
- `docs/ARCHITECTURE_GATE0_CERTIFICATE.md`
- `docs/EXPERIMENT_001_EXACT_SPAN_REJECTION.md`
- `architecture_gate0_budget.json`
- `results/tinyllama_1_1b_exact_span_warm_decode.json`

The project no longer promotes local prototype behavior into a full-target claim.

## Current evidence level

The repository contains E1 primitives and falsification infrastructure:

1. disk-backed progressive LM-head certification;
2. tiny-model streamed Llama execution;
3. tiny-model Jacobi sequence equivalence;
4. exact-on-span `OnlineAtlasLinear` replay;
5. real `nn.Linear` replacement with prefill/decode accounting;
6. approximate capsule, exact layer-suffix, and output-row tile repair modes;
7. model-wide Gate 0 memory/traffic/compute equations.

Validation commands:

```bash
python -m pytest -q
python scripts/run_validation.py
python scripts/run_architecture_gate0.py
```

## Architecture Gate 0 envelope

`VORTEX-WAVE-1` assumes:

- rank-32 INT8 session capsules for every attention/MLP projection and LM head;
- rank-64 summarized old-context attention;
- exact recent KV window;
- weight-stationary multi-position proposal;
- selective exact weight/KV repair;
- final token certification and full exact fallback.

Analytic envelope:

```text
memory total: 4.30497 GiB / 8 GiB
hot traffic: 1.29249 GiB/token
projected traffic at design threshold: 2.47674 GiB/token
traffic gate: 2.83517 GiB/token
projected compute at design threshold: 4.85264 GFLOP/token
compute gate: 12.13274 GFLOP/token
required repair efficiency E=A/rho: 491.29916
promotion threshold: 600
```

These equations close only when the repair mechanism reaches the stated efficiency.

## Experiment 001 result — exact-span path rejected

Real model:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
44 replaced O/down projections
rank limit 32
English build prompt
Korean disjoint evaluation prompt
8 generated tokens / 7 warm-decode steps
```

Measured:

```text
exact greedy token match: true
warm-decode fast vectors: 0 / 308
warm-decode exact fallback: every module on every token
rank growth during evaluation: 23 per module
observed E: 3.1790542140
required E: 491.2991599793
shortfall: 154.5426x
```

Decision:

```text
REJECT exact-span Atlas as the steady-state warm-decode mechanism.
```

Increasing exact-span rank is not the next action because model-wide capsule memory scales linearly with rank and breaks the 8 GiB envelope.

## Active experiments

### Experiment 002 — rank-32 approximate layer-suffix oracle

`scripts/run_oracle_suffix_repair.py`:

1. builds every managed projection to rank 32 using four disjoint task prompts;
2. runs the evaluation prompt using only projected capsule outputs;
3. restores exact O/down projections from the final layer backward;
4. finds the smallest exact layer suffix that restores the original greedy sequence;
5. reports the optimistic full-model-equivalent repair efficiency.

Even this oracle must reach `E >= 600` to advance.

### Experiment 003 — output-row tile oracle

`scripts/run_oracle_tile_repair.py`:

1. profiles exact-vs-projected error per 128-output-row tile on the evaluation input;
2. ranks tiles by error reduction per exact weight byte;
3. repairs the highest-value tiles during actual generation;
4. exhaustively tests the top-prefix choices inside the `E >= 300` rejection budget;
5. continues coarse searches to determine the first repair size that restores the sequence.

For the TinyLlama test model, the Gate 0 byte budget corresponds to roughly:

```text
full model bytes / 491.29916
```

exact weight bytes per generated token. If the optimistic row-tile oracle cannot restore output inside that budget, the current rank-32 capsule and repair family is rejected before building physical streaming kernels.

## Exact next actions

1. Finish CI run for the rank-32 suffix and row-tile oracles.
2. Commit both raw JSON results and artifact digests.
3. Apply results to `architecture_gate0_budget.json`.
4. Reject the current repair family when oracle `E < 300`.
5. Advance only a repair granularity that reaches `E >= 600` on disjoint real-model generation.
6. Physical safetensors/NVMe streaming starts only after the logical oracle survives.

## Current files

- `vortex_runtime/feasibility.py`
- `vortex_runtime/falsification.py`
- `vortex_runtime/gate0_observations.py`
- `scripts/run_architecture_gate0.py`
- `scripts/run_real_operation_falsification.py`
- `scripts/run_oracle_suffix_repair.py`
- `scripts/run_oracle_tile_repair.py`
- `tests/test_feasibility.py`
- `tests/test_gate0_budget_file.py`
- `tests/test_falsification.py`
- `architecture_gate0_budget.json`
- `docs/ARCHITECTURE_GATE0_CERTIFICATE.md`
- `docs/EXPERIMENT_001_EXACT_SPAN_REJECTION.md`
- `results/tinyllama_1_1b_exact_span_warm_decode.json`

## Communication rule

Current status must be described as:

> E0/E1: the model-wide analytic envelope closes only at `E >= 491.3`. A real 1.1B disjoint-prompt experiment measured exact-span `E=3.179`, so that steady-state mechanism is rejected. Rank-32 approximate layer and row-tile oracle repairs are the active falsification gates.

Do not describe the flagship target as feasible until the required evidence gates pass.
