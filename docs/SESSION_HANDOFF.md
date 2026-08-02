# Session handoff

Last updated: 2026-08-02 (Asia/Seoul)

## Mandatory first read

Read:

- `docs/PROOF_FIRST_CONTRACT.md`
- `AGENTS.md`
- `docs/ARCHITECTURE_GATE0_CERTIFICATE.md`
- `architecture_gate0_budget.json`

The project no longer promotes local prototype behavior into a full-target claim.

## Current verified evidence level

The repository has E1 primitives:

1. disk-backed progressive LM-head certification;
2. tiny-model streamed Llama execution;
3. tiny-model Jacobi sequence equivalence;
4. exact-on-span `OnlineAtlasLinear` replay;
5. tiny-Llama O/down projection replay with persisted atlas state.

Validation commands:

```bash
python -m pytest -q
python scripts/run_validation.py
python scripts/run_architecture_gate0.py
```

## Architecture Gate 0 analytic candidate

The first complete candidate is `VORTEX-WAVE-1`.

Its model-wide path contains:

- rank-32 INT8 session capsules for every attention/MLP projection and LM head;
- a row-level embedding cache;
- rank-64 INT8 summaries for old attention context;
- an exact BF16 recent KV window;
- a weight-stationary multi-position proposal block;
- selective exact BF16 weight/KV repair;
- final token certification and full exact fallback.

The correctness path is proposed but not implemented. The certificate is an analytic envelope, not an E2 result.

## Populated gate result

Generated result:

```text
status: blocked-mechanism-unproven
memory total: 4.30497 GiB / 8 GiB
hot traffic: 1.29249 GiB/token
projected traffic at design threshold: 2.47674 GiB/token
traffic gate: 2.83517 GiB/token
hot compute: 3.53152 GFLOP/token
projected compute at design threshold: 4.85264 GFLOP/token
compute gate: 12.13274 GFLOP/token
```

The controlling quantity is repair efficiency:

```text
E = committed tokens A / full-model-equivalent repair fraction rho
required E: 491.29916
candidate design target: 640
current observed E: 1.27518
shortfall: 385.2786x
```

The design target uses `A=160`, `rho=0.25`. Those are threshold values, not observations.

## Exact next task

Implement the smallest real-operation falsification harness for `VORTEX-WAVE-1`.

The harness must replace actual operations during generation and record, per repair batch:

```text
proposed positions
committed causal-prefix tokens A
exact weight bytes read
exact KV bytes read
rho
E = A / rho
hot bytes/token
peak device memory
wall-clock/token
exact-token agreement
```

Requirements:

1. real pretrained 1B–3B causal model;
2. disjoint build and evaluation prompts;
3. actual operation replacement, not hooks alone;
4. exact committed-token agreement;
5. machine-readable raw output;
6. reject the candidate if measured `E < 300` after planned repair sweeps;
7. promote toward E2 only if measured `E >= 600` with the other resource gates intact.

## Files added for Gate 0

- `vortex_runtime/feasibility.py`
- `scripts/run_architecture_gate0.py`
- `tests/test_feasibility.py`
- `tests/test_gate0_budget_file.py`
- `architecture_gate0_budget.json`
- `docs/ARCHITECTURE_GATE0_CERTIFICATE.md`

## Communication rule

Current status must be described as:

> E0/E1: VORTEX-WAVE-1 has a model-wide analytic envelope whose memory, traffic, and compute equations close only at a repair-efficiency threshold that is not yet measured. The observed mechanism is currently 385x below that threshold.

Do not describe the target as feasible until the required evidence gates pass.
