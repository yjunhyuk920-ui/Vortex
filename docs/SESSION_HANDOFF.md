# Session handoff

Last updated: 2026-08-02 (Asia/Seoul)

## Mandatory first read

Read:

- `docs/PROOF_FIRST_CONTRACT.md`
- `AGENTS.md`
- `docs/ARCHITECTURE_GATE0_CERTIFICATE.md`
- `docs/REAL_OPERATION_FALSIFICATION.md`
- `architecture_gate0_budget.json`

The project no longer promotes local prototype behavior into a full-target claim.

## Current verified evidence level

The repository has E1 primitives:

1. disk-backed progressive LM-head certification;
2. tiny-model streamed Llama execution;
3. tiny-model Jacobi sequence equivalence;
4. exact-on-span `OnlineAtlasLinear` replay;
5. tiny-Llama O/down projection replay with persisted atlas state;
6. a drop-in `nn.Linear` replacement wrapper with logical repair accounting.

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

## Real-operation falsification phase A

Implemented:

- `AtlasLinearModule`, a real `nn.Linear` replacement;
- model traversal and replacement by projection suffix;
- disjoint build/evaluation snapshots;
- exact output checks in tests;
- logical cold-weight byte accounting;
- managed and full-model repair fractions;
- `E=A/rho` calculation;
- optional Hugging Face runner.

Run on a local or downloadable pretrained model:

```bash
pip install transformers

python scripts/run_real_operation_falsification.py \
  --model <local-path-or-hf-repo> \
  --device cpu \
  --max-new-tokens 16 \
  --max-rank 256
```

The runner compares greedy token sequences before and after replacement, uses disjoint build and evaluation prompts, records rank growth and repair efficiency, and writes `real_operation_falsification.json`.

Important boundary: the Transformers model remains physically resident. Reported cold bytes are logical exact-weight uses, not measured NVMe/PCIe traffic. This phase can reject activation-span reuse, but cannot pass the final 8 GiB or wall-clock gate.

## Exact next task

1. Run phase A on a real pretrained 1B–3B Llama-family model.
2. Replace O/down first, then all seven projection families only when exact token agreement holds.
3. Record the generated JSON in the repository.
4. Reject `VORTEX-WAVE-1` if full-model-equivalent `E < 300`, rank saturates, or unseen prompts require near-full repair.
5. Continue only when `E >= 600` with bounded representation growth.
6. Then replace resident exact weights with safetensors-backed cold loaders so logical repair bytes become physical host/storage traffic.
7. Add block proposal and causal-prefix commit measurement; phase A currently measures generated tokens over cumulative repair rather than block-level `A`.

## Gate 0 files

- `vortex_runtime/feasibility.py`
- `vortex_runtime/falsification.py`
- `scripts/run_architecture_gate0.py`
- `scripts/run_real_operation_falsification.py`
- `tests/test_feasibility.py`
- `tests/test_gate0_budget_file.py`
- `tests/test_falsification.py`
- `architecture_gate0_budget.json`
- `docs/ARCHITECTURE_GATE0_CERTIFICATE.md`
- `docs/REAL_OPERATION_FALSIFICATION.md`

## Communication rule

Current status must be described as:

> E0/E1: VORTEX-WAVE-1 has a model-wide analytic envelope and a real-operation logical-repair falsification runner. The required repair efficiency remains unmeasured on a real 1B–3B model, and the currently observed tiny-model mechanism is 385x below the gate.

Do not describe the target as feasible until the required evidence gates pass.
