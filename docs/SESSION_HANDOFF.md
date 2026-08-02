# Session handoff

Last updated: 2026-08-02 (Asia/Seoul)

## Mandatory first read

Read `docs/PROOF_FIRST_CONTRACT.md`, `AGENTS.md`, and `docs/ARCHITECTURE_GATE0_CASCADE_CAPSULE.md` before changing the architecture.

The fixed target remains an unmodified 405B-class Hugging Face dense model under 8 GiB VRAM with original-model quality and same-machine 4B-class wall-clock behavior. No E0/E1 result may be promoted into a feasibility claim.

## Main baseline

Architecture Gate 0 foundation was merged through PR #6.

```text
merge commit: be46308bc662c9a193bfc912840a795f9eb9c998
```

The merge passed GitHub Actions on Python 3.10 and 3.12. Both jobs completed installation, the full pytest suite, and `scripts/run_validation.py` successfully.

## Current evidence

Existing E1 primitives:

1. disk-backed progressive LM-head certification;
2. tiny-model streamed Llama execution;
3. tiny-model Jacobi equivalence;
4. exact-on-span `OnlineAtlasLinear` replay;
5. tiny-Llama O/down replay with persisted atlas state.

Gate 0 adds E0/E1 infrastructure:

- `vortex_runtime/gate0_budget.py`: 405B memory, traffic, and compute equations;
- `scripts/run_gate0_budget.py`: machine-readable certificate generation;
- `gate0_budget.json`: committed conditional candidate;
- `vortex_runtime/gated_projected_linear.py`: projected fast path plus exact CPU cold path;
- source-module storage release after replacement so stale references do not retain GPU weights;
- `scripts/run_gate0_falsification.py`: disjoint-prompt, real-operation replacement harness;
- fixed Korean/English/code/math/JSON build and evaluation prompt sets;
- budget and operator tests included in green CI.

No pretrained 1B–3B falsification result has been committed. Evidence remains E0 architecture / E1 calculator and operator semantics.

## Candidate

Name: **Cascade Capsule v0**

```text
warm activation
  -> domain basis coordinates
  -> low-bit projected operator images
  -> bounded active-token attention
  -> progressive/projected LM head

basis or token-set miss
  -> exact original layer/tile repair
  -> measured bytes and compute
  -> optional capsule expansion
```

Per-layer activation domains:

1. pre-attention input shared by Q/K/V;
2. attention output input to O;
3. pre-MLP input shared by gate/up;
4. MLP product input to down.

Current assumptions:

- ranks 64 / 48 / 64 / 48 and LM-head rank 64;
- basis 8-bit, projected image 3-bit;
- active attention positions 256;
- compact KV 2-bit;
- cold original model 4-bit equivalent;
- target cold-stream amortization `A=512`.

## Conditional budget

```text
M_hot    = 1.365 GiB
M_kv     = 0.015 GiB
M_work   = 1.750 GiB
M_repair = 0.750 GiB
M_total  = 3.881 GiB <= 8 GiB

B_total(A=512) = 1.650 GiB/token <= 2.400 proxy limit
C_total(A=512) = 7.898 GFLOP/token <= 9.600 proxy limit
```

Decisive threshold:

```text
A >= 246.889 tokens per full-model-equivalent repair
full-model-equivalent repairs/token <= 0.0040504
```

Status: **conditional_pass**, not a passed Architecture Gate 0. The 4B values are unmeasured proxies; low-bit kernels and bounded active-token attention are unimplemented; real 8 GiB peak VRAM is unmeasured.

## Exact next task

Use hardware with enough RAM and preferably an NVIDIA GPU. Install the reproducible experiment environment:

```bash
python -m pip install -e '.[experiments]'
```

Run a pretrained 1B–3B Llama-family checkpoint:

```bash
python scripts/run_gate0_budget.py --output gate0_budget.json
python scripts/run_gate0_falsification.py \
  <model-id-or-local-path> \
  --build-prompts experiments/gate0_build_prompts.json \
  --eval-prompts experiments/gate0_eval_prompts.json \
  --gate0-budget gate0_budget.json \
  --rank 64 \
  --epsilon 0.05 \
  --max-new-tokens 64 \
  --minimum-token-agreement 1.0 \
  --output gate0_falsification.json
```

The harness uses hooks only to collect build activations. Evaluation replaces actual Q/K/V/O and gate/up/down modules with `GatedProjectedLinear`, retains exact matrices on the CPU cold path, and records cold-path invocations.

## Decision rule

Promote the hidden-axis candidate only when all are true:

- disjoint build/evaluation prompts;
- real-operation replacement;
- declared token agreement;
- observed `A >= 246.889`;
- bounded rank and capsule growth;
- reproducible raw JSON metrics.

On failure, record it, identify the projection groups dominating repair traffic, revise the mechanism, and rerun on a new held-out split. Do not tune indefinitely on the same prompts.

Do not begin the full CUDA backend or universal graph lowering until this repair-rate gate survives. Attention compression remains a separate gate.

## Communication rule

Correct:

> E0/E1: Cascade Capsule v0 closes the symbolic 405B budget only under explicit proxy and amortization assumptions. A real-model operation-replacement test is executable; no pretrained result has passed.

Forbidden:

> The 405B target is feasible or solved.
