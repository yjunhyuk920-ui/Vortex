# Session handoff

Last updated: 2026-08-02 (Asia/Seoul)

## Mandatory first read

Read `docs/PROOF_FIRST_CONTRACT.md`, `AGENTS.md`, and `docs/ARCHITECTURE_GATE0_CASCADE_CAPSULE.md` before changing the architecture.

The target remains an unmodified 405B-class Hugging Face dense model under 8 GiB VRAM with original-model quality and same-machine 4B-class wall-clock behavior. No E0/E1 result may be promoted into a final feasibility claim.

## Active branch

```text
research/architecture-gate-zero
```

This branch contains the first complete model-wide candidate budget and its executable falsification harness.

## Current verified evidence

The previous repository state remains E1:

1. disk-backed progressive LM-head certification;
2. tiny-model streamed Llama execution;
3. tiny-model Jacobi sequence equivalence;
4. exact-on-span `OnlineAtlasLinear` replay;
5. tiny-Llama O/down projection replay with persisted atlas state.

New branch work is also limited to E0/E1 until a real pretrained model run is committed:

- `vortex_runtime/gate0_budget.py` calculates the 405B memory, traffic, and compute equations;
- `scripts/run_gate0_budget.py` emits a machine-readable certificate;
- `gate0_budget.json` records the current conditional candidate;
- `vortex_runtime/gated_projected_linear.py` replaces a real `Linear` operation with projected and exact cold paths;
- `scripts/run_gate0_falsification.py` builds bases on one prompt set and evaluates replacement generation on a disjoint set;
- `experiments/gate0_build_prompts.json` and `experiments/gate0_eval_prompts.json` define the split;
- new isolated tests passed locally: `8 passed`.

The full repository test and validation workflow must still pass through the pull request CI before merge.

## Candidate architecture

Name: **Cascade Capsule v0**

Normal warm path:

```text
activation
  -> domain basis coordinates
  -> low-bit projected operator images
  -> bounded active-token attention
  -> progressive/projected LM head
```

Cold path:

```text
basis or token-set miss
  -> exact original layer/tile repair
  -> account bytes and compute
  -> optional online capsule expansion
```

Budgeted activation domains per layer:

1. pre-attention input shared by Q/K/V;
2. attention output input to O;
3. pre-MLP input shared by gate/up;
4. MLP product input to down.

Budget assumptions:

- ranks: 64 / 48 / 64 / 48, LM-head rank 64;
- basis: 8 bits;
- projected image: 3 bits;
- active attention positions: 256;
- compact KV: 2 bits;
- cold original model: 4-bit equivalent;
- target cold-stream amortization: `A=512`.

## Conditional Gate 0 result

The committed proxy calculation produces:

```text
M_hot    = 1.365 GiB
M_kv     = 0.015 GiB
M_work   = 1.750 GiB
M_repair = 0.750 GiB
M_total  = 3.881 GiB <= 8 GiB

B_hot = 1.281 GiB/token
B_cold = 188.988 GiB/full stream
B_total(A=512) = 1.650 GiB/token <= 2.400 proxy limit

C_hot = 6.312 GFLOP/token
C_repair = 811.698 GFLOP/full stream
C_total(A=512) = 7.898 GFLOP/token <= 9.600 proxy limit
```

The stricter falsification threshold is:

```text
A >= 246.889 tokens per full-model-equivalent repair
full-model-equivalent repairs/token <= 0.0040504
```

The status is **conditional_pass**, not a passed Architecture Gate 0. The 4B comparison values are unmeasured proxies, the low-bit kernels do not exist, and active-token attention is not implemented.

## Exact next task

First, merge only after pull request CI passes.

Then run the real-operation harness on a pretrained 1B–3B Llama-family model:

```bash
pip install transformers
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

The harness uses hooks only to collect build activations. During evaluation it replaces actual Q/K/V/O and gate/up/down modules with `GatedProjectedLinear`, moves exact matrices to CPU, and records actual cold-path invocations.

## Decision rule

Promote the hidden-axis candidate only if the real pretrained result has:

- disjoint build/evaluation prompts;
- real-operation replacement;
- declared token agreement;
- observed `A >= 246.889`;
- bounded rank/capsule growth;
- reproducible raw JSON metrics.

If it fails, do not tune indefinitely on the same prompts. Record the failure, identify which weighted projection groups dominate repair traffic, revise the architecture, and rerun on a new held-out split.

Do not begin CUDA kernel implementation or universal graph lowering until this repair-rate gate survives. Attention compression remains a separate required gate even if hidden-axis projection succeeds.

## Communication rule

Correct current statement:

> E0/E1: Cascade Capsule v0 closes the symbolic 405B budget only under explicit proxy and amortization assumptions. A real-model operation-replacement test is now executable; no pretrained result has passed yet.

Forbidden statement:

> The 405B target is now feasible or solved.
