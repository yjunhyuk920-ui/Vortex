# VORTEX Research State

Last updated: 2026-08-03 Asia/Seoul

## Fixed final objective

Execute an arbitrary publicly released, unmodified Hugging Face dense Transformer by replacing only the runtime:

- real 405B-class dense flagship;
- peak GPU VRAM <=8 GiB;
- no retraining, distillation, fine-tuning, LoRA, or user-authored model-specific adapter;
- original declared ability/output contract preserved;
- p50 warm time/token <=1.2x native 4B Q4 on the same target machine;
- independently reproducible evidence.

The objective is unchanged.

## Current environment

MEASURED capability: GitHub repository, GitHub Actions CPU, Python, and small downloadable checkpoints.

Unavailable and NOT TESTED: target 8 GiB GPU, 405B checkpoint execution, CUDA, PCIe, SSD, target power, TTFT, tokens/second, and target peak VRAM.

## Validation system

- Phase A theory/structure;
- Phase B synthetic/reference;
- Phase C small-real-model falsification or real-operation replacement;
- Phase D actual target hardware, currently NOT TESTED.

Evidence E0–E7 and MEASURED/DERIVED/PROJECTED/UNVERIFIED remain mandatory.

## Component classification

Auxiliary accepted:

- exact/checksummed mmap pointer VM;
- bounded TinyLlama compiler in its finite tested grammar;
- exact future-suffix DAG as body compression only;
- CPTC causal certificate/fault rejection/exact fallback at E1;
- EXP-048 exact longest-prefix-plus-correction block verifier at E1.

Rejected as core:

- raw prefix/future routing for unseen prompts;
- static compression, deterministic residual, recurrent program, repair, and related families in `FAILED_APPROACHES.md`;
- global/oracle-tight/stratified range-based CPTC;
- hard Jacobi block decoding under the tested accounting;
- training-free partial-layer self-draft using the same target LM head.

## EXP-047/047R closed evidence

EXP-047 established certificate correctness but evaluated almost all tiles. EXP-047R used the exact realized per-state range oracle and still evaluated 100% at median and p90 across 18 states from three pinned trained checkpoints.

Decision:

```text
REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY
```

Authoritative summaries:

```text
results/exp_047/summary.json — workflow 30793232558
results/exp_047r/summary.json — workflow 30795946233
```

## EXP-048 authoritative evidence

```text
results/exp_048/summary.json
workflow 30798936320
source head SHA 484a1f0f313d88733d2f7210f2a24d3904bf1373
workflow merge SHA d60e392d66d694fc020f2cfe2435e47e5f5a22ca
artifact 8850040445
artifact SHA-256 67c587da36b968f9c38e0a7774ea03cecd2ad2d7d274d3e83c833c56529c3443
phase A/B/C-observation
evidence E1
```

Pinned checkpoints:

```text
TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

MEASURED correctness and causality:

```text
9 EXP-048 tests passed
repository validation passed
3 models × 6 held-out families = 18 cases
B1 exact mismatches 0
B2 exact mismatches 0
B3 exact mismatches 0
B3 future information uses 0
```

MEASURED condition results:

```text
B1 perfect future oracle:
  96 exact tokens / 1 target pass
  target-equivalent fraction 1.0416667%
  future information true
  deployable false

B2 hard Jacobi:
  p50 58 target passes / 32 exact tokens
  p50 fraction 181.25%
  p90 fraction 193.75%
  maximum matching prefix 3

B3 partial-layer self-draft:
  54 fixed variants
  best cases with any matching proposal token 4/18
  maximum matching prefix 1
  p50 committed tokens / target verification 1
  model medians 1 / 1 / 1
  minimum fully accounted fraction 1333.463%
  p90 fully accounted fraction 2893.843%
```

PROJECTED:

```text
405B Q4 full stream 188.592821 GiB
1.2x 4B Q4 allowance 2.235174 GiB/token
required target-equivalent fraction 1.185185%
zero-cost perfect-proposal minimum 85 tokens/full target pass
B1 oracle fraction / required 0.87890625
B3 p90 fraction / required 2441.6793
```

## Scientific decision

```text
EXP-048 exact block verifier: ACCEPT E1 AUXILIARY
B1 perfect proposal: ACCEPT AS NON-DEPLOYABLE UPPER BOUND ONLY
B2 hard Jacobi core: REJECT
B3 partial-layer self-draft core: REJECT
B4 tree continuation from failed B3: DO NOT IMPLEMENT
complete real operation replacement: NOT TESTED
Phase D: NOT TESTED
E6/E7: not achieved
```

Required wording:

> EXP-048, E1: the exact block verifier safely preserved greedy output, and a future-aware 96-token oracle reached a logical 1.0417% target-stream fraction. The deployable early-layer draft matched at most one proposal token and had p50 one committed token with p90 28.9384 target-equivalent streams per token, so partial-layer self-drafting is rejected as the core runtime. No 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, or tokens/second claim was tested.

## Primary unresolved bottleneck

The exact verifier is not the bottleneck. The runtime needs a causal, training-free source of at least roughly 85 accurate future tokens without paying one sequential LM-head/target pass per proposed token.

B1 proves that sufficiently accurate long proposals would satisfy the logical traffic arithmetic. B2/B3 prove that hard Jacobi and early-layer sequential drafting do not supply them.

## Current frontier

`EXP-049 — Anderson-Accelerated Continuous Block Fixed-Point Gate`, defined in `NEXT_EXPERIMENT.md`.

EXP-049 removes the separate per-token draft loop. It represents a large future block as soft token embeddings, applies a small fixed number of full batched target passes, uses damped/Anderson updates, hardens a proposal, and verifies it with the retained exact block verifier. Hard Jacobi is the baseline.

The candidate has a major theoretical risk: the causal triangular dependency may limit reliable information propagation to approximately one new exact position per solver round. The experiment must include both an explicit dependency lower-bound audit and real small-checkpoint counterexamples. Failure retires target-only fixed-point proposal generation as a core family.

## Reproduction

```bash
git checkout research/exp-048-causal-block-amortization
python -m pytest -q tests/exp_048
python scripts/run_validation.py
bash experiments/exp_048/reproduce.sh
```

Frozen authoritative evidence is under `results/exp_048/`; reproduction must write to an isolated directory.

## Next-session reading

1. `AGENTS.md`
2. this file
3. `FAILED_APPROACHES.md`
4. `DECISION_LOG.md`
5. `ASSUMPTION_REGISTER.md`
6. `VALIDATION_MATRIX.md`
7. `NEXT_EXPERIMENT.md`
8. EXP-048 document and frozen summary
9. PR #58
