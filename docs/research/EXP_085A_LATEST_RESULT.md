# EXP-085A — Joint Exact SwiGLU Compiler Result

```text
Source head: eaf9b9d016e9f652a14fa8bae90c355adaeb9e02
PR merge-ref execution SHA: 1b773a3107f1815a54233fa5ae0fed24c5fd7f95
Workflow run: 32246784954
Artifact: 9362920446
Artifact SHA-256: d4361acd5f1e258117fc3886a2147c79c7b35251efa93ea5ad4140d5ec4cde5a
Decision: REJECT_PAGE_SEPARABLE_JOINT_EXACT_SWIGLU_REFINEMENT_AS_CORE
```

The official `HuggingFaceTB/SmolLM2-135M` BF16 checkpoint executed on 24 causal layer-0 MLP states. Both the impossible exact-contribution oracle and the sound metadata selector read all `192/192` fused intermediate macro-pages on every state and required fallback on every state.

Measured population result:

```text
oracle p50/p95 pages                    192 / 192
oracle whole-model-equivalent fraction 81.255038% / 81.255038%
sound p50/p95 pages                     192 / 192
sound operation fraction                94.797544%
fallback rate                           100%
false accepts                           0
candidate/reference mismatches          0
projected 405B metadata                 102.387497 GiB
```

This rejects only page-separable additive exact refinement. It does not reject a cross-page nonseparable function compiler or a whole-transition program. Complete layer replacement, 128-step state equality, CUDA, 405B, 8-GiB residency, and target latency remain `NOT TESTED`.
