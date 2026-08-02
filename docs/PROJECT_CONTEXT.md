# Project context

## Why VORTEX exists

Conventional inference runtimes assume that most model weights remain in accelerator memory or can be streamed for every generated token. That fails the project objective: a 405B dense model is far larger than 8GB, and repeatedly moving hundreds of gigabytes per token cannot produce a 4B-class user experience.

VORTEX therefore treats model execution as a runtime research problem rather than only a quantization problem.

## Fixed product definition

Input:

- an existing Hugging Face repository or local model directory;
- no model retraining;
- no user-authored architecture adapter;
- no manual calibration procedure.

Output:

- automatic execution under an 8GB VRAM budget;
- original-model behavior/quality preserved by exact execution or certification and repair;
- a long-term target of native-4B-class wall-clock generation for a 405B dense model.

## Design history

Several conceptual directions led to the current executable prototype.

### ORB

The initial response-basis idea cached `W U` for the subspace visited by the current session. It reduced repeated weight access but retained large output dimensions and did not solve future trajectory branching.

### HyperFold

HyperFold proposed running the model entirely in reduced coordinates, including DEIM-style nonlinear reduction. This exposed an important direction—reduce executed state, not merely stored weights—but required assumptions about reachable-state rank that must be measured rather than presumed.

### ATLAS

ATLAS introduced branch-conditioned charts, repairable subspaces, and a cold exact oracle. It clarified that a runtime must distinguish fast-path execution from exact repair.

### ProofStream

ProofStream reframed execution as progressive decision proof: compute a low-cost center, maintain bounds for unread residual information, and refine only while the final token decision remains uncertain. This is the principle implemented first for the LM head.

## Current practical architecture

The repository starts from primitives that can be validated exactly:

1. discover Hugging Face tensor locations without constructing the full model;
2. enforce a byte budget around tensor residency;
3. generate a low-bit base and retain lossless residuals;
4. compute rigorous output bounds from unread residual tiles;
5. refine selected tiles until the exact argmax is certified;
6. compare block-parallel decoding against exact sequential output.

This is intentionally narrower than the final architecture. It creates a reliable measurement base before introducing CUDA kernels, asynchronous storage pipelines, reduced internal states, or model-wide sensitivity propagation.

## Core unresolved question

The project succeeds only if internal target execution can be skipped or amortized at an extreme rate without losing the original model's decision. The key empirical quantities are:

- how much residual weight traffic internal projections require;
- how errors amplify through attention, gated MLPs, normalization, and 126 layers;
- how many tokens can share each target weight stream;
- how frequently exact repair is required;
- whether the resulting wall-clock reaches the fixed acceptance target.

All future architecture proposals must produce executable experiments that measure these quantities.
