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

## Current exact batch-geometry boundary

For `K` rank-one bilinear questions, all left factors span at most `K`
dimensions and all right factors span at most `K` dimensions. The complete
batch therefore lies in one `K^2`-dimensional `R tensor U` envelope; for
`K=32`, the envelope has at most 1,024 slots.

This is a design interface, not a stored cache or a working decoder. Literal
catalogs of all registered envelopes are too large, and exact Segre counting
does not force enough probes to decide the runtime budget. The open task is an
implicit near-source-size encoding that produces only the requested envelope
restriction, with every address, native summary, probe, and decode cost
charged.

The envelope also has a strict causal-utility boundary. Computing all entries
of `R^T W U` for 32 left directions and 32 state columns yields 1,024 exact
measurements, but the model still has only 32 forward/KV states. The scalar
table may reduce repeated certification work; it may not be used as a
1,024-token denominator. A full near-Shannon BF16 sweep remains
`525.1467264 ms/token` over the valid state count.

## Current tabulation/supercode boundary

Arbitrary nonlinear stored bits do not create a new source when a query uses
fixed addresses and XOR-only decoding. The unique algebraic normal form of
each cell proves that every exact recovery set has an equivalent collection
of linear functional atoms. This closes the named tabulation route without
assuming that the original preprocessing was linear.

The aligned sparse dictionary itself remains open. Favorable local counting
first has enough subset names at `b=23`, `S=619`, `t=6`, but random atoms miss
the binary rank-one set exponentially and counting supplies neither atoms nor
a decoder. Adaptive word-RAM schemes and native numerical queries are outside
the theorem. No Core Candidate survives.

## Current sparse-cover Fourier boundary

The former `23 x 23` capacity witness is now closed. Exact Hamming-ball
Fourier constraints force rank-one dual character bias larger than arbitrary
spanning atoms can have in their pairwise second moment. The contradiction
allows arbitrary atom matrices and duplicates; it is not a random-code test.

The first nine shapes in exact subset-slack order fail. `13 x 89` is the
first unclosed rectangle in the deterministic `1..128` scan, but it is only a
parameter frontier. Any cover must contain kernel relations of weight at most
39 and still lacks atoms, a decomposer, native semantics, and joint causal
traffic. No Core Candidate survives.

## Current Segre-ruling preimage boundary

The `13 x 89` slack case is now closed without random search. Its
89-dimensional ruling needs `2^89` low-weight representatives, but its
285-dimensional preimage can contain at most `Ball(285,13)`, smaller by a
factor above 59,000. This follows from an exact information-set projection.

The first survivor of the combined restricted-preimage and Fourier scan is
`18 x 36`, `S=758`, `t=7`. Survival is not a candidate. Even one ruling
requires a `[146,110]` binary covering code of radius at most seven, and all
rulings must share the same kernel. No code, Segre alignment, decoder, native
lift, or physical batch path exists. No Core Candidate survives.

## Current determinantal rank-amplification boundary

Sparse rank-one coverage now carries an exact closure obligation: rank-at-
most-`r` matrices must fit inside the atom Hamming ball of radius `rt` for
every `r`. The prior `17 x 43` frontier fails already at rank two by a factor
above 64. A mixed-weight sampling theorem rejects four balanced cases that
survive raw determinantal counting.

Together the Gates close the first 770 subset-capacity-ordered rectangles.
`30 x 40`, `S=1,404`, `t=14` is the first unclosed parameter point, but no
atoms, decoder, native semantics, or shared physical batch exists. No Core
Candidate survives.
