# VORTEX — latest-source audit and constructive native-observation source

Date: 2026-09-05 (Asia/Seoul). Base: `d4ae2ce79de49b90c1fc0188f212ba7d8b92550c`.
**No complete 405B / one total-8-GiB / native-4B-latency theory or implementation is delivered.**
This is a literature-driven discovery round and a bounded constructive reference, not an admitted high-speed core. The CTC-2026-09-05 mission and O1–O6 are unchanged.

## 1. What the new sources actually establish

The sources below were read through primary abstracts/full text, not accepted from headlines. Paper-reported measurements were not reproduced locally. September coverage includes a 2 September preprint; this is not a claim to have exhaustively searched all literature. The observation dates, formats, targets and baselines must remain attached to the numbers.

| Primary source/version | Actual relevant result | VORTEX interpretation |
|---|---|---|
| [DFloat11, v3, 1 Jan 2026](https://arxiv.org/html/2504.11651v3) | Approximately 30% weight-size reduction, bit-preserving reconstruction; 405B demonstrated on **8 × 80 GB** GPUs. Reported 2.3–46.2× throughput is against CPU offloading of uncompressed models. | Useful lossless storage technology, not 405B on an 8-GiB GPU at a native-4B baseline. |
| [ZipServ, 18 Mar 2026](https://arxiv.org/html/2603.17435v1) | Fused compressed loads to Tensor Core registers; up to 30% size reduction, 2.21× kernel speedup, average 1.22× end-to-end against vLLM. | Eliminates decode buffers/traffic, not most dense arithmetic. Do not multiply unrelated kernel/headline speedups. |
| [RSR-core, 29 Mar 2026](https://arxiv.org/html/2603.27462v1) | Binary/ternary matrices; logarithmic pattern-sharing arithmetic reduction. Up to 62× CPU versus BF16 PyTorch on ternary models; up to 1.9× CUDA token generation. | The low-cardinality weights are an essential premise. Does not provide a bit-exact native lift for arbitrary BF16 checkpoints. |
| [Accurate Models of NVIDIA Tensor Cores, v4, 11 Jun 2026](https://arxiv.org/html/2512.07004v4) | Architecture-dependent alignment, normalization, truncation, extra bits, and unnormalized products; software models validated against hardware samples. | Concrete numerical semantics to investigate; sampled model validation is not a proof for all GPU inputs or all library kernels. |
| [Unweight, technical report, 17 Apr 2026](https://research.cloudflare.com/papers/unweight-2026.pdf) | BF16 exponent coding and fused decompression on Hopper, about 30% MLP / 20% model compression in the reported Llama-3.1-8B setting. | Lossless weight movement aid; report explicitly preliminary. No universal dense-work elimination. PDF pp.1,4 visually inspected. |
| [Bit-Exact AI Inference Verification, v2, 5 Jun 2026](https://arxiv.org/html/2606.00279v2) | Pinned software/hardware/kernel execution can be emulated bit-exactly in studied configurations, including native intermediate behavior. | Verification/emulation supplies a better reference ABI; it does not make the producer's dense inference free. |
| [Approaching Shannon Bound, 14 Jun 2026](https://arxiv.org/abs/2606.15789) | Mixed-format headline up to 10×; **BF16 Figure 6 is roughly 1.4–1.5×**, not 10×. Tile ANS improves serving batch capacity/throughput. | PDF p.9 inspected visually. Table II Qwen-14B seq1024 says batch 47→60, while abstract says 47→75. Preserve the discrepancy; no B=1 latency proof is inferred. Empirical symbol entropy is not a universal program-compression lower bound. |
| [Oilbird, 4 Aug 2026](https://arxiv.org/abs/2608.03839) | Abstract: committed hidden-state retrieval improves accepted length 24–29%; API-Bank speed 4.4× versus 3.9× strongest training-free harness baseline. | A causal draft source worth recording, but still uses a target verifier. No arbitrary-input acceptance-length/state-cost bound. Abstract-level screen only. |
| [FluxBin, 16 Aug 2026](https://arxiv.org/html/2608.15602v1) | Post-training binary-basis approximation with specialized LUT kernels, up to 5.92× speed and 70B on one A100 80 GB. | No fine-tuning is not no weight change. The approximation equation does not satisfy the unchanged-checkpoint contract. |
| [Strong Drafts Need Compact Memories, 31 Aug 2026](https://arxiv.org/html/2608.30252v1) | Draft KV reduction over 70%, up to 3.33× on a 70B target; trained draft memory adaptor, full target KV/verifier retained. | Reduces draft-side cost, not the full target state/dense computation. Distributional speculative correctness alone is weaker than the same-RNG trace requirement. |
| [Unfolding the Leech Lattice, 2 Sep 2026](https://arxiv.org/abs/2609.02652) | Abstract explicitly distinguishes 2-bit disk representation from a 4.80-bit served layout; quality changes reported. | Strong reminder to count the physical serving format. Bit-exact layout decoding is not equality to the pre-quantization model. Abstract-level screen only. |

Additional searches covered exact/structured matrix–vector data structures, quantization, cached-state reuse and recent speculative work. Existing repository failures F-036–F-057 were used to avoid rebranding static tables, residual balls, Hamming trees, ideal drafts or uncharged dual images. No claim of novelty over compiler correctness, interval arithmetic, adaptive precision or tensor-core modeling is made.

## 2. Final theorem and three-way constructive screen

The full target theorem remains: a uniformly constructible executor preserves the declared native output/RNG/successor state for every allowed unmodified checkpoint/input, uses at most 8 GiB of total GPU memory, and meets the same-machine native-4B-Q4 latency/TTFT contract after all costs. No result below proves that theorem.

**A — Apply encoded checkpoint symbols directly, instead of reconstructing every weight.** Construct a lossless tiled code and a causal input-dependent interpreter that updates native accumulators. Desired >=10× route: one paid code action must replace many otherwise separate coefficient uses, not merely shorten each coefficient. Exactness requires `DecodeAct(E(W),x)=Native(W,x)` including operand-level numerical behavior. Full cost is preparation amortization plus selected-code IO, codebook/index IO, interpreter work and state. ZipServ/ANS reconstruct each consumed weight; their published rates do not close this new interpreter. RSR's binary/ternary premise cannot be imported into arbitrary BF16. No qualifying full-source constructor was found. Reject current implementation as a core; retain the open coded-action question.

**B — Commit native state fragments, rather than only propose future tokens.** Build a causal index from already committed exact states; retrieve candidate token/state fragments, check exact preconditions and RNG correspondence, and execute unpaid portions only. Desired >=10× route: a long fragment supplies both answers and successor state without an equivalent dense producer. Full cost includes index builds, fragment storage, searches, rejection, native checks and all required KV. Equal token suffixes or nearby hidden vectors are not equal full causal states. Oilbird improves the draft, not this missing exact-state constructor. A draft acceptance improvement is not a lower bound on arbitrary-input accepted length. No core admission.

**C — Preserve actual observable storage/control boundaries, not every private register.** Compute a sound native-result enclosure from checkpoint bit prefixes; refine only unresolved source bits; return an exact stored value when the enclosure lies in one rounding cell. Desired >=10× route: prove that most physical checkpoint/code work is unnecessary before a complete native output/state cut is fixed. This reverses the verification unit, not the objective. The finite prefix constructor below closes a small concrete source, but its literal-header cost already rejects it as a 10× core. It is retained as an ABI/certificate reference only, not a new giant backend.

These are discovery comparisons, not three falsely admitted winners. A different information source is required; changing prefix widths, seeds or the block size cannot defeat the literal-header floor proved below.

## 3. Constructive source: unknown weight bits to an exact native stored value

Reference files: `experiments/native_observable/reference.py`, `audit.py` and the preregistration.

### Declared model, not an asserted complete GPU ABI

There are eight BF16 products, a known FP32 accumulator and an **already present** final BF16 nearest-even store. Operand exponents are restricted to [-30,30]; the nonzero accumulator is an exactly representable FP32 value in [2^-60,2^60] in magnitude. Exceptional hardware behavior is outside scope. No extra BF16 store may be inserted between native FP32 blocks. The source cannot replace a full real projection until its complete native reduction graph is reproduced.

The raw exponent alignment follows the normal-range structure described for A100 in the cited paper (v3 inspected first; latest v4 and its Section 4.1.2 checked before publication) and upstream `models/A100TC.m`, Git blob `a8bd8af19589671099f31f0c0d8c58642c3398c1`. The upstream MATLAB implementation was inspected but not executed or copied. A100 is a reference architecture, **not an assertion that the target 8-GiB device is an A100**.

For normal operands, let e(w),e(x) denote their input exponents, without normalizing the product. Put

```
E = max_j(e(w_j)+e(x_j), e(c) when c != 0)
u = 2^(E-24)
q = trunc(c/u) + sum_j trunc(w_j*x_j/u)
z = RZ_FP32(q*u)
y = RN_BF16(z)
```

In the guarded range, minimum-exponent clamps do not activate. Truncation is toward zero. This is a declared finite numerical model; empirical similarity to hardware is not promoted to universal hardware proof.

### Where the causal information comes from

The constructor reads each original 16-bit weight once and separates its 9-bit sign/exponent header from seven mantissa bits. It never changes a weight. At query time it reads all headers, so E is known from headers and current inputs without consulting undisclosed mantissas. With k revealed leading mantissa bits p,

```
m_min = p * 2^(7-k)
m_max = m_min + 2^(7-k) - 1
w is in sign * (128 + [m_min,m_max]) * 2^(e-7)
```

Multiplication by the known input and fixed-E truncation give integer intervals `[q_j^-, q_j^+]`. Add those intervals and the known accumulator contribution, then apply the monotone FP32 toward-zero conversion to obtain `[z^-,z^+]`. If the two **BF16 casts** coincide, return that stored value. Otherwise reveal one bit of the currently widest contribution interval, update it, and repeat. No future token, exact answer, completed full dot product or hidden native oracle is used by the generator. The independently evaluated reference output is used only after generation in tests.

### Correctness and termination

Every real weight lies in its prefix interval. Since E is fixed by operand headers, each aligned integer contribution lies in the computed endpoint interval. Their sum contains the original integer sum, and the post-normalization conversion preserves containment. If the final cast maps the enclosing interval to one value, the original store must equal it. This uses monotonicity of the **cast and fixed-E scalar truncation**, not an unjustified claim that an arbitrary Tensor Core operator is globally monotone.

At most 56 mantissa-bit reveals make all eight intervals singleton, so the guarded model terminates. Zero-crossing intervals are not accepted unless exactly zero in this model. Signed-zero, NaN, infinity and subnormal hardware semantics require their own native treatment and are not certified here.

### Complete-cut state theorem

Let a native region consume state/input S and expose a complete cut C containing every live-out value, externally visible output, persistent KV/state write, control decision and RNG state/consumption needed outside the region. If a replacement leaves C bit-identical for every admissible S and performs no additional observable side effects, then replacement preserves all later execution by induction over regions and token steps. It need not preserve dead private FP32 registers.

This is a **sufficient compositional rule**, not a discovery of a cheap generator for every region. It does not authorize token-only checks, forgetting live FP32 accumulators, changing the sampling coupling, or declaring a synthetic dot-product test a Transformer state proof.

### Counterexample makes the distinction testable

The two eight-product cases have identical mathematical products. In case A the first operands are 1.5 and 1.5; in B they are 1 and 2.25. Both add four products `2^-12 * 2^-12`, three zero products and accumulator `2^-7`.

The declared raw-exponent model yields:

```
A: pre-store = 9469953/4194304 ; BF16 = 145/64 = 2.265625 (0x4011)
B: pre-store = 289/128        ; BF16 = 9/4    = 2.25     (0x4010)
```

Thus preserving real products alone is not sufficient even for the final BF16 store in this model. Conversely, `1` and `1+2^-23` have different private FP32 bits but the same BF16 store. Subtracting 1 **before** a store distinguishes them, demonstrating why complete cut placement matters. The witness is locally reproduced in a model, not claimed as a new GPU measurement.

## 4. Cost closure: why this is not a fast core

For P coefficients this literal-header source reads at least `9P` header bits before any mantissa refinement. Relative to a raw 16P-bit scan, even free refinement/certification has at most `16/9 = 1.777...` logical reduction. The floor is specific to this source; it is not a lower bound on all compressed or global query-adaptive codes. Physical pages/transactions, metadata and cold-to-hot movement can only add cost to the registered implementation.

For one eight-product block with r revealed mantissa bits:

```
constructor: 128 original weight bits read; all 128 represented losslessly
query: 72 + r logical weight bits, 0 <= r <= 56
interval endpoint products: 16 + 2r, at most 128
certificate checks: 1 + r, at most 57
plus endpoint additions, selection, input access, casts and program storage
```

These are operation/bit counts, **not measured speedups**. The Python implementation is a correctness reference and uses rational arithmetic; it is not a candidate GPU kernel.

Registered 256 normal synthetic blocks gave zero output mismatches; 254 certified before all mantissa bits were disclosed. Nevertheless mean weight reads were **116.39453125/128 = 90.933227539%**, and mean endpoint products were **104.7890625** versus eight original product terms. High certificate frequency is not cheap certificate generation. Exhaustive checks of 65,536 suffix completions and 4,096 independent rounding controls passed. Twelve focused unit tests passed. This does not establish even the 10× admission gate.

An optimistic full-resident illustration, not a universal memory lower bound: 405e9 coefficients at 16 bits occupy about 754.37 GiB; at 11 bits about 518.63 GiB. All 8 GiB devoted to weights would allow only about 0.16968 bits/coefficient, before KV/workspaces. Cold-backed execution remains a permitted research class, but must supply and pay its actual query mechanism.

For cold-streaming speculative proposals, a necessary traffic equation is
`T >= D_cold_per_verified_block / (A * bandwidth_upper_bound)` for accepted block length A. It proves no sufficient latency bound and omits nonnegative drafting, dense arithmetic, verification and state costs. A successful target proof needs a realizable schedule and sufficient upper bound, coupled to the real same-machine baseline; multiplying 1.5× compression by a paper's unrelated speedup is not that proof.

## 5. Obligations, decision and handoff

| Obligation | Change this round | Full target status |
|---|---|---|
| O1 uniform constructor | Finite lossless header/prefix source for a guarded block, no undisclosed-bit oracle | OPEN for the mission |
| O2 causal execution | Paid prefix generator terminates in at most 56 reveals | Complete prefill/decode/state program OPEN |
| O3 native/state | Explicit raw-operand model, verified store certificate, complete-cut theorem | Actual pinned kernel graph and whole-state implementation OPEN |
| O4 full cost | Header floor and finite reference work bounds quantified | No cheap full-model source |
| O5 target budget | Literal source rejected before any backend admission | 8-GiB/latency/TTFT NOT ESTABLISHED |
| O6 reproducibility | Source, frozen protocol, exact captures and independent controls retained | Only scoped results reproduced |

**Decision:** retain the numerical source/checker as auxiliary; do not enlarge its synthetic benchmark, optimize it into CUDA, or claim feasibility increased. Continue the coded-action or exact-state-fragment question only when a new paid source removes the per-coefficient access premise. No universal impossibility is claimed.

The prior local-only causal-source ZIP was inspected as context, not silently treated as part of the remote base. Its SHA is recorded in provenance. No target weights, remote compute, GPU allocation or GitHub Actions were used. Shell Git failed DNS and the local-PC connector was offline; the connected GitHub writer is available for this round.

Scientific status: `THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`. Repository recording is independently verified after commit creation. A saved discovery round is not the requested completed execution theory.

## Reproduce

```
python experiments/native_observable/audit.py
python -m unittest discover -s tests/native_observable -v
```

The audit requires Python 3 and NumPy. It writes a human-readable local `cases.jsonl`; the committed summary contains its lossless fixed-width base64 capture and SHA256. The decoder embedded in `audit.py` checks byte-identical reconstruction. This capture encodes test records, not checkpoint weights, and is not evidence of model compression.
