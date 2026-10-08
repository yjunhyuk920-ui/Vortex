# Vortex source-constructor follow-up

Prepared 2026-10-08 UTC. This review adds public-source and implementation evidence after commit `2dc337cf1e87faa755638272e6d042dbf65b4d18`. It contains no new numerical experiment, model run, downloaded checkpoint, or hardware performance measurement.

## Result

No reviewed construction is admitted as a new Vortex core. The mission remains open. The original arbitrary unmodified dense HF405B, native output/RNG/successor-state, total 8 GiB GPU, same-machine latency and fully charged-cost contract is unchanged. A tenfold work **or** traffic route is an entry requirement, not a measured success or a requirement that both axes separately shrink tenfold.

This review compares three different information-dependence principles. They are diagnostic categories, not three newly qualified candidates, and do not reopen excluded codec, graph-enlargement, approximate pruning or low-rank experiments.

## 1. Weight words generated from a compact description

### Brevis: new primary evidence, archival scope

[Lossless Tensor Compression as Program Synthesis, sections 2–3](https://arxiv.org/html/2608.02162v1) constructs a finite typed program from the exact source stream. Reversible nodes reconstruct physical tensor words, including exceptional encodings; literals keep every supported tensor representable. Its objective is complete serialized size. The reported corpus compression ratio is 1.5135, with 6.61 GB/s decompression on a 192-core EPYC server with 724 GiB RAM and warm caches. Timing excludes cache conditioning and verification. Accelerator-aware decoding is future work. This is an archival result, not a dense-effect or native inference speedup.

The independently inspected [interpreter](https://github.com/jiekeshi/Brevis/blob/d8d25085605d23faf1c6784a30c0cc4b4fcaa34e/src/interpreter.zig) creates the full output stream. Map first writes its child then transforms in place; Scan performs a wordwise pass; Merge allocates one child scratch stream at a time. [executeVerified](https://github.com/jiekeshi/Brevis/blob/d8d25085605d23faf1c6784a30c0cc4b4fcaa34e/src/tensor_archive.zig#L500-L527) executes then hashes decoded bytes, with a root-literal view shortcut. These paths generate weight words, not their input-dependent original dense effects.

Cheapest screening evidence is already present: serialized source ratio, complete output writes and remaining downstream dense work. No compression run is warranted to repeat the closed codec branch. A genuinely different future proposal would need fused native-effect generation with charged construction, decoding, scratch, verification and state costs, rather than only a smaller weight file. No such implementation was found here.

## 2. Whole computation rearranged by symbolic equivalence

### EquiForge: explicit real-arithmetic and tolerance contract

[Unleashing the Power of Equality Saturation for Tensor Program Superoptimization, section 4](https://arxiv.org/html/2609.12330v1#S4) expressly assumes real arithmetic; finite-precision results may change through reassociation and internal precision. Section 7.1 uses SymPy repair identities, a four-hour search/tuning budget per configuration, A100 80 GB and RTX 5090 devices, and relative L2 tolerance 0.01. Tensor benchmarks report 1.32x geometric-mean speedup over the fastest baseline. These are compiler/scheduling results with a different numerical contract.

Cheapest validation is the paper's own semantic and checker specification. No new reassociation counterexample or compiler run is needed. Its unified IR and extraction do not provide a compact original dense-effect producer, and expanding prior FX/e-graph/CSE families remains outside the next-core frontier.

## 3. Input-dependent certificate obtained before omitted work

### Exact remaining bounds versus empirical sign prediction

[Threshold-Based Early Stopping of Accumulations in Neural Networks with Binary Activation, sections 3.3, 4.3, 6.1 and 7.3](https://arxiv.org/html/2608.06177v1) uses binary inputs and binary sign outputs. Its exact certificate compares an evaluated prefix with the sum of absolute remaining weights. The F7 median exact firing point is 3,789 of 4,608 terms, leaving only about 18% unevaluated. The much larger 86.6% saving belongs to an empirical policy. The Gaussian calibration is not a formal error guarantee; full fallback still allows floating-point reordering differences. Traffic, synchronization, branching and checkpoint latency/energy are not modeled.

For M outputs and N terms, directly implementing the stated permutation and bounds requires source-weight inspection, unit-specific sorting and suffix-bound construction. Storing J checkpoint bounds costs M×J entries; runtime pays evaluated-prefix loads/accumulation and certificate loads/comparisons. These are analytical costs of that procedure, not measured RSS or a universal lower bound. The real-sum certificate does not enclose every original native rounded reduction. No solver or floating-point experiment was launched.

### ReLU early-exit circuitry is not a free weight-read oracle

[Early exit for ReLU-based activation, Table 7 and Figure 29](https://patents.google.com/patent/US20250284937A1/en) checks all operand-pair sign XORs and their AND for its first exit. One embodiment computes signs alongside mantissa products; another allows the sign result to bypass later operations. The second exit occurs after products, alignment and summation, saving renormalization/rounding. Operand components are already held by input circuitry. This technical construction is not evidence of skipped weight fetches, a native SiLU certificate or a tenfold whole-system cost reduction. Mixed product signs defeat the first guard. No legal opinion or hardware execution is implied by this review.

### Standard gate cost and the architecture-change distinction

[Model Casting and Low-Parameter Gating, sections 3.1–3.4](https://arxiv.org/html/2609.31975v1) explicitly charges the always-on gate: with sparsity s its standard FFN arithmetic ratio is 3/(3−2s), at most 3. Going beyond that changes the gate parameterization to a sparse-plus-low-rank structure. Casting changes SiLU to R-S+, approximates the original gate where applicable, and resumes training. Thus its cheaper source comes from changing the model, which Vortex excludes. This cap is scoped to the displayed full-gate/two-sparse-projection procedure, not every possible FFN algorithm. No native SiLU universal-nonzero claim is made.

The separately reviewed [ADET publisher preview](https://www.sciencedirect.com/science/article/abs/pii/S1434841126003547) discusses binary/ternary CNN hardware. Its body equations, bound storage and full producer overhead were not accessible in this review. A preview's accuracy/power description is insufficient to admit a verifier or an experiment.

## Unresolved soft-gate source

The [official TMLR index](https://jmlr.org/tmlr/papers/) links *Mechanistic Interpretability of Transformer MLPs Through Exact Soft-Gate Decomposition* to [OpenReview](https://openreview.net/forum?id=20lTANRDAz) and [Zenodo code DOI](https://doi.org/10.5281/zenodo.22959811). The [university abstract](https://www.ipr.mdu.se/publications/7463-Mechanistic_Interpretability_of_Transformer_MLPs_Through_Exact_Soft___Gate_Decomposition) and verified author/project links were checked. The exact paper body and matching code remain unread: browser verification or explicit copyright acceptance was not performed, and other public metadata paths returned tool errors/cache misses. A different January ReLU repository was not substituted.

An input-dependent effective matrix does not necessarily require the entire original forward to finish first. In a bias-free gated MLP, the gate branch can define the algebraic matrix `Wdown diag(SiLU(Wgate x)) Wup`. Conventional explicit formation costs approximately M×D² operations and D² output storage; factorized evaluation retains the dense projections. This is a general implementation analysis, not an assertion about the unread paper's actual algorithm or a lower bound over all algorithms. Neither floating-point precision wording nor a real identity establishes original bit/RNG/state equality.

## Admission condition for the next genuine lead

A future lead must supply a finite constructor, causal executable source, native-word/state induction, and a sufficient total-cost upper bound before any new experiment. The following cost screen makes the producer dependency explicit without selecting a new core:

Let W be charged baseline work (or, separately, charged baseline traffic), f the share that remains unchanged outside the proposed replacement, p the charged certificate/producer/maintenance cost as a fraction of W, and r the fraction of the replaced work retained. Then the proposal's charged ratio is at least `f + p + (1−f)r`. A tenfold route on that axis requires this quantity to be at most 0.1. If `f+p>0.1`, improving only the omitted part cannot satisfy that route. This is an accounting identity for a specified partition; it is not a universal impossibility theorem, and arithmetic and traffic must not be added without a declared cost model.

For a native reachability-certificate lead, the cheapest useful first deliverable is its exact finite input/state domain, rounded-reduction enclosure, certificate procedure and metadata-size/operation upper bound. Failure to provide those blocks an experiment. A few passing values, empirical sparsity or an oracle label do not replace this deliverable. The present sources do not justify a new scalar toy, target download or full-model launch.

O1–O6 remain OPEN. CORE_ADMISSION=false. NEW_EXPERIMENTS=0. MODEL_RUNS=0. TARGET_PERFORMANCE_NOT_MEASURED. Repository preservation is a separate operation, not acceleration progress.
