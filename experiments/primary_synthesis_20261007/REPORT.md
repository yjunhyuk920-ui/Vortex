# Vortex native execution research review

Prepared 2026-10-07 UTC. This is a primary-source review and one public-constant compatibility experiment. It does not establish a new acceleration method.

## Conclusion

No newly reviewed method has supplied a source-supported, fully costed replacement for the dominant original dense operations while preserving the original finite-word behavior and required state. The specific unguarded STENSO simplification `x/sqrt(x) -> sqrt(x)` fails binary32 equality at the positive finite input `x=3`. That finding rejects that rule under Vortex's exact contract, not every STENSO output, floating-point optimization, or possible Vortex algorithm.

The overall mission remains open. One rejected implementation or completed review does not cancel continued discovery. Existing failed CSE, codec, scalar-chart, low-rank, quotient and input-template approaches have not been reopened by this review.

## Unchanged acceptance contract

The repository's current mission is a uniform executor for arbitrary public, unmodified dense Hugging Face 405B-class checkpoints, batch one, one GPU with total peak allocation at most 8 GiB, original output and RNG behavior, and required successor-state correspondence for legal continuations. The same-machine native 4B Q4 warm latency requirements are p50 at most 1.2 times baseline and p95 at most 1.5 times baseline; the existing TTFT requirement remains. Every preparation, source, host-memory, SSD, transfer, arithmetic, state, verification, guard, repair and fallback cost is charged.

A tenfold work or traffic route is a core-entry criterion, not final latency acceptance. The official contract specifically does not impose separate universal tenfold reductions on both arithmetic and traffic. A lower bound that fits a budget is not a sufficient algorithm upper bound.

Sources: [mission](https://github.com/yjunhyuk920-ui/Vortex/blob/research/cloud-continuation-20260930/MISSION_AND_WORKING_PRINCIPLES.md), [constructive contract](https://github.com/yjunhyuk920-ui/Vortex/blob/research/cloud-continuation-20260930/docs/CONSTRUCTIVE_THEORY_CONTRACT.md), [current next obligation](https://github.com/yjunhyuk920-ui/Vortex/blob/research/cloud-continuation-20260930/NEXT_EXPERIMENT.md).

## New public-constant experiment

STENSO's CGO 2026 paper uses symbolic algebra for program synthesis and shows `x/sqrt(x) -> sqrt(x)` among its simplifications. Exact preservation of that displayed unguarded rule was preregistered independently of target data. [Paper, section VII D and page 9](https://alexanderb14.github.io/papers/stenso.pdf).

Under binary32 round-to-nearest-even:

- Original `3/RN32(sqrt(3))`: `0x3fddb3d8`
- Rewritten `RN32(sqrt(3))`: `0x3fddb3d7`
- Positive finite `x=4` control: both `0x40000000`

The independent Fraction proof checks that the squared midpoints around `0x3fddb3d7` strictly bracket 3, and that the exact rational `3/decode(0x3fddb3d7)` lies strictly between the midpoints around `0x3fddb3d8`. All four strict inequalities hold, with no ties.

One local cloud CPU process executed the two inputs through strict-FP GCC 14.2 and existing libm. Its exit status was 0, and build and run stderr were empty. The preregistration, source and exact-expectation hashes remained unchanged after execution. The experiment read no original tensors and ran no model, GPU or user-computer commands.

Observed C exception flags were 32 for both x=3 paths and 0 for both x=4 paths. These are the five `FE_ALL_EXCEPT` classes exposed by C `fetestexcept`, not a full x87/MXCSR raw-state comparison. The restoration field records successful `fegetenv`/`fesetenv` API calls; it does not independently certify complete environment-byte equality.

Evidence: `PREREGISTRATION.json`, `PREEXECUTION.sha256`, `check_stenso.c`, `prove_expectation.py`, `EXACT_EXPECTATION.json`, `COMMANDS.txt`, `RUN.json`, and the unchanged execution logs.

## Primary literature dispositions

### Ozaki 2.5

The manuscript studies FP64 emulation on FP8 tensor cores, particularly operand deconstruction and proposed Rubin hardware assistance. Its abstract explicitly classifies every performance result as a model projection pending measurement. It does not supply an original ordered BF16/FP32 reduction and state-preserving 405B executor on an 8 GiB GPU. Conversion/reconstruction accounting is useful background, but projected hardware rates are not Vortex measurements. [Primary manuscript](https://arxiv.org/html/2609.09095v1).

### Bit Exact AI Inference Verification

This work emulates the hardware and software numerical fingerprints of inference, including reduction topology and special-function behavior. The original order of accumulation remains important in the emulator. Section 6's metadata list explicitly excludes software sampling from its scope. Thus the paper offers numerical-audit machinery, not a cheaper original dense-effect source or a complete RNG and successor-state proof. [Primary manuscript, sections 2, 4 and 6](https://arxiv.org/html/2606.00279v2).

### ARCH floating point operators

ARCH's paper proves result-value properties under a fixed RNE regime and describes a bounded sticky-fold FMA datapath. Section 4.1 expressly excludes dynamic rounding and exception status flags. This distinction prevents direct promotion to Vortex's full native-state contract. The public flags document at inspected commit `71156beda2fcafe9c864e32f19b48e4fa0691a32` labels itself an implementation-pending proposal, uses four flags and tininess before rounding, and requires explicitly wiring a status accumulator. We did not run its Lean or SMT proofs. [Paper](https://arxiv.org/html/2607.23715v1), [inspected flags proposal](https://github.com/arch-hdl-lang/arch-com/blob/71156beda2fcafe9c864e32f19b48e4fa0691a32/doc/proposal_fp_flags.md).

### ROVER

ROVER provides bit-vector rewriting, shared-expression-aware extraction and decomposed equivalence checking. Its cost objective is circuit area, not Vortex's total software execution time. A verifier and extractor do not themselves establish a compact native source. Applying them merely to enlarge the already rejected CSE or scalar graph families would violate the unchanged research frontier. No ROVER solver or hardware synthesis was executed. [Primary manuscript, sections V and VI](https://arxiv.org/html/2406.12421v1).

### Cleave

The October 6 manuscript separates algebraic graph search from concrete operator scheduling and uses probabilistic equivalence testing. Its evaluation uses FP16 inputs/outputs and FP32 accumulation on A100 40 GB and H200 141 GB devices. It does not establish the exact original finite-word trace or Vortex's target memory and latency requirements. Inspected public code defaults to `assert_close` tolerance `rtol=0.5, atol=0.5`; callers can supply other tolerances, so this is not a statement about every benchmark's setting. The default checker does not certify bit identity. [Paper](https://arxiv.org/html/2610.07742v1), [runtime at inspected commit](https://github.com/nyu-systems/cleave/blob/ab4cecb40c92f07c8d4247c5b8e0fc9f38ae7eb8/python/cleave/util/runtime.py), [autotune](https://github.com/nyu-systems/cleave/blob/ab4cecb40c92f07c8d4247c5b8e0fc9f38ae7eb8/python/cleave/util/autotune.py).

## Next evidence needed

The decisive missing object remains an automatically constructed, causal native-effect source that produces the required outputs and state before doing the dominant operations it replaces, with a sufficient target-scale upper cost bound. A compiler, exact weight decoder or function quotient without that object remains auxiliary.

The next discovery target is native finite-word reachability-certified structural slicing for SwiGLU or SiLU: a source construction that certifies an always-identical effect before executing its dominant dot product, over the unchanged legal token and state domain. A useful proposal must state the finite constructor and its input-dependent acceptance predicate, preserve reduction order and all exposed state, and charge recognition, maintenance and fallback. No qualifying primary construction or justified implementation was obtained in this review, so no solver or model launch is admitted merely by naming this target.

O1 uniform construction, O2 causal program, O3 native output and state induction, O4 total costs, O5 target budget, and O6 complete independent evidence remain OPEN. The STENSO experiment supplies a bounded compatibility counterexample only; it does not close any full-mission obligation.

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
HANDOFF_STATUS=IN_PROGRESS

These status fields are the pre-publication authoring snapshot. Repository preservation is reported separately after commit and exact readback. No 405B run, speedup or target memory result is claimed.
