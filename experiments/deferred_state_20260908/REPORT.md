# Deferred native state and synchronizing suffix — 2026-09-08

**Bounded synthetic state construction, not the 405B engine.** Fixed mission/CTC unchanged. Base PR144 b8ce3f927d090b3e16a4cf4292ee96a0a4d8e91d. THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED; CORE_ADMISSION=false; FULL_MISSION_O1_O6=OPEN; THREE_QUALIFYING_NEW_PRINCIPLES=false.

## Constructed algorithm and proof boundary

Represent the exact native state by an unevaluated expression plus a sound interval, not merely by the token already chosen. Gamma(node)=original state is an inductive invariant. Given the SAME uint32 RNG word, a Bernoulli token is certain when u<pLower or u>=pUpper. Otherwise the runtime evaluates the pending original transitions and charges every original weight read. No reference future tokens/states are supplied to the candidate. Only a WHOLE STATE interval whose endpoints have identical nonzero BF16 words authorizes deleting history. Zero is excluded from deletion to avoid merging signed zero.

The tested original recurrence contains dense G/U/D and SiLU: g=dot(G,h), v=dot(U,h), r=dot(D,BF16(SiLU_BF16(g)*v)), h'=BF16(BF16(BF16(.5h)+E[token])+r). Dot means separate FP32 products, balanced reduction, final BF16 RNE. Dense observer plus frozen sigmoid feeds an original two-symbol Bernoulli interface. This is NOT a Transformer, attention/KV, arbitrary HF categorical sampler or exposed-logits API. Finite nonoverflow bounds and unary ABI are explicit restrictions, not changes to the final mission.

Coarse bounds are generated from exact rational row L1 norms: with width b and depth d, F=(1+2^-24)^(d+1), native magnitude <=F*rowNorm*maxAbsInput+b*F*2^-150. Outward BF16 rounding and exact discrete SiLU min/max queries provide an executable sound enclosure. Unary tables come from torch2.10.0+cpu and ARE SHARED with the independent arithmetic C reference; no independent transcendental proof or CUDA equivalence is claimed.

## Deferred work is debt, not elimination

For T steps: T=F+Z+P, where F is evaluated original bodies, Z is bitwise-proved permanent deletions and P is unresolved work. Exact final-state flush enforces F+Z=T.

18 preregistered models,36runs,4608causal steps,86016state-coordinate checks: no token/RNG disagreement or bound escape. Tiny-gain defer-only performs zero bodies before flush but owes all128 afterwards. Tiny coalescence deletes128, yet ablation finds nonlinear branch changes0/14336statecoordinates: not an active-wide-body breakthrough. Moderate/wide coalescence still evaluates127/128bodies including flush. One n32 moderate run has per-decision body-resolution p50=0,p95=6,max=9; these are workcounts, NOT latency ratios. Range/metadata/table/observer costs are additional.

## Follow-on: generate a native synchronizing suffix certificate

First use ORIGINAL coefficient signed endpoint arithmetic to verify invariant B=[0,4]^n for both legal inputs. Then transport B through the actual recent word v. If the resulting interval is a bit singleton h*, soundness proves T_v(s)=h* for EVERY predecessor s in B. No all-state enumeration and no ideal state selector. Original input history must already be known; generating its past outputs is not free.

Initial tiny/moderate12models pass invariant checks and first certify at TESTED suffix16; wide6 fail the box. Each of the12recovered states continues32steps with full original body/C comparison:384correctness steps, not acceleration.

A later explicitly post-hoc ACTIVE-POSITIVE control uses positive dense G/U/D with gain7/(64n). Its nonlinear branch changes14224/14336statecoordinates, so this is not the tiny silent branch. Six models pass B; five certify final state at tested16, one n16seed17 does not certify through128. Candidate/original endpoints agree. Their independent128step lazy-policy experiments add1536causal steps, but still pay127/128bodies afterflush. These are favorable SYNTHETIC controls, not trained checkpoints or held-out HF generalization.

Every sharp interval step reads all original body coefficients and uses twice the original linear product count, plus range/reduction/allocation work. Actual attempted lengths1,2,4,8,16,32,64,128 total255interval passes, plus2invariant passes. Do not report only the successful16 as full construction cost. An online cost-optimal suffix chooser was NOT implemented: the suffix follow-on is a final-state recovery/certificate experiment.

## Scope witnesses

Native f(x)=BF16(BF16(.5x)+.50390625) has distinct exact fixed points1.0078125 and1.015625, verified by separate C. Real contraction alone does not prove bit synchronization. Equal current sampled tokens also fail continuation equivalence. Literal append-only caches preserve distinct prefixes under the same appended suffix; this does not rule out every behavioral state quotient. No actual HF KV experiment is claimed.

## Complete accounting and missing obligation

Original G/U/D remain in cold files; exact forcing actually reloads them. n32sample has12288Bweights and8048BsourceJSON; metadata Fraction work counts. Unary table/index payload1703424B and131072initial unary evaluations are additional. Pending intervals alone need8*n*P bytes (32768B for n32,P128), plus expressions/objects/input records. Tracemalloc probes exclude preloaded tables/model/interpreter/RSS/GPU and are diagnostic, not deterministic evidence or total memory.

Preparation, coefficient reads, range construction, observer, RNG, copies, allocation, sharp attempts and flush all remain. No >=10x full-work bound or405B/8GiB/4BQ4/TTFT result exists. Public-checkpoint retrieval failed container DNS; not a scientific rejection. The missing mechanism is a cheap producer of original wide-body/KV-dependent exact effects or certified coalescence, rather than moving dense work into bounds or deferred reconstruction.

## Reproduce and preservation

Five [CAPSULE_0.txt](CAPSULE_0.txt) through CAPSULE_4.txt contain19checksummed text files: Python/C sources,17tests,initial preregistration,two labeled later plans,full Korean proof/report,frozen152-file manifest,validation/environment. Archive compression is NOT model compression. Raw science and available development logs are in user ZIP; raw science regenerates. No claim that unlogged development commands were preserved.

Run `python restore.py --output restored`, inspect `restored/docs/REPORT_KO.md`, then `cd restored && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify.py`. Python/NumPy/torch2.10.0+cpu/GCC required. No network required. Do not refreeze the expected manifest. Fresh ZIP extraction, C recompilation,17tests and152sciencefile hash equality were verified locally. Manifest SHA256:49098e4052a39dfa942c980696d838161f81b32bc87a4df1a57cffae61a43c21. XZ SHA256:81050105dcbfba723defdb8c9ee05901d7777cd21de2704c59a4cd3c99363243. JSON SHA256:6c737b39b9e8e984558d916b00dd832cfe1104e42420e1b3ad6b246e476802d9.

No Actions/fullrepositorysuite or target hardware run. Prior four root blobs are preserved unchanged under docs/research/history/pre_deferred_state_20260908. The previous local-only error-transport study is not silently claimed as part of this remote commit.

## Primary context, not novelty claims

- CGAL Lazy_exact_nt: https://doc.cgal.org/latest/Number_types/classCGAL_1_1Lazy__exact__nt.html
- Synchronizing Random Almost-Group Automata: https://arxiv.org/abs/1805.02154
- NVIDIA original floating point order/FMA/ABI: https://docs.nvidia.com/cuda/floating-point/index.html
