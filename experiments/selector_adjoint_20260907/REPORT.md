# Selector-linear extraction: exact auxiliary construction, failed source-cost gate

2026-09-07. Parent PR141 / 8cb10dab1d3d1fdc2e2dc857b5db5bb9500def09.
Fixed VORTEX mission and CTC-2026-09-05 are unchanged. This is bounded auxiliary work, not a complete theory or a fast neural executor.

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false

## Three compared principles and admission

A: Obtain all required output/state bits as coefficients of one selector-linear scalar program, using a paid reverse pass. A small exact scalar producer is the missing construction; a scalar value or a wrapper around the original outputs is not such a producer.
B: Make original gate constraints a unique zero-energy state. Uniqueness does not give a cheap solver; the direct greedy proposal has a concrete local trap.
C: Recover a vector from directional expectations. Unbiasedness is not exact finite reconstruction; a fixed linear frame has a specific rank requirement.

No candidate established a >=10x route through the arbitrary native wide nonlinear body. Only the small extraction proof/checker was implemented; no large neural backend, source search or response enumeration. This does not reopen F-040's rejected universal self-contained hot circuit.

## Constructed typed program

Let x include current input/state/RNG bits, and b_i(x) denote required output and next-state bits. Work over F2 and form Phi(x,z)=XOR_i z_i*b_i(x). The z_i are formal selectors, NOT floating inputs.

Coefficient nodes C are constants0/1, input bits, XOR, AND, independent of z. Linear nodes L are zero, z_i, L1 XOR L2, and C*L. Products of two selector-bearing nodes are forbidden. The source constructor, binary serialization, type/cycle validation and interpreter are in src/adjoint.py.

Algorithm: evaluate C in topological order. Initialize all linear adjoints to zero, root to one. Scan L in reverse topological order. At XOR, XOR the adjoint into both children. At C*L, XOR the adjoint AND C into L. At z_i, XOR the adjoint into output i. Scan all nodes; no uncharged pruning.

Proof: every legal L is a linear form in z. The adjoint is the coefficient with which a node contributes to the root. XOR contributes additively and scaling contributes its z-independent coefficient. Reverse induction over the finite DAG therefore returns exactly every coefficient b_i. If required successor roots are included, equality can be used inductively across continuations, but no full Transformer bit-source constructor was produced here. Actual state test only retains the input's original32bits.

Important type trap: z and z^2 agree on Boolean values, but formal derivatives at zero over F2 are1 and0. Boolean idempotence must NOT be applied to selector-bearing multiplication. The grammar excludes it. This is coefficient extraction, not an STE or real differentiation through rounding.

## Paid costs and missing source

With c coefficient XOR/AND gates, a linear XOR nodes, g scaling nodes and ell selector leaves, extraction performs c+2a+2g+ell logical bit operations. Store all coefficient values, reverse adjoints and M output bits. Serialized format uses a24byte header and16bytes/node. Python objects, input/output lane packing, allocation, address handling and physical memory transactions are additional.

Phi(x,0)=0 for every x. Reading that zero reveals nothing about b. The actual program and current-input coefficients must still be available. Wrapping an S-gate original output circuit retains S and adds roughly5M-2 extraction operations. It does not eliminate any generic weight dependence. Preparation still reads original coefficients and generates topology. A cheap producer remains OPEN.

## Native primitive, not a neural layer

Constructed a FP32 raw-word to BF16 round-to-nearest-even Boolean circuit plus32unchanged raw-state roots. Modular32bit add-bias rule: (word+0x7fff+((word>>16)&1))>>16; original exponent/fraction separately detect NaNs and select canonical +0x7fc0. This ABI includes signed zero, subnormals, infinities and overflow, but does NOT claim arbitrary runtime NaN payload preservation. The low16bits and retained-bit parity prove the rounding rule, including carry at exponent boundaries.

Fixed inputs: all65536 upper halfwords crossed with lower [0,32767,32768,32769,65535], plus4096 random words seed17029:331776words. Candidate Python typed interpreter outputs and raw-state roots matched independent C frexp/ldexp rounding in every case.512Fraction neighbor-search checks also matched. This is not exhaustive over all2^32 words. Tests use packed lanes for verification, not batch1 latency.

C src/reference.c is an independent REFERENCE, not a fast candidate executor. Direct Boolean source225gates; extracted463logical operations;6472serialized bytes. Exactness held, work increased. No hardware speedup was measured.

## Arbitrary GF2 matrices and explicit source

Constructed Phi= XOR_j x_j (XOR_{i:W_ij=1} z_i), without first materializing vector y. W information lives in topology. No rank, repeated-input or sparse-difference assumption.8matrices n16/32/64/128, seeds17/29,256queries each:2048queries and122880output bits match direct GF2 products.128x128 original bitpacked2048bytes; program134680/134584bytes (65.7617/65.7148x). Every node field is scanned by this interpreter. This is a measured format cost, not a lower bound on all programs. GF2 parities are NOT numeric neural dot products.

12random AND/XOR programs with8inputs and16outputs were checked on all256inputs:3072inputs,49152outputbits. Reverse results match direct coefficients and independent single-selector forward probes. These are algebra checks, not model inference.

## Numerical and alternative-principle witnesses

Native reduction witness: weights[2^24,1,-2^24,1], inputs all1. BalancedFP32 output1; reverse sequential accumulation2. Therefore ordinary real transposition may not preserve the original numerical ABI. The bit method avoids this by constructing the actual native bits, but pays that source cost.

Energy: E=(a-1)^2+(b-a)^2+(c-a)^2 on binary a,b,c has unique zero111.000hasE1, each single-bit neighbor hasE2. All8assignments stored. This defeats only direct one-bit descent, not all solvers: the original forward copies solve it trivially.

Directions: for independent Rademacher r, E[r(r^T b)]=b. b=[1,0],r=[1,1] yields[1,1], not exact. Averaging the four2Dsign directions gives[1,0]. In the fixed input-independent linear frame R^TR/k=I_M, rank<=k implies k>=M. Exact Fraction elimination verifies ranks1..4 for firstkrows of a4x4Hadamard frame. This is not a lower bound on nonlinear/adaptive decoders or encodings with additional information; huge scalar precision must also be charged.

## O1-O6 / limitations

O1 uniform cheap constructor for unchanged checkpoints:OPEN.
O2 full loading/prefill/causal output/state program:OPEN.
O3 original output/RNG/all-continuation state proof:OPEN; only typed extraction/primitive state here.
O4 full construction/storage/movement/arithmetic/state costs:OPEN; local logical counts are not target bounds.
O5 target405B/8GiB/4BQ4/p50/p95/TTFT:OPEN,NOT TESTED.
O6 independent complete-theory evidence:OPEN; bounded source/tests/replay are available.

PREREGISTRATION.md is preserved byte-for-byte. Its informal O-label descriptions were imprecisely mapped; the canonical ledger immediately above corrects the labels without changing hypotheses, cohorts or thresholds. Original preregSHA256:a83f2cb5a509f93a5d4929fb0d9be75728ab7bba7afacd36374f331b1186b2cb.

16unit tests pass;84scientific files replay to frozen manifest SHA256 b3a1f11efec95418855c985478e9dbb58ff10b7003528e0444345389d3b6c96e. verify.py compares the pre-existing hash and never updates expected results. Raw programs/input/output/state files regenerate from source and are in the userZIP; NOT all raw binaries are committed. Full Korean report/development logs are in the userZIP; this remote report contains the complete bounded construction, proof, costs and scope. No full pretrained model/KV/RNG/CUDA/405B/VRAM/4B latency/Actions/full-repository suite run.

Next: do not grow AD wrappers, GF2 sizes, energy search or random samples without a new source mechanism eliminating original native work and preserving state. A cheap decoder/extractor is not a cheap producer. Reject these candidates, not the overall mission.

## Primary references and nonclaims

Fischer, Deterministic Sparse Pattern Matching via the Baur-Strassen Theorem, https://arxiv.org/abs/2310.11913 : prior scalar/circuit differentiation context, not a405B result.
Atkey/Perera, Data Provenance as Automatic Differentiation, https://arxiv.org/html/2511.09203v2 : semiring/provenance context; Boolean dependency overapproximation is NOT this exact F2 coefficient extraction.
NVIDIA, Floating Point and IEEE754, https://docs.nvidia.com/cuda/floating-point/index.html : rounding/order distinctions.
