# Native row-frontier producer (2026-09-08)

Base: PR145 bc72b62e742a04557c042a7af30f3c569ddac39c. Fixed mission/CTC unchanged. CORE_ADMISSION=false; THEORY_STATUS=NOT_ESTABLISHED; target GPU/HF/405B/8GiB/native4BQ4/TTFT not tested. This is a bounded numerical bridge to known Hamming-coherence algorithms, NOT three qualifying new principles or a completed mission theory.

## Executable construction and exactness

Input is the ORIGINAL finite BF16 matrix W and any current finite BF16 vector x. There is no limited activation alphabet, previous-query similarity, oracle, response bank or low-rank output assumption. Declared original ABI: BF16 expansion, separate FP32-RNE products, +0 padding to power-of-two N, fixed balanced FP32-RNE tree, canonical positive qNaN after operations, final BF16 RNE. Input NaN/Inf, exception flags, arbitrary vendor NaN payloads and CUDA reduction ABIs are outside this declaration.

Build exact ordered subtree signatures by interning pairs of child IDs; no unverified hash collision. For rows p,q let H be differing coefficient leaves and A their union of internal ancestors. D=4H+12A is a nonnegative sum of discrete metrics, hence satisfies triangle inequality. Construct an exact Prim MST under D and its root0 preorder. Shortcutting gives tour(D)<=2 MST(D)<=2 optimal root0 path(D). This is a bound for that metric, not hardware time.

Store first row, then for each next row its ORIGINAL output ID and sorted (column,new BF16 rawword) edits. Replaying the source reconstructs W exactly; compiler checks that. A query initially computes the first row's whole native heap. Replace edited leaves using CURRENT x, then recompute the union of ancestors bottom-up, once per node. No row heap copies, answer lookup tables or source W pointer in candidate C. Unchanged subtrees have identical coefficients and current x; changed nodes perform identical ordered operations on identical children. Induction establishes the entire heap and FP32/BF16 output for every row. The proof applies to all finite input vectors in this ABI, not just tested vectors. All finite matrices in the uint16 column-address format compile; poor compression causes expense, not refusal or retry.

If a caller really uses this exact ABI and consumes these outputs, substitution preserves its observable continuation. Full Transformer/KV/RNG correspondence was NOT implemented or tested. A stateless projection theorem is not a complete causal program.

## Complete local cost model and limitations

For m rows, n columns, N padded width and tour totals E,A:
- original: mn products + m(N-1) additions;
- candidate: n+E products + (N-1)+A additions;
- source B=16+2n+8(m-1)+4E bytes;
- explicit C heap8N + index scratch8n bytes, plus source, inputs, outputs, code, allocators and wrappers;
- counted logical bytes B+8(n+E)+12(N-1+A)+10m. Index scratch/padding/cache-line/library traffic adds further cost. Denominator in reported ratios is original WEIGHT bytes2mn, not total reference traffic.

E<=n(m-1), A<=(N-1)(m-1): FP operation count cannot exceed the direct reference, but addresses and interpretation can make execution slower. Original weights remain preserved externally; their information is in edited source values. Source-size ratios do not measure peak memory or GPU latency.

At fixed row order, t arbitrary coefficient changes increase E by at most2t and A by at most2t log2(N). This gives robustness of the representation, NOT generic small E,A. The actual compiler consumes full W, O(mN) signature storage and Python interning objects, O(m^2 N) comparisons, MST and full replay verification. At n=m=512 it performs268173312 signature-word comparisons. It does NOT implement the faster theoretical nearest-neighbor preprocessing from the cited paper. (2mn+10B)/(8*2mn) counts initial W read/source write/source load/eight source reads only; generation and runtime work are additional. It is~14.35-14.65% for coherent controls,~19.94-20.02% for1%corruption,~262.54-262.57% for random controls. No short-session or TTFT closure.

## Preregistered results

18 square matrices: n32/128/512, seeds17/29, permuted dense prefix-sign controls with per-column BF16 values, their EXACT1%mantissa-bit perturbations, independent random BF16 controls. These are synthetic originals, not modified public models. Compiler gets no planted row ordering.20vectors per case:16random moderate, +/-0, alternating minsubnormal/maxfinite.

All18 compile and all360queries return all80640FP32/BF16 coordinates, with0mismatch versus separately coded C and NumPy.76692FP32coordinates finite; nonfinite outputs are counted separately, not useful-model successes.378coordinates match independent Fraction IEEE rounding.14tests additionally include all65280finite scalar inputs at4weights, nonpower2, signedzero, overflow/underflow, parser checks, triangle inequality and MST bound. Scalar test is not an all-vector-input exhaustiveness claim.

512 results (original524288B):
- coherent: source7748-9032B(1.48-1.72%); FP arithmetic1.26-1.36%; counted logicalbytes16.69-17.81%;
-1%perturbed: source31204-31556B(5.95-6.02%); FP arithmetic8.14-8.21%; counted logicalbytes99.12-100.05%;
-random: source1048740-1048876B(~200%); FP arithmetic~99.86%; counted logicalbytes~1198.7%.
Heap+indices+expandedinput+FP32/BF16outputs at512=13312B of explicit arrays, NOT total process or GPU memory.

A POST-HOC CPU diagnostic, not an acceptance protocol, uses batch1 calls,16moderate inputs,8measured passes=128pairs/case after warmup, alternating function order, checking both outputs. Sources already RAM-resident; compile/load excluded; ctypes and function malloc/free included. No affinity/frequency control. Baseline is handwritten fixed-order C, NOT optimized BLAS, HF or native4BQ4. Ratios of measured medians: coherent0.05050/0.04855;1%perturbed0.12922/0.13010;random1.58802/1.61857. Thus conditional~20x/~7.7x warm kernel observations do not establish any target GPU or full-model gain. CPU summary and benchmark code are in capsule; all raw timing samples are in user ZIP and are not deterministically reproducible.

## Other producers and scope

B scalar row correction: old W=[2^24,1,1,-1], new W=[0,1,1,-1], x=ones. Old native root2^24 plus delta-2^24 gives0; actual new native root1. Original heap updates return1. This rejects uncorrected scalar transport only.
C rectangle prefix: x=[2^24,1,1,-2^24], row=[0,1,1,0]. Native balanced output2 but sequential prefix P3-P1=0. Exact-real range-sum identities are not native rounding proofs. Other range data structures are not ruled out.

Related primary sources: Kozma/Opler arXiv:2602.20023v1 (2026-02-23), https://arxiv.org/html/2602.20023v1, Lemma1.6 and robustness; NVIDIA https://docs.nvidia.com/cuda/floating-point/index.html sections2.2-2.3. We did not reproduce the paper's full twin-width algorithm or prove real pretrained weights have this structure.

## O1-O6 and next obligation

O1: projection constructor exists, generic cheap HF constructor missing. O2: full-vector stateless output only. O3: local numeric induction proved; real HF ABI/KV/RNG missing. O4: local layout, bounds and observations recorded; whole-system cost closure missing. O5: target memory/latency closure missing. O6: bounded code/C/Fraction/replay evidence, not full-theory certificate. All mission obligations OPEN; THREE_QUALIFYING_NEW_PRINCIPLES=false.

Do not make larger planted matrices, MST tuning or a CUDA port the next core result. Need a cheap exact-effect producer for original diverse coefficients even when rows differ almost everywhere, or a genuinely different principle. Generic data is correct but almost full work. Random synthetic matrices are not all real models; nor are structured controls evidence of generic feasibility. Prior native_cover local artifacts are not silently included. Preserve older scopes and fixed mission.

## Reproduction and preservation

python experiments/native_row_frontier_20260908/restore.py --output /tmp/nrf-replay --run

11files restored from checksummed XZ JSON: source/C/tests/run/verify/benchmark/prereg/CPUsummary/validation. Requires Python,NumPy,GCC. verify rebuilds C,14tests and147sciencefiles, and compares canonical manifest SHA256 e3018ea424d0e3e2b6b91f20bc3f579eabc33e3f1df6635762d2ad17081eec42. The manifest is reconstructed, not an unverified new freeze. Full source bytes and deterministic raw W/input/output/source/metrics regenerate. XZ SHA256 ef1eee9827cbd62bb8f48af958154b0f05c23f87107db66c2b8219e9c0b919e2; JSON SHA2561e578851f98173c435b7b5fe83b2f394ead47da4a24f2eadc89f37cb660006f0. Capsule compression is archival packaging, not model compression. Detailed Korean report, all original CPU samples and available logs are in user ZIP; not all raw files/logs are embedded in Git. No Actions dispatch/fullrepositorysuite/main/force/merge. Old four root blobs preserved unchanged in history. Persistence is separate from scientific success.
