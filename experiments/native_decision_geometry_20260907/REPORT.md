# Native rounding-aware decision geometry — 2026-09-07

Base: PR139,404c80b35f6feffc52bfd215d2a7b0cfd0751c50.
[Replay and full Korean report](README.md), [scoped ledger](LEDGER.md).
This is bounded auxiliary research, not an admitted universal core.

## Constructive result

Reference score: l_i=RN_BF16(RN32(RN32(a_i*x)+b_i)). Return the smallest numeric-maximum ID and its original BF16 score bits. Original BF16 coefficients a_i,b_i are in[-1,1] on a2^-16 grid. Any finite BF16 scalar x, including signed zero and subnormals, is admitted. This domain is not a relaxation of the fixed VORTEX mission. InputNaN/Inf, arbitrary multidimensional native dot products, full-logit interfaces and stochastic sampling are not covered.

Products have at most16 significant bits and no nonzero bit below2^-149, so the specified multiply is exact FP32. R=RN_BF16 o RN32 is monotone; therefore max_i R(s_i)=R(max_i s_i) for exact affine s_i=a_i*x+b_i. This equality is about maximum VALUES, not maximum IDs.

Construct an original-index interval tree with exact upper envelopes, rational breakpoints and serialized coefficients/addresses. Query root native maximum M; descend left when its native maximum also equals M, otherwise right. The leaf is the earliest native tie. Evaluate only that leaf's original arithmetic to recover signed-zero score bits. No reference scan, ideal selector, response enumeration, similar prior input or sparse correction is used at query time. Original coefficient information remains in the index.

Constructor O(V log^2V), source O(V logV), query O(log^2V) rational comparisons/evaluations. Big integers are not free operations. Observed geometry operands reach150bits; rounding internals were not instrumented and have a separately described conservative320bit bound. These are not hardware latency/complete resource upper bounds.

## Evidence

9 registered synthetic matrices: V64/256/1024/4096, seeds1401/1402,256queries each; V16seed1403 exhausts65280 finite BF16 words. Total67328 queries have0 ID/selected-score mismatches against C and NumPy. Exact Fraction checks are also preserved. Two synthetic scalar recurrences independently consume their own output/state/LCG for256steps without mismatch. Two explicit history pairs preserve nonidentity context dependence; not language understanding, HF or KV evidence.

Rounding-tie witness: row0=(0,1/2),row1=(2^-16,1/2),x=1. Real winner1, native winner0. Multidimensional lifting counterexample: A=(2^24,1,-2^24,1),B=(2^24,-2^24,1,1),x=ones: exact sums2/2 but balancedFP32 sums1/2. General exact-real support queries do not automatically preserve native ordering.

## Paid costs and no core admission

V4096 originals16384B; index216912/216944B (~13.24x). Logical query fields plus input/output p50=432/400B,p95=456/440B (~2.78/2.69% of original coefficients). Integer arithmetic, objects, scratch, physical caches, addresses and loading are additional. No latency claim. V64p95=78.125%;V1024p95=8.98/8.79%; all registered groups retained.

Originalread+indexwrite+reload+first8queryfields already cost~3.4612/3.4597x original8 coefficient scans in V4096. Construction arithmetic and temporary memory are additional. Small warm queries are not small startup/storage.

Previously fixed405B dimensions h16384,k1024,f53248,L126,V128256 give head2101346304 and body401646551040 MAC. Making the head entirely FREE still leaves99.47954% of counted projections. This is an optimistic scope gate, not hardware timing. No high-dimensional/backend expansion followed that rejection.

## Other two premise changes

RNG-first probability decision still needs an explicit cheap native prefix/total-mass source. With p=(1/4,3/4), tape=(1/2,1/16), inverseCDF chooses1 while exponential race chooses0. Same distribution is not same-tape behavior. This is a mathematical coupling witness, not a tested HF multinomial implementation.

Current-output state quotient fails continuation: logits=(-h,h), states h=1 and h=2 both choose1; the same transition h'=h-1.5 yields next choices0/1. Current argmax cells alone are not right-congruence classes. No hidden state decoder is assumed.

## Reproduction and status

13tests;51scientificfiles reproduce manifestSHA256 2c19e62c6d751a99cdd5c2c2eef8710d72344decd07da93a1d60edfe3ccb7023. The original22text capsule was restored in an empty folder and replayed. The identical capsule split into4pieces was independently restored/replayed as well. Initial run and the later explicit history-witness addition are retained separately; primary9groups were unchanged. Full Korean proof, preregistration, source/C/tests, fixed summaries, original manifest and retained logs/history are in the capsule. All raw binary arrays/indices/traces are in the user ZIP and regenerate, not all embedded in Git. XZ/base64 is archival, not inference compression.

THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED; CORE_ADMISSION=false; FULL_MISSION_O1_O6=OPEN; THREE_QUALIFYING_NEW_PRINCIPLES=false. No publicmodel,fullHF,fullKV,full-logitAPI,softmaxsampler,CUDA,405B,8GiB,4BQ4,TTFT or whole-repo-suite execution. Checkpoint acquisition failed DNS before weights; infrastructure failure is not scientific rejection. No main push,force,merge or Actions dispatch.

Prior tools: Wang2020 Algorithms for Subpath Convex Hull Queries and Ray-Shooting Among Segments(arXiv2002.10672); NVIDIA Floating Point and IEEE754 Compliance. No claim of first convex-hull/argmax index. Need a cheap nonidentity context-dependent source through the wide body and native arithmetic, not just a faster final decision.
