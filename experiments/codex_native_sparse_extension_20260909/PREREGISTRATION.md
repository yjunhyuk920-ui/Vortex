# Restricted native producer extension — 2026-09-09

Base bbba0c2298785e9737d916c7a3ccc64bc471bbaa. This continues the surviving
restricted E1 producer at experiments/causal_global_bridge_20260908, following
NEXT_EXPERIMENT's explicit source-alphabet/right-factor expansion obligation.
It is not a new three-principle core round and cannot narrow the full mission.

Comparison within the existing A/B/C frontier: a global nonlinear producer (B)
and arbitrary dense-change dynamic state (C) still have no concrete sufficient
decoder. Continue the actual restricted causal producer (A) by removing its
binary-weight and single-basis-input assumptions. Preserve the all-checkpoint,
all-continuation mission and all failed general dense-delta/catalog/replay paths.

The intended local theorem addresses O1/O2/O3/O4: compile arbitrary finite BF16
matrix words into paid columns and row zero-sign metadata; for BF16 inputs with
at most two nonzero coordinates and a fixed declared sign pattern on inactive
zeros, read only selected columns and emit the exact original balanced FP32 RNE
projection (including signed zero, overflow/NaN results) and BF16 RNE store.
All loops and metadata/probes/initialization/verification/fallback costs count.

Native scalar ABI: finite BF16 operands; separate FP32 RNE products; fixed
balanced FP32 RNE additions with +0 padding; canonical quiet NaN for invalid
infinite cancellation; optional BF16 RNE store. No arbitrary HF/CUDA ABI claim.

Compiled metadata stores each row's count of positive-zero products under the
fixed zero sign pattern, with padding accounted. Runtime subtracts active-slot
counts, evaluates selected products at their original tree slots, and restores
-0 only when every actual product/padding leaf is -0. Skipping zero subtrees
otherwise uses +0. This rule needs a proof and adversarial validation; it is not
accepted from sparse-input intuition alone. Nonfinite source words, unsupported
zero signs, or support above two are refused, not silently approximated.

Frozen scalar controls: all finite BF16 source words on one selected coordinate
with fixed dyadic input patterns; fixed matrices/inputs stressing +/-0, positive
and negative tiny values, max finite values, overflow/cancellation and non-power
of-two widths. Compare against a full native-order reference. Use no larger
randomized search after a pass. Report coefficient reads, metadata reads,
product/add counts, compile/storage and input/output terms at 16384 square,
k=1,2 and dense k=n fallback. The dense-input runtime remains unclosed.

Actual HF CPU extension: a normal LlamaForCausalLM fixture built from ordinary
checkpoint tensors (not an altered public target), with two-sparse +/-1 token
embeddings, unit norms, zero q/k/o/MLP/lm_head and independently generated bounded
BF16 v_proj entries. Freeze source exponent range [-8,8], mantissa patterns and
seed 260909 before running. Use hidden=32,value=16,head_dim=2,layers=2, legal
tokens all 32 plus a fixed repeated continuation. Source range is deliberately
bounded in this HF fixture to avoid nonfinite attention/residual contamination;
the scalar producer's finite-BF16 domain is broader and separately stated.
The compiled 64-position RoPE template bounds this experiment's supported
context; it is not a claim that the HF API rejects all longer continuations.
The full mission's all-legal-continuations obligation therefore stays OPEN.

HF fixture details before its first run: Python Random(260909) independently
draws sign bit, exponent uniformly in [-8,8], and mantissa uniformly in [0,127]
for each source word; every 37th linear entry is signed zero with the drawn
sign. Token j has +1 at j and (-1 if j is odd else +1) at (j+1) mod 32.
Manual legal trace is list(range(32))+[31,0,17,17,3,25,0,31]. Generation control
uses prefix [0,7,31], eight new tokens, do_sample=True, top_k=0, top_p=1,
temperature=1, no EOS stop, and the same torch RNG state before native and
compiled sampling. The native generate result/cache/RNG is authoritative.

At compile time pay full checkpoint validation/copy, native two-support RMSNorm
metadata and position-dependent zero-key RoPE template. At runtime no original
dense v_proj/model forward may be called: selected columns produce V, native
DynamicCache.update appends exact K/V, and the fixed zero head supplies logits.
Compare the native CPU model's logits and all cache words after every step,
checkpoint before/after hashes, native and compiled RNG states, and actual
sampled continuation using the same native sampling procedure/state. Full
prefill versus incremental agreement is tested separately and not assumed.

A mismatch stops promotion and is retained with the first failing word. Do not
change the checkpoint or thresholds after seeing it. A fix to an actual producer
bug needs a documented diagnosis and fresh immutable result version. Any native
ABI gap remains OPEN. No CUDA, 405B, <=8GiB GPU, native4BQ4 latency or TTFT run
is possible here; no HF-wide/native-universal proof follows from this fixture.

All CPU/RAM/SSD/PCIe/HBM/GPU/KV, compile/generation/storage/movement/arithmetic,
metadata, initialization, validation, recovery and fallback terms remain paid.
Sparse local admission is not arbitrary dense admission. CORE_ADMISSION=false;
THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED; O1-O5 OPEN/O6 PARTIAL.
