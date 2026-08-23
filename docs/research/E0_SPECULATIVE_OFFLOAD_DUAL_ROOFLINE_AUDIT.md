# E0 Speculative-Offload Dual-Roofline Audit

## Status

```text
Phase: E0 / model-free target equation
Evidence ceiling: E0 plus source-audited literature observations
Target hardware execution: NOT TESTED
405B execution: NOT TESTED
Scientific decision:
  RETAIN_LOSSLESS_OFFLOAD_AND_OVERLAP_AS_AUXILIARY
  REQUIRE_FINE_DENSE_ARITHMETIC_REDUCTION_FOR_CORE
```

This round asks whether the newly published combination of lossless weight
compression, parameter offloading, transfer/compute overlap, and deep
speculative verification changes the fixed 405B/8-GiB mission enough to promote
a new core executor.

The answer is conditional but decisive for the next experiment: these systems
form a useful **transport shell**, but accepted length alone cannot promote a
core while exact target verification still performs one dense checkpoint
column per verified candidate node.

## Fixed mission boundary

The audit preserves the repository contract:

```text
checkpoint       arbitrary unchanged public HF dense transformer
flagship         405,849,243,648 parameters
weight ABI       exact BF16 words for this favorable transport screen
GPU hot state    <= 8 GiB
quality/state    exact token and exact-or-bisimilar successor state
p50 target       <= 1.2 x native same-machine 4B Q4 p50
p95 target       <= 1.5 x native same-machine 4B Q4 p95
training         forbidden
hidden compute   forbidden
fallback         charged
```

The registered target inventory is retained without pretending it was measured
in this session:

```text
GPU                         Quadro M5000, 8 GiB, compute capability 5.2
favorable link ceiling      PCIe Gen2 x16 = 7.4506 GiB/s
favorable device peak       4.3e12 FP32 operations/s projection
registered root free space  97.6183 GiB
native 4B p50/p95           NOT TESTED
```

Using the published FP32 peak for exact BF16 work is deliberately favorable.
The M5000 has no modern BF16 Tensor Core path, so this is a lower bound, not a
performance prediction.

## Source-audited 2025-2026 research surface

### SubSpec

SubSpec is directly relevant: it is training-free, constructs low-bit
substitute layers for offloaded target layers, shares resident layers and KV,
and verifies a large speculative tree in one target pass. Its paper reports a
Qwen2.5-7B 8-GiB result near 25 tokens/s, 288 verified draft tokens in its
implementation discussion, and acceptance lengths around 27-30 for the
reported Qwen2.5-7B settings.

Primary source: *Speculate Deep and Accurate: Lossless and Training-Free
Acceleration for Offloaded LLMs via Substitute Speculative Decoding*,
arXiv:2509.18344, NeurIPS 2025.

### SpecOffload

SpecOffload interleaves target offloading and speculative drafting, placing the
draft model in otherwise latent GPU capacity. It reports 4.49x GPU utilization
and 2.54x throughput over its best baseline. This validates overlap and
placement as a systems shell; it does not remove exact target verification
arithmetic.

Primary source: *SpecOffload: Unlocking Latent GPU Capacity for LLM Inference
on Resource-Constrained Devices*, arXiv:2505.10259.

### CATS

CATS is the closest flash/DRAM analogue. It streams a shallow draft/verifier and
then verifies a correction tree with the remaining target layers. Under its
default five draft steps, the Vicuna-7B MT-Bench greedy acceptance length is
3.052. Its adapters are trained with a Reduced KL objective, including 68,000
ShareGPT samples; therefore its measured core is not admissible under the fixed
no-training mission. The architecture remains useful evidence about candidate
inflation and memory-limited scheduling.

Primary source: *CATS: Cascaded Adaptive Tree Speculation for Memory-Limited
LLM Inference Acceleration*, arXiv:2605.11186.

### ZipServ and Shannon-bound lossless coding

ZipServ decodes losslessly compressed weights directly into Tensor Core
registers and reports up to 30% model-size reduction. Its Llama-3.1-405B result
is a matrix-shape kernel benchmark, not a complete 405B/8-GiB executor. The
2026 Shannon-bound study measures Llama-405B BF16 at about a 1.5x lossless
ratio, with effective BF16 entropy around 10-12 bits/weight. This audit grants
the full favorable 1.5x ratio.

Primary sources:

- *ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware
  Lossless Compression*, arXiv:2603.17435, ASPLOS 2026.
- *Approaching Shannon Bound with Lossless LLM Weight Compression*,
  arXiv:2606.15789.

## The missing equation in accepted-length-only arguments

Let:

```text
P  = 405,849,243,648 parameters
c  = exact lossless compression ratio
B  = host-to-device bandwidth
F  = favorable device operation rate
N  = target-verified candidate positions/nodes in one cycle
A  = exact committed tokens from that cycle
r  = fraction of original dense target arithmetic retained per candidate
```

`N` includes any dense column needed to materialize the committed successor
state. A speculative “bonus token” is not free under this ABI: if it is counted
in `A`, its exact KV/cache successor state must also be produced and charged.
Therefore the favorable exact-state accounting retains `N >= A`.

The most favorable complete-cycle lower bound grants perfect overlap and makes
all draft, attention, KV, synchronization, decoding, and branch-management work
free:

```text
S_c              = 2P / c bytes
T_io             = S_c / B
T_dense_candidate= 2P / F
T_cycle          >= max(T_io, r * N * T_dense_candidate)
T_token          >= max(T_io / A, r * (N/A) * T_dense_candidate)
```

This separates two independent requirements.

### I/O Gate

For native 4B p50 `L_4B`:

```text
A >= T_io / (1.2 * L_4B)
```

Accepted length amortizes checkpoint traffic.

### Dense-arithmetic Gate

```text
r * (N/A) <= 1.2 * L_4B / T_dense_candidate
```

Accepted length does **not** reduce dense arithmetic when `N=A` and `r=1`.
A tree with rejected branches has `N/A>1`, so it makes the dense floor worse.
This is why multiplying a compression speedup by a speculative speedup is not a
valid target proof.

## Registered-target result

With `c=1.5`, `B=7.4506 GiB/s`, and `F=4.3e12`:

```text
raw exact BF16 checkpoint              755.953125 GiB
favorable compressed checkpoint        503.968750 GiB
current registered storage deficit     406.350450 GiB
one compressed target-sweep I/O floor   67.641364 s
one dense candidate arithmetic floor     0.188767 s
I/O/compute crossover                   358.332400 nodes
first integer crossover chain           359 tokens
```

Even the perfect chain `N=A` cannot satisfy p50 unless the unmeasured native
4B p50 is at least:

```text
0.188767 / 1.2 = 157.305908 ms
```

This is not an impossibility claim because the same-machine native 4B baseline
has not been measured. It is a hard prerequisite. At any faster 4B baseline,
`r` must fall below one: the target mechanism must actually remove fine dense
arithmetic, not merely batch it.

Examples from the deterministic audit:

| Diagnostic cycle | N | A | N/A | optimistic floor/token | required native 4B p50 |
|---|---:|---:|---:|---:|---:|
| perfect exact chain | 128 | 128 | 1.000 | 528.448 ms | 440.373 ms |
| EXP-091A raw threshold | 128 | 85 | 1.506 | 795.781 ms | 663.151 ms |
| EXP-091A compressed threshold | 128 | 67 | 1.910 | 1,009.573 ms | 841.311 ms |
| SubSpec cross-system diagnostic | 288 | 27.08 | 10.635 | 2,497.835 ms | 2,081.529 ms |
| CATS favorable five-node minimum | 5 | 3.052 | 1.638 | 22,162.963 ms | 18,469.136 ms |
| perfect crossover chain | 359 | 359 | 1.000 | 188.767 ms | 157.306 ms |

The SubSpec and CATS rows are deliberately transferred as diagnostics only.
They are not target projections, because their checkpoints, hardware, target
precision, and verification kernels differ. Their purpose is to show that the
published `N/A` regime is far from the near-one chain required by the
registered compute floor.

## Draft-residency Gate

A whole-model substitute for 405B requires:

```text
4-bit substitute  188.988281 GiB
1-bit substitute   47.247070 GiB
```

The complete 8-GiB grant can encode only:

```text
0.169322668 whole-model bits/parameter
```

before KV, activations, scales, code, allocator, or target-resident layers.
Therefore SubSpec's fully GPU-resident substitute-layer design does not scale
as written to the 405B/8-GiB endpoint. A future implementation needs a
materially smaller causal source, not merely a lower substitute bit width.

## Decision

```text
RETAIN_LOSSLESS_OFFLOAD_AND_OVERLAP_AS_AUXILIARY
DO_NOT_PROMOTE_SPECULATIVE_TREE_WITH_UNREDUCED_FINE_DENSE_ARITHMETIC
EXP_097A_MUST_REDUCE_FINE_ARITHMETIC_AND_COMMIT_EXACT_SUCCESSOR_STATE
```

The new literature changes the transport architecture but does not supply the
missing VORTEX core. ZipServ/Shannon coding may reduce cold bytes. SubSpec,
SpecOffload, and CATS may hide draft cost and amortize one stream. None of the
source-audited results establishes the simultaneous endpoint:

```text
arbitrary unchanged dense 405B
single 8-GiB GPU
same-machine 4B-class p50/p95
exact token and successor state
no training or hidden compute
```

## Required next Gate for EXP-097A

The active causal coefficient/codebook continuation may proceed only under this
ABI:

1. The generator emits every finite coefficient/certificate word before target
   continuation and without target-token leakage.
2. The run records both committed tokens `A` and every target-verified candidate
   node `N`; accepted length without `N` is invalid evidence.
3. The exact compressed-source bytes satisfy `S/(B*A)` under the measured native
   4B allowance.
4. The remaining fine arithmetic fraction `r` satisfies
   `r*(N/A)*(2P/F)` under the same allowance.
5. The committed token, KV/cache state, RNG state, and next successor state are
   exact or formally bisimilar under the frozen ABI.
6. Hot state, local source storage, build amortization, verification, misses,
   repair, and fallback are all charged.

A causal coefficient generator that still needs the complete `100%` fine sweep
is an auxiliary certificate generator, not a core executor.

## Reproduction

```bash
python experiments/e0_speculative_offload_roofline/run_audit.py \
  --config experiments/e0_speculative_offload_roofline/config.json \
  --output /tmp/e0-speculative-offload-summary.json

diff -u \
  results/e0_speculative_offload_roofline/summary.json \
  /tmp/e0-speculative-offload-summary.json

python -m unittest discover \
  -s tests/e0_speculative_offload_roofline \
  -p 'test_*.py' -v

sha256sum -c results/e0_speculative_offload_roofline/checksums.sha256
```

## Claim boundary

This audit is a deterministic, favorable target equation. It did not execute a
405B checkpoint, did not measure the M5000, did not measure native 4B p50/p95,
and did not implement ZipServ, SubSpec, SpecOffload, or CATS. Its conclusion is
restricted to the fact that transfer amortization and dense candidate arithmetic
must pass simultaneously.
