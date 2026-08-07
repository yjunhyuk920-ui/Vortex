# Next Experiment

## Closed Audit — EXP-071

EXP-071 audited whether two primary unconditional online matrix-vector lower-bound results can rigorously rule out the registered conventional exact VORTEX runtime model.

Source-audited population:

```text
CGL15 Theorem 3
CKL18 Theorem 1.2
CKL18 Theorem 1.3
```

Exact reduction and registered-plan integrity:

```text
1,052,740 exhaustive binary matrix/vector cases
4,164 direct float32 replay cases
reduction mismatches 0
control failures 0
9 Llama-405B tensor families
884 tensor instances
405,849,243,648 registered parameters
```

Applicability result:

```text
largest valid square subproblem n          16,384
CKL18 maximum side-information range       67,108,864 bits = 8 MiB
registered hot/side-information allowance  68,719,476,736 bits = 8 GiB
allowance / theorem range                  1,024x
CKL18-covered tensor families              0 / 9
model-wide direct-sum theorem              not established
finite constants hidden by Omega           unresolved
```

Even the deliberately non-authoritative unit-constant CGL15 indicator, illegally summed as if a direct-sum theorem existed, was only `0.0248972%` of packed Q4 cells. This is diagnostic only and is not a certified finite lower bound.

Decision:

```text
INSUFFICIENT_LOWER_BOUND_DO_NOT_CLAIM_IMPOSSIBILITY
```

This does not establish feasibility. It establishes only that the registered papers do not justify a 405B impossibility claim under the full 8 GiB jointly computed side-state regime.

Authority:

```text
results/exp_071/summary.json
results/exp_071/raw/theorem_hypotheses.jsonl
results/exp_071/raw/tensor_rows.jsonl
results/exp_071/raw/control_rows.jsonl
results/exp_071/processed/direct_sum_audit.json
workflow 30965323458
artifact 8914506737
artifact ZIP SHA-256 bc81e90e3b5a35935f893ad7396d4b41a13de46606ce14bccc53cf79e30e8ba4
```

Permanent restrictions:

- do not divide the shared 8 GiB state by tensor/layer count without a direct-sum theorem;
- do not add independent per-matrix asymptotic lower bounds without a composition proof;
- do not set hidden asymptotic constants to one and call the result certified;
- do not equate cell probes with physical GPU, PCIe, or SSD transactions;
- do not claim that all exact software runtimes are impossible from EXP-071.

## Closed EXP-072A — Self-Contained Exact Q4 DAG Information-Capacity Gate

The previously registered EXP-072 synthesis search is deferred behind a cheaper proof-first Gate.

Question:

```text
Can a self-contained exact arithmetic-DAG artifact encode every arbitrary Q4
dense checkpoint inside the static/hot-state resource envelope without reading
the original coefficients during a query?
```

For `P` Q4 coefficients there are `16^P` distinct linear maps. Exact equality on standard-basis activations uniquely recovers the matrix, so distinct maps require distinct self-contained artifacts. A universal variable-length artifact cap of `B` bits must satisfy:

```text
2^(B+1) - 1 >= 16^P
```

The registered 405B population contains `1,623,396,974,592` Q4 information bits (`188.98828125 GiB`) before scales, bias, opcodes, alignment, interpreter state, or workspace. The fixed 8 GiB hot allowance is only about `4.2331%`; the former `10%` static Gate is not sufficient for final residency.

EXP-072A validated the finite-domain injectivity reduction exhaustively and froze the target arithmetic before any synthesizer was built.

Promotion requires a universal self-contained cap at or below 8 GiB plus a route toward the fully charged `1.185185%` target-equivalent fraction. Failure decision:

```text
REJECT_SELF_CONTAINED_EXACT_Q4_DAG_AS_UNIVERSAL_CORE
RETAIN_RESTRICTED_SYNTHESIS_AUXILIARY
```

This result will not cover a runtime that reads the original checkpoint or another lossless cold representation during a query. That is a different online execution class and must charge every cold-data probe.

Authoritative result:

```text
finite matrices/signatures     272 / 272
signature collisions           0
control failures               0
Q4 information                 188.98828125 GiB
hot allowance                  8 GiB = 4.2331%
required / hot                 23.62353515625x
```

Decision:

```text
REJECT_SELF_CONTAINED_EXACT_Q4_DAG_AS_UNIVERSAL_CORE
RETAIN_RESTRICTED_SYNTHESIS_AUXILIARY
```

Authority: `results/exp_072a/summary.json`; source `468f297925e10bdc541fe48f19c2f72a1e3f5e14`; evidence commit `f9ac26befb01fd9a71c7c6e1efed4c4b4df31389`.

Full preregistration and interpretation: `docs/research/EXPERIMENT_072A_SELF_CONTAINED_DAG_INFORMATION_CAPACITY.md`.

## Archived EXP-072B — Exact Nonlocal Q4 Shared Arithmetic-DAG Synthesis Gate

The synthesis plan below is retained for provenance but is **not active**. EXP-072A prohibits it as a universal core because an exact self-contained hot artifact cannot cover the arbitrary Q4 checkpoint class inside 8 GiB. It may be reopened only as an explicitly restricted auxiliary or after introducing a materially different cold query-time information source with fully charged probes.

### Execution-class change

EXP-070 tested only contiguous short coefficient patterns. EXP-067 tested only whole-row equality/sign/proportional reuse and common-right rank. Neither test covered a general exact straight-line arithmetic program that can share **non-contiguous intermediate linear forms** across many output rows and across projections consuming the same activation.

For symbolic input coordinates `x_j`, a circuit node may form an exact integer linear form:

```text
z_k = a * z_i + b * z_j
```

where `a,b` are bounded registered small integers and the resulting symbolic coefficient vector is verified exactly. Outputs must reconstruct every requested Q4 row as an integer coefficient vector. The circuit may discover reusable forms such as:

```text
x_3 - x_19
2*x_7 + x_41
(x_3 - x_19) + (2*x_7 + x_41)
```

These forms need not correspond to contiguous blocks, identical rows, low-rank factors, or a fixed tensor decomposition.

### Why this class remains eligible

- Q4 coefficients come from a small alphabet;
- many rows consume the same activation vector within one projection or Q/K/V group;
- arbitrary linear circuits can share partial expressions that block dictionaries cannot see;
- exact symbolic reconstruction is possible without training or activation approximation;
- a positive result would expose a true smaller executable program rather than only a compressed byte layout.

Prior probability remains low because general dense matrices usually have high linear-circuit complexity. Therefore the experiment is bounded and must stop before kernel work unless the full Gate passes.

### Registered synthesis scopes

Use the unchanged frozen real-Q4 population and checksums. Evaluate only preregistered scopes:

```text
single projection tiles
complete Q/K/V groups sharing one input
complete Gate/Up groups where present
```

Registered column widths:

```text
16, 32, 64
```

Registered output-row tile heights:

```text
16, 32, 64
```

No post-result tile size, model, matrix role, or selected favorable subset may promote the candidate.

### Registered circuit search

Implement deterministic bounded search families only:

1. pair-frequency common-subexpression elimination over signed linear forms;
2. bounded beam search seeded by the highest-support exact pair forms;
3. addition-chain seeds for Q4 constants `-8..7`;
4. joint-output search for matrices sharing an identical runtime input.

Every candidate node is represented by its full exact integer coefficient vector during compilation. Hash matches must be confirmed by full-vector equality. Cycles, approximate coefficient matches, floating tolerances, and activation-derived search are forbidden.

### Accounting

Report separately:

```text
operation fraction
query-description byte fraction
static circuit-storage fraction
compiler work and peak memory
```

Charge at minimum:

- every runtime add, subtract, small-constant multiply, and output accumulation;
- every required input load not already resident under the registered baseline;
- circuit opcodes, operand IDs, constants, output maps, tile maps, and row scales;
- repeated execution of a circuit template on different activation values;
- cross-tile output accumulation;
- dense fallback for any unsynthesized row.

One plan per scope must minimize the maximum of operation, query-byte, and static-storage fractions. Per-axis cherry-picking is prohibited.

### Exactness controls

- exhaustive symbolic reconstruction for every emitted circuit;
- random activation replay in exact integer arithmetic;
- float32 replay reported separately without promoting changed reduction order to bitwise equivalence;
- Hadamard/butterfly and hand-shared linear-form positive controls;
- forced-unique and dense-random Q4 negative controls;
- one-coefficient mutation must invalidate or alter the expected circuit;
- deterministic reruns must produce identical circuit and evidence hashes.

### Promotion Gate

```text
zero checksum/reconstruction/collision/control mismatch
100% registered population coverage
p50 operation fraction <=10%
p90 operation fraction <=25%
p50 query-description fraction <=10%
p90 query-description fraction <=25%
p50 static-circuit fraction <=10%
p90 static-circuit fraction <=25%
no required matrix role or shared-input group with p90 >25%
no largest-model degradation >25%
```

Passing authorizes only a bitwise floating replay-order Gate and a physical packed-circuit kernel Gate. It does not authorize a 405B or 8 GiB claim.

### Failure decision

```text
REJECT_EXACT_NONLOCAL_Q4_ARITHMETIC_DAG_AS_CORE
RETAIN_LINEAR_CIRCUIT_SYNTHESIZER_AUXILIARY
```

On failure, do not rescue the family by hiding circuit bytes, counting compiler work as free amortization without a query threshold, reporting positive structured controls as real-model evidence, selecting only Q/K/V or only favorable tiles, or using approximate linear forms while claiming exactness.

### Stop rule

Before survival, prohibit:

```text
CUDA circuit kernels
model-wide circuit transcoding
unbounded SAT/SMT synthesis
learned or activation-specific circuits
floating-point semantic claims
405B implementation work
```

### Claim boundary

Phase A/B/C real-Q4 symbolic-circuit evidence, ceiling E1. Floating-point reduction-order equivalence, a physical circuit kernel, actual Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/second remain **NOT TESTED**.

## Active EXP-073 — Private Ubuntu Target Resource-Contract Calibration

### Purpose

Replace proxy hardware assumptions with a sanitized, reproducible envelope from the privately identified Ubuntu host before another cold-backed core candidate is selected.

This is a Phase-D **calibration prerequisite**, not evidence that any VORTEX mechanism works. Private hostnames, addresses, usernames, keys, and internal paths must never enter this public repository or uploaded artifacts.

### Stage 1 — COMPLETE: read-only inventory

Collect without installing or restarting anything:

```text
OS/kernel and CPU
total/available host RAM
GPU model, compute capability, total/usable VRAM
driver and CUDA runtime compatibility
PCIe link generation/width when exposed
filesystem and block-device type
available local capacity
existing Python/Ollama/runtime versions
fio availability
thermal and power telemetry availability
```

Success means a complete sanitized inventory with zero mutation. Missing tools are recorded as `NOT AVAILABLE`, not installed during Stage 1.

Measured result:

```text
decision                         COMPLETE_SANITIZED_READ_ONLY_TARGET_INVENTORY
GPU                              Quadro M5000, 8,192 MiB, compute 5.2
VRAM free at snapshot            8,058 MiB
PCIe exposed current / maximum   Gen1 x16 / Gen2 x16
host RAM total / available       23.4983 / 21.8865 GiB
local block storage              238.4749 GiB, non-rotational ATA
root filesystem total / free     233.6702 / 97.6183 GiB, ext4
Python / Ollama                  3.12.3 / 0.30.6, service active
fio / nvcc                       NOT_AVAILABLE / NOT_AVAILABLE
power / temperature / clock      telemetry interfaces available
```

Derived resource consequences:

```text
registered packed 405B Q4 information     188.9883 GiB
root free-capacity deficit                  91.3700 GiB before overhead
favorable PCIe Gen2 x16 ceiling              7.4506 GiB/s
full packed-Q4-equivalent transfer floor    25.3656 s before overhead
```

Authority: `results/exp_073/summary.json`; source `d3b1d2e4dd08e73781c969814cb4d181377a054d`; checkout-stable evidence `4e35afb9648dc0c513c3a90c32d604f4c0b0fd21`; core SHA-256 `aa9cae0457a6b92fcb75da35fedc1a2a2a9f3da115808d341a5a498ca4722da2`.

### Stage 2 — PENDING SEPARATE AUTHORIZATION: baselines

Only after Stage 1 review and separate authorization:

```text
cached and direct sequential reads
4 KiB and 64 KiB random reads
host-to-device transfer bandwidth
native 4B Q4 cold/warm TTFT
native 4B Q4 p50/p95/p99 time per token
peak VRAM, host RSS, page faults, power, and thermal state
```

Use an already present 4B-class model when possible. Do not download a 405B checkpoint, install packages, restart services, or evict production workloads without explicit authorization.

### Gate and handoff

Stage 1 passes its inventory Gate. EXP-073 as a whole passes calibration only when the same-machine 4B baseline and storage/transfer envelope are reproducible and raw evidence is sanitized. It cannot promote a runtime candidate.

The next core E0 candidate must then state, using measured values:

```text
maximum hot state
maximum cold state and preprocessing budget
allowed physical bytes/token
required reuse/amortization factor
serial latency floor
driver/kernel compatibility constraints
```

If the host cannot run a valid native 4B Q4 baseline or required profilers without disruptive changes, record `INFRASTRUCTURE LIMITATION — NO SCIENTIFIC DECISION` and do not substitute projections.

## Closed EXP-074 — Weight-Stationary MTP/Expert Block Budget Gate

EXP-074 placed the proposed Qwen3.5-122B-A10B surrogate behind a no-download
logical resource Gate before any checkpoint or paging backend work.

The standard MTP-1 plus expert-paging path failed even under a zero-cost causal
proposal and a fixed routed-expert set:

```text
active target equivalent                         10B
native baseline equivalent                        1B
MTP-1 optimistic normalized traffic             10.0x
p50 allowance                                     1.2x
```

The broader block form is not yet rejected because an ideal fixed expert set
and free proposal reaches p50 at nine perfectly accepted tokens. Fully charged
or route-diverse controls move the requirement substantially:

```text
fixed routes + 0.8B draft                        25
fixed routes + 1.0B draft                        50
independent-uniform route expectation             98
maximally distinct route control                 102
```

Decision:

```text
REVISE_MTP1_AND_EXPERT_PAGING_INSUFFICIENT_REQUIRE_LONG_CAUSAL_PROPOSAL_AND_ROUTING_LOCALITY_GATES
```

No server command, model download, inference, or physical measurement occurred.

Authority: `results/exp_074/summary.json`; source
`8abc06e73c884b839927cf41d5f4fa6cbb8fc051`; evidence
`c1d778af011672ec7fadfa66935ba2548de8e115`.

## Candidate next Gate — Native MTP Surface and Accepted-Prefix Audit

Do not open a model execution branch until a metadata-only checkpoint audit
proves that the selected unchanged small Qwen3.5 distribution contains the MTP
tensors and that the chosen runtime/quantization path preserves and exposes
them.

If that audit survives, the cheapest real-model Gate is:

```text
smallest official unchanged Qwen3.5 checkpoint
pinned revision and file hashes
causal MTP proposals only; no target future tokens
K in 2, 4, 8, 16, 32, 64
exact longest-prefix target verification
proposal, LM-head, verification, rejected-position and fallback cost charged
p50/p95 normalized traffic reported directly
```

Rejection occurs if the favorable exact-reference accepted-prefix distribution
cannot reach the EXP-074 fully charged threshold or if the public checkpoint
does not expose usable MTP state. Only a surviving proposal Gate can authorize
a 35B-A3B expert-route trace. A 122B download remains prohibited at this stage.

## Closed EXP-075 — Native MTP Checkpoint and Runtime Surface Audit

EXP-075 audited the smallest official unchanged Qwen3.5 checkpoint without
downloading its weights. The pinned config declares one MTP layer, the weight
index contains the exact 15 registered `mtp.*` tensor keys, and pinned vLLM
source exposes the corresponding configuration rewrite, model registry, weight
loader, and recursive speculative-step path.

```text
metadata/source files                         6
metadata/source bytes                   255,779
MTP tensor keys                          15 / 15
runtime source surfaces                    3 / 3
controls                                    5 / 5
declared checkpoint size            1.626911 GiB
```

Decision:

```text
PROMOTE_TO_PINNED_QWEN35_08B_ACCEPTED_PREFIX_GATE
```

Authority: `results/exp_075/summary.json`; source
`f85ac583a129070247992987d1b3c63634e6447f`; evidence
`2fcf7315cf9da491a5ca361536eb0f07e325e74c`.

This promotes only the next small-checkpoint falsification. No weight payload,
inference, runtime installation, quantization path, or server command occurred.

## Candidate EXP-076 — Qwen3.5-0.8B Native MTP Accepted-Prefix Gate

Use the unchanged checkpoint and revision selected by EXP-075. Before download,
commit the exact file manifest, dependency lock, prompt hashes, state semantics,
and accounting formulas. The currently present local Transformers 4.50.3 is
older than the checkpoint's declared 4.57.0 development format, so dependency
compatibility must be isolated and pinned rather than silently assumed.

Required causal protocol:

```text
exact greedy committed prefix
-> one unchanged-target state/hidden output
-> native MTP recursively proposes K in 2,4,8,16,32,64
-> proposal never reads target future tokens
-> unchanged target computes exact verification continuation
-> measure longest matching prefix and first correction
-> discard rejected suffix and restore every hybrid/KV state exactly
```

Use disjoint prompt families including English, Korean, code, structured JSON,
math, and adversarial low-acceptance cases. Report configured proposal length
separately from accepted prefix. At minimum publish p05/p50/p95 accepted length,
zero-accept rate, per-position acceptance, exact token agreement, target and MTP
forward counts, parameter-by-shape proposal/LM-head/verification cost, RSS, CPU
time, exclusions, and all fallback work.

The reference EXP-074 fully charged fixed-route thresholds require at least 25
accepted tokens at the median and at least 15 at the fifth percentile when a
0.8B-equivalent proposal is charged. Actual shape-derived MTP cost may change
the arithmetic but may not be omitted. Rejection occurs on any causal leakage,
state rollback mismatch, silent wrong accept, required-family failure, median
below the recomputed p50 threshold, fifth percentile below the recomputed p95
threshold, or a favorable fully charged ceiling outside the Gate.

Passing authorizes only a 35B-A3B metadata/route-trace Gate. It does not
authorize 122B download, page scheduling, Phase D, or a dense-405B claim.
