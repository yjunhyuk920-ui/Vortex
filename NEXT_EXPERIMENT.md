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

## Closed EXP-076 -- Qwen3.5-0.8B Native MTP Accepted-Prefix Gate

The canonical unchanged-checkpoint CPU reference selected `K=4`. Held-out
accepted-prefix p05/p50/p95 was `0/4/4`; two of 18 prompts accepted no first
proposal token. Shape-derived minima were p05 `9` and p50 `11`. The p50
accepted prefix therefore missed the necessary bound, the p05 bound failed at
zero, and fail-closed realized traffic failed. All registered causality,
acceptance, committed-cache, and rollback controls passed.

Decision:

```text
REJECT_NATIVE_MTP_LONG_BLOCK_AS_SURROGATE_CORE
```

Authority: `results/exp_076/summary.json`; source
`5e331137f8e03250cc74aa796abbf69f49ef87a5`; evidence
`55b79937c1f21887ae76b7e56ad61ba7dde8322a`.

Permanent stop rule: do not rescue this branch with another K sweep, selected
prompts, sampling, quantization, another small Qwen size, a 35B/122B download,
router tracing, or a physical page scheduler. None addresses the measured
accepted-prefix deficit.

## Superseded reset boundary -- portfolio reset and physical calibration

No new core implementation is authorized immediately after EXP-076. A proposed
EXP-077 must first identify a materially different query-time information
source, demonstrate a favorable fully charged route toward `1.185185%`, and
explain why it is not static compression, local equality/reuse, temporal replay,
absolute-unread certification, fixed-point generation, external drafting, or
native MTP under a new name.

The only already specified next measurement is EXP-073 Stage 2 on the private
Ubuntu target:

```text
native 4B Q4 cold/warm TTFT and p50/p95/p99 decode time
peak VRAM and host RSS
bounded sequential/random local-storage reads
host-to-device transfer bandwidth
power, clock, temperature, and loaded-link state
```

Stage 2 requires separate explicit authorization and a fresh preregistration;
it may not install packages, restart Ollama, download a large checkpoint, or
disturb existing workloads without narrower permission. Passing calibration
does not promote a VORTEX mechanism. Until either Stage 2 is authorized or a
new candidate passes E0, the honest core status is `NO_SURVIVING_CANDIDATE`.

## Closed EXP-077A -- Activation-Informed Fractal MLP Oracle Gate

EXP-077A supplies the required materially different query-time information
source: the current causal token's nonzero post-SiLU SwiGLU intermediate
contributions, weighted by unchanged down-projection column norms. It passed
only the paper E0 admission scorecard; no model result exists yet.

The pinned unchanged Qwen3.5-0.8B checkpoint will test whether retaining at
most 10% of each MLP's intermediate channels preserves target logits on the
frozen EXP-076 trajectories. The registered 10% Gate requires at least 99%
held-out top-1 agreement, at least 95% in every family, mean target-to-candidate
KL at most 0.02, and p95 KL at most 0.05.

The selector is a deliberately favorable, non-deployable oracle: it computes
the full current MLP intermediate and receives its cost for free. A pass would
authorize only an all-operator and deployable-selector cost Gate. A failure
closes this activation-norm fracturing score without post-result rescue. No new
checkpoint, 35B/122B payload, target-server command, or physical kernel is in
scope. Contract: `docs/research/EXPERIMENT_077A_ORACLE_FRACTAL_MLP_GATE.md`.

The corrected authoritative causal-cache replay had zero mismatch across 192
registered target positions. At the realized `9.988839%` MLP fraction,
held-out top-1 agreement was only `71.5278%`; mean/p95 KL was
`0.884161/3.080752`. Every family missed its 95% top-1 Gate. Even the 20% arm
reached only `83.3333%` top-1 agreement.

Decision:

```text
REJECT_ACTIVATION_NORM_FRACTAL_MLP_10PCT_PATH
```

Authority: `results/exp_077a/summary.json`; source `33fed17`; evidence
`0970c66`; core
`e25083693c6db21a0da16c9305816958b149865f2fcfde0bd9b7e95c43022411`.

## Next research Gate -- new dependency, not a Fractal score rescue

No core candidate survives. Do not sweep fractions, layers, prompts, channel
blocks, trained routers, or larger Qwen checkpoints around EXP-077A. The next
candidate must introduce a different execution dependency that explains how
the omitted nonzero MLP contribution is recovered or amortized while charging
selector, correction, attention/DeltaNet, LM-head, fallback, and cold traffic.
It must pass an E0 final-fraction route before implementation. EXP-073 Stage 2
remains a separately authorized hardware calibration and is not a solution.

## Candidate EXP-078A -- Frozen Tangent Macroblock Lifetime Gate

The candidate composes the complete anchor-conditioned SwiGLU path rather than
dropping nonzero channels:

```text
c(a) = SiLU(W_gate a)
M(a) = W_down diag(c(a)) W_up
candidate(x_t) = M(a) x_t
```

The exact prompt's last token supplies `a`; no future target token is visible.
All 24 layer maps are reused across the next seven registered causal positions,
including causally divergent candidate recurrent/KV state.

The direct constructor changes the cost threshold materially. The 0.8B dense
screen requires p50/p05 reuse spans of `13,821/6,249`; the registered nine-path
122B surrogate screen requires `115,299/26,354`. Therefore the seven-token run
is only a cheap early-failure test. If any registered quality or family Gate
fails, reject this frozen-anchor form. If every observation survives, classify
the trace as right-censored and preregister a longer Gate; do not claim success.

No macro matrix is physically materialized in EXP-078A. No new checkpoint,
Ubuntu command, sentinel, repair path, rank sweep, or speed claim is authorized.
Contract: `docs/research/EXPERIMENT_078A_TANGENT_MACROBLOCK_GATE.md`.

## Closed EXP-078A -- Frozen Tangent Macroblock Lifetime Gate

The corrected baseline had zero mismatch across 192 decisions. The frozen
complete anchor operator then matched `0/126` held-out later-token top-1
decisions. Every evaluation case failed at the first reuse position; mean/p95
KL was `14.898423/25.335417`. This misses the direct-construction requirements
of `13,821/6,249` small-checkpoint and `115,299/26,354` surrogate reuse tokens
by orders of magnitude.

Decision:

```text
REJECT_FROZEN_TANGENT_MACROBLOCK_REUSE_PATH
```

Authority: `results/exp_078a/summary.json`; source `cc36819`; evidence
`fe60819`; core
`c743ae14748effaad3a034def7d8d92e67fa09abf65eeb931bef3e8a26368667`.

## Next research Gate -- causal delta construction, not frozen reuse

No core candidate survives. Do not extend the trace or sweep rank, layer,
prompt, or sentinel variants around the frozen anchor. A next candidate may
study a causally delta-updated computation law only after an E0 Gate identifies
information available before the skipped work and derives the cost of:

```text
delta extraction + operator update + hot application
+ error detection + exact cache repair/fallback + cold traffic
```

The optimistic equation must close the final target fraction without assuming
that `W_gate x_t` or a full Jacobian/operator is free. Until then the honest
status is `NO_SURVIVING_CANDIDATE`. EXP-073 Stage 2 remains separately
authorized hardware calibration, not a solution.

## Candidate EXP-079A -- Causal Proof-State DCT Block-Zonotope Gate

EXP-079A registers one concrete cold-backed proof-state mechanism rather than a
frozen operator update. Every MLP down matrix keeps the exact image of four
procedural DCT directions hot. The omitted checkpoint remains an exact deferred
dependency, bounded as disjoint output-row L2 balls and resolved only through
complete cold row-block reads.

The p50/p95 logical ceilings are the final `1.2*4/405` and `1.5*4/405`
fractions. Every non-down target operation is granted free and the full global
allowance is concentrated on the 24 down projections. A non-deployable oracle
sees the complete current residual and chooses the best quality blocks; a
separate oracle chooses the blocks giving the minimum sound local radius.

The shape-only dense-405B p50 plan charges `2,204,895,168` bytes per token
(`1.08883712%`) and `0.976002844%` favorable operations. Its DCT images occupy
`487,327,680` bytes and proof records `516,096` bytes, before KV/workspace. This
is a derived route, not measured traffic or VRAM.

Promotion requires the unchanged 192-decision control, held-out 99% top-1,
95% in every family, mean/p95 KL `<=0.02/0.05`, and p50/p95 minimum local sound
radius no larger than the exact MLP-output L2 signal. Failure closes this exact
DCT pilot plus row-block zonotope path with no rank/block/prompt rescue. A pass
authorizes only a nonlinear proof-propagation Gate.

Contract: `docs/research/EXPERIMENT_079A_CAUSAL_PROOF_STATE_GATE.md`.
Status: PREREGISTERED; NO MODEL RESULT OR TARGET-SERVER ACTION.

## Closed EXP-079A -- Causal Proof-State DCT Block-Zonotope Gate

The unchanged target control had zero mismatch across 192 decisions. The p50
logical plan stayed below its final fraction at `1.163034707%` and granted 23 of
256 row blocks per down projection. Nevertheless the complete-residual quality
oracle matched only `6/144` held-out top-1 decisions; mean/p95 KL was
`7.544862/12.616226`.

The separate proof-optimal oracle was even more decisive: minimum sound-radius
p50/p95 was `48.663918x/57.778748x` the exact MLP-output signal. The p95
allowance improved top-1 only to `14/144` and left median radius `46.993979x`.

Decision:

```text
REJECT_DCT_BLOCK_ZONOTOPE_CAUSAL_PROOF_PATH
```

Authority: `results/exp_079a/summary.json`; source `e0c661e`; evidence
`38bfd6c`; deterministic core
`c8d794bea8f17b31e40b9f667836d2198ce80137e23080260f81ea6866112eeb`.

## Next research Gate -- residual information, not another fixed pilot

No core candidate survives. Do not sweep fixed bases, ranks, row blocks, layers,
prompts, or proof-radius thresholds around EXP-079A. The failure occurred under
two free oracles and a radius Gate fifty times too wide, so implementing a
selector or nonlinear propagator cannot rescue this sidecar.

A next candidate must explain how current causal information determines the
dense non-pilot residual without first executing it. Acceptable E0 classes must
introduce a materially new query-time dependency, such as a lossless interactive
code with a proven sublinear decode route or a causal multi-token dependency
that amortizes exact residual work. It must charge code/prover generation,
queries, verification, correction, fallback, RAM/SSD/PCIe/VRAM, and show the
final `1.185185%` equation before implementation. The honest status remains
`NO_SURVIVING_CANDIDATE`.

## Candidate EXP-080A -- Hyperblock Exact Rectangular-Multiplication Gate

The broad Hyperwave proposal contains a component not tested by EXP-048/049:
replace `K` independent dense matrix-vector operations with one exact
rectangular `W @ X` operation and share scalar arithmetic across token columns.

EXP-080A grants the exact future block for free and therefore does not claim a
causal executor. It first applies the cheapest arithmetic Gate to the frozen
405B tensor shapes. The constructive arm is recursive Strassen with exact
multiply/add/padding/accumulation counts and a free best leaf-size oracle. A
unit-constant modern matrix-multiplication exponent is only a theoretical
diagnostic.

The path promotes only if one preregistered block length simultaneously reaches
the final p50 traffic and arithmetic fractions and the favorable workspace
stays within 8 GiB. Failure stops packed kernels, model downloads, speculative
backends, and target-hardware work. A pass still does not reopen prior causal
proposal families; that would require a separately justified new information
source.

Contract: `docs/research/EXPERIMENT_080A_HYPERBLOCK_FMM_GATE.md`.
Status: COMPLETE; STANDARD STRASSEN CORE REJECTED.

## Closed EXP-080A result

All 80 exact controls passed, but no constructive block length jointly passed.
The best registered row, `K=16,384`, reduced traffic to `0.006103516%` and fit
the favorable `7.539063 GiB` workspace equation while leaving standard
Strassen arithmetic at `36.111580%`, versus the `1.185185%` p50 allowance. The
remaining constructive arithmetic gap is `30.469145x` before physical overhead.

The unit-constant exponent oracle is useful only as a target: it first passes at
`K=512`, or `K=8,192` with the streamed 4B-draft charge. It is neither a
constructive packed algorithm nor a causal future-block source. Therefore do
not build a Strassen kernel, combine it with rejected Jacobi/draft families, or
download a larger checkpoint.

Decision:

```text
REJECT_STANDARD_STRASSEN_HYPERBLOCK_AS_CORE_RETAIN_COST_MODEL_AUXILIARY
```

## Next research Gate -- two missing inventions

No core candidate survives. A successor must provide both:

1. an explicit exact rectangular/packed algorithm whose fully constructive
   constants close at least the measured `30.469145x` arithmetic gap; and
2. a materially new causal information source capable of forming or committing
   hundreds to thousands of valid future columns without executing an
   equivalent target path first.

An exponent, an uncharged tensor-core primitive, or a new proposal-quality
sweep is insufficient. The next admissible E0 artifact is a closed equation and
constructive algorithm for one of these two missing interfaces, with the other
still granted as an oracle. The honest status is `NO_SURVIVING_CANDIDATE`.

## Closed EXP-081A -- syndrome-recovered nonlinear lookup MatVec

The candidate generator maps the current causal activation through eight
depth-four regression trees and sums eight selected int8 output pages. A
rank-eight residual dictionary is recovered from a precomputed syndrome; six
independent field fingerprints verify the complete result. Verification failure
executes the original `W @ x` and expands cost, never silently approximates.

This is admissible because nonlinear code cells plus query-time residual
syndromes are a new information interface. It does not rely on an exact prior
activation span, a frozen tangent, a fixed DCT pilot, or future columns. At the
frozen 405B shapes it has a favorable `0.9265250%` traffic path but can tolerate
only `0.2586602%` weighted fallback. The cheapest decisive question is whether
held-out real projection residuals are recovered at least `99.75%` of the time.

The finite-field mechanism passed all 321 controls, and its favorable
metadata-complete target equation remained under the logical resource ceilings.
The real premise failed: held-out exact coverage was `8.681672%`, family
coverage was `6.7164%-10.7143%`, and favorable corrected relative-L2 p50/p95
was `0.351269/1.213828`. Observed fallback makes logical traffic
`92.244986%`, not `1.185185%`.

Decision:

```text
REJECT_SYNDROME_RECOVERED_LOOKUP_RESIDUAL_CODE_PATH
```

Contract: `docs/research/EXPERIMENT_081A_SYNDROME_RECOVERED_LOOKUP_MATVEC.md`.
Authority: `results/exp_081a/summary.json`; source `1d3e91f`; evidence
`ee9573d`; core
`8621f6357536b6fc3396872668484c52103e28d2af8291b96575bc4d007c2ccc`.

## Next research Gate -- certify the final decision, not each dense residual

No EXP-082 number is authorized yet. A materially different candidate may use
an end-to-end adaptive certificate:

1. run a cheap causal proposal and fix the exact sampling randomness;
2. propagate signed contribution intervals from unread checkpoint pages to the
   final logits rather than reconstructing every intermediate `W x`;
3. read the page with maximum possible effect on the unresolved top-token or
   fixed-RNG CDF boundary;
4. commit only when every completion of all unread pages yields the same token;
5. otherwise continue reading and finally execute the unchanged dense path.

The cheapest admissibility test is an intentionally favorable oracle that sees
the exact target trace and chooses the minimum page set needed for a valid final
decision certificate. It must report population and family page fractions,
certificate construction cost, bound-state bytes, original-page traffic,
verification, correction, fallback, and the complete 405B equation. If even
that oracle exceeds `1.185185%`, close the decision-certificate family before
building interval propagation. Passing would authorize only a causal bound
construction Gate. It would not validate physical speed or 405B execution.

## E0 closure -- final-head and proof-carrying trace routes

Do not open the absolute-unread final-token certificate described above.
EXP-068 already tested its strongest per-competitor reveal form. A correlated
exact MIPS index can change output-head access, but that entire family is only
`0.520459999%` of the registered non-embedding 405B coefficients and cannot be
the core.

A whole-network proof-carrying trace also stops at E0 as a standalone local
executor. Six random linear checks fit the final budget, but a same-machine
dense trace proposer leaves fully charged operation/traffic fractions at
`100.058120260%/100.266838924%`. Retain the verifier only as an auxiliary for a
future independently justified trace generator.

Authority: `docs/research/E0_DECISION_AND_PROOF_TRACE_TRIAGE.md`.

## Candidate EXP-082A -- exact differential spanning-tree MatVec

The next Gate tests a published exact structural algorithm not measured by
EXP-064: connect all quantized rows or all columns in a Hamming minimum
spanning tree and evaluate only parent-child differences. This strictly
contains a bounded prototype dictionary and has a low-VC/Pollard-dimension
subquadratic route.

Stage 1 uses exact 32-coefficient block-pattern IDs to certify a favorable
lower bound on every possible row and column tree. It must stop immediately if
the weighted best-orientation coefficient lower bound exceeds
`1.185185185%`. Only an inconclusive lower bound authorizes exact MST
construction and fully charged sparse-delta storage/traffic accounting.

Contract: `docs/research/EXPERIMENT_082A_DIFFERENTIAL_SPANNING_TREE_GATE.md`.
Status: E0 PREREGISTERED; NO RESULT EXISTS.

## EXP-082A closed -- do not construct the tree

The clean E1 run rejected the family at the cheapest registered Gate. Across
21 real Q4 projections, the favorable weighted lower bound was `1.562367394%`
against `1.185185185%`; p50/p90 were `1.562935965%/1.564025879%`. All 72 exact
controls passed and an independent replay reproduced the decision, core hash,
and tabular payloads byte for byte.

Do not spend the remaining research budget on an exact MST, block-size sweep,
tree heuristic, runtime, CUDA, larger checkpoint, or hardware measurement.

## Next admissible frontier -- synthetic-intermediate exact circuits, E0 only

No experiment number is promoted yet. A terminal-only tree is now closed, but
an exact network containing synthetic coefficient vectors not present as rows
or columns is not logically closed by the same nearest-terminal bound. The
generic metric relation only turns this result into a favorable Steiner lower
bound of `0.781183697%`, below the final target, so it does not prove rejection.

Before preregistration, an E0 note must determine whether such a network is
actually new relative to EXP-053/054/072's exact DAG, dictionary, and circuit
families. It must specify a causal compiler that discovers useful synthetic
nodes without charging dense work, and account for every edge delta, synthetic
activation, metadata byte, build cost, and fallback. Without both novelty and
a favorable fully charged equation, the correct state remains
`NO_SURVIVING_CANDIDATE`.

## Active research map

The persistent effort is indexed at
`.scratch/vortex-certain-milestone/map.md`. The first unblocked ticket is
`Derive the Query-Adaptive Cold-Backed Equation`; the synthetic-circuit audit
is resolved. Per the map contract, only the claimed child ticket may be worked
in one session. No EXP-083 branch is authorized by map progress alone.

## Synthetic-intermediate E0 closure and next admissible frontier

The required novelty audit is complete. A synthetic Hamming/Steiner tree
expands exactly into a static linear straight-line program and is a restricted
instance of the synthetic-form arithmetic DAG already archived as EXP-072B.
Cold placement changes residency, not the computation or its information
source.

The generic `SMT >= MST/2` relation yields only `0.781183697%` from EXP-082A,
so this is a taxonomy/resource rejection rather than a claimed finite Steiner
lower bound. Static synthetic circuits have no new mechanism, no favorable
fully charged operation/traffic closure, and adverse random-tree and
linear-circuit counterevidence. Do not assign EXP-083 or implement a Steiner
compiler, generic circuit synthesizer, cold interpreter, or kernel.

Authority: `docs/research/E0_SYNTHETIC_INTERMEDIATE_CIRCUIT_AUDIT.md`.

No experiment number is promoted. The only materially distinct adjacent class
is a Query-Adaptive Cold Source: for observed causal activation `x`, a selector
identifies a small subset of lossless checkpoint-derived pages or operations
without a dense pass. The next ticket must derive its most favorable complete
equation for selection, probes, page bytes, execution, intermediate/hot state,
verification, misses, fallback, and compile amortization at the registered
405B shapes.

An equation alone does not promote the class. A later ticket must identify a
causal information source that instantiates it without target-future leakage,
checkpoint modification, training, added hardware, a free oracle, or hidden
dense discovery. Until both exist, the state is `NO_SURVIVING_CANDIDATE`.

## Query-adaptive equation resolved -- search the causal information source

The registered-shape equation is now fixed in
`docs/research/E0_QUERY_ADAPTIVE_COLD_EQUATION.md`:

```text
R = g + kappa/N + rho*h + (1-rho)*(m+1)
best-case p50 exact coverage                  >=98.814814815%
coverage after known verifier                >=99.081653739%
best-case useful information amplification   >=84.375x
amplification after known verifier            >=108.891389x
```

Raw Q4 page selection is not a candidate unless a checkpoint-derived code or
certificate determines the omitted contribution. The next unblocked map ticket
is `Find a Causal Circuit Information Source`. It must enumerate genuinely
causal coded sources, reject any that reduce to static circuits, replay,
training, free trace generation, or dense selector scans, and identify the
cheapest finite Gate for any source that survives on paper.

Do not assign EXP-083, build a runtime, download a model, or use the private
Ubuntu server merely because the equation is known. The state remains
`NO_SURVIVING_CANDIDATE` until a concrete source instantiates it.

## Causal source identified -- preregister its cheapest Gate

`docs/research/E0_CAUSAL_RESIDUAL_ATLAS_SOURCE.md` identifies the only current
paper candidate: a **Causal Residual Atlas** built from exact committed-prefix
`(x, W x)` pairs. It evaluates the cached prefix subspace and keeps every
out-of-span component as an explicit cold residual with a sound bound or exact
fallback. This is materially different from tolerance-only Atlas replay,
fixed DCT pilots, raw page omission, and source-free proof traces.

The favorable rank-16 registered screen fits the logical p50 fractions only
at a requested `0.2%` cold share: 64-column rounding yields `1,009` pages/token,
`0.980995178 GiB` capsule state, and 64-token amortized
traffic/operations of `1.085025716%/0.557958575%`. It consequently needs
`99.899840530%` certificate coverage. Rank 16 plus a requested `0.5%` cold
share and rank 32 plus `0.2%` are already traffic-infeasible.

The next Wayfinder ticket must preregister the cheapest decisive test before
assigning an experiment number. It must freeze one rank, page width, prefix
length policy, held-out prompt population, exact-reference favorable selector,
output-contract test, and stop rule. First measure whether committed-prefix
subspaces plus the most favorable legal residual-page order could possibly
meet the `99.899840530%` coverage frontier on the already pinned small
checkpoint. Failure stops bound propagation and all backend work. Passing may
authorize only a deployable outward-rounded residual propagator.

Do not run a parameter sweep, use target-future activations as the deployable
selector, construct a cold runtime, download a larger model, or access the
private Ubuntu server. The current result is source identification with
synthetic controls, not a Surviving Candidate, EXP-083 result, positive
milestone, or E2-E7 evidence.

## Causal Residual Atlas cheapest Gate preregistered -- execute next

The map's preregistration ticket is resolved. The frozen contract is
`docs/research/CAUSAL_RESIDUAL_ATLAS_CHEAPEST_GATE.md`; no checkpoint execution
or experiment number exists yet.

The next execution is deliberately smaller than an eight-token, three-depth,
all-role trace:

```text
checkpoint/revision       pinned Qwen3.5-0.8B / existing local payload
population                18 evaluation prompts, six families
causal position           teacher index 1, first post-prefill module call
prefix basis              prompt-only top-16 SVD, current input excluded
layer                     11
required projections      q_proj and down_proj
page width/counts         64 columns; 16 and 56 pages
oracle candidates         1,296
token states/branches     18 / 36
required successes        18/18 tokens, 3/3 each family, 36/36 branches
quality                   mean/p95 KL <=0.02/0.05
```

Each page candidate receives an illegal native-anchored exact center and all
other model work remains dense. The exact-reference oracle enumerates all
pages, prefers unchanged top-1, then minimizes KL. This is a favorable ceiling;
it cannot become the runtime selector.

Any valid token failure rejects the frozen rank-16/page-64 path and stops
rank/page/layer/prompt/tolerance rescue. A complete pass authorizes only a
pair-constructed center plus outward-rounded bound Gate over more positions and
depths. It does not authorize a backend, cold scheduler, CUDA, larger model,
private-server action, E2, or a 405B claim.

The next unblocked map ticket is `Run the Causal Residual Atlas First-Decode
Gate`. It may assign the next experiment number only while implementing this
unchanged contract.

## EXP-083A passed -- preregister the legal causal-pair and bound Gate

The frozen favorable oracle passed and reproduced: 18/18 tokens, 36/36
branches, mean/p95 KL `0.007225545/0.037660753`, zero invalid controls, and the
same deterministic core across two complete model runs. Do not rerun rank,
page, layer, prompt, or tolerance sweeps; the favorable existence question is
answered for this population.

The next cheapest uncertainty is executability. Before another experiment
number, freeze a Gate that separates and tests:

1. pair-only construction of `Q` and native-numerical `Z=WQ` from committed
   prefix inputs/images, without a checkpoint scan or target-future token;
2. a target-logit-free page rule using only the current committed activation
   and prepaid checkpoint metadata;
3. an outward-rounded unread-residual certificate propagated to the declared
   output contract, with corruption/non-finite failure; and
4. fully charged miss, exact completion/fallback, state, build, and query work.

The observed 18 evaluation prompts are no longer a tuning set. Selector or
bound choices must be analytic and frozen without consulting their target
logits, or use a newly hashed untouched evaluation population. The post-hoc
minimum-radius rule is evidence against using that rule unchanged: it failed
2/18 tokens and its p95 KL was `0.075846638542797`.

Passing the next Gate still does not authorize E2 until the selected projections
are actually replaced simultaneously under fail-closed execution. It does not
authorize CUDA, the Ubuntu host, a larger download, or 405B. The next unblocked
map ticket is `Preregister the Legal Causal-Pair and Outward-Bound Gate`.

## Legal pair/outward Gate frozen -- execute the untouched last-down test

The executability contract is now authority at
`docs/research/CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_GATE.md`. No experiment
number or checkpoint result exists yet.

Frozen first stage:

```text
population              24 newly hashed prompts; prior 18 rows quarantined
position                first genuine post-prefill call
layer / projection      23 / down_proj
pair compiler           two-pass causal MGS from prefix pairs only
rank / page             16 / one contiguous 64-column page
selector                maximum current residual-page energy, lowest tie
local certificate       verified spectral unread radius + pair/native errors
declared output         strict final RMSNorm + LM-head native BF16 greedy top-1
required result         24/24, 4/4 each family, 0 false accept, 0 fallback
quality                 mean/p95 target-to-candidate KL <=0.02/0.05
```

The selector interface contains no weights, native current output, or logits.
The evaluator may compare native logits only after the selection and
certificate verdict are immutable. The first valid unresolved row rejects the
path; it may not trigger page enumeration or method tuning.

Before implementation, pin the authority and prompt hashes in a new config.
The runner must freeze pair/certificate rows, selected page, every outward
term, dense completion, taint/leakage controls, exact baseline comparison,
resource counts, environment, logs, and checksums. It must independently
verify the result bundle.

A pass authorizes only a new backward-layer/eight-position composition Gate.
It does not authorize E2, a scheduler, CUDA, the private Ubuntu server, a
larger checkpoint, or physical claims. The next unblocked map ticket is `Run
the Legal Pair and Outward-Bound Last-Down Gate`.

## EXP-083B source implemented -- freeze and execute exactly once

EXP-083B now implements the unchanged legal-pair authority with a runner,
independent no-forward verifier, focused tests, and a fail-closed workflow.
The candidate stops at layer 23 before reading the current dense `down_proj`,
builds the page decision and strict final certificate, freezes that verdict,
and only then runs the native dense evaluator and deterministic replay.

The static compiler was preflighted without tokenizing or executing any new
prompt. Its outward Gram/verified-Cholesky proof accepted the registered
matrix in one attempt with `beta_W=1.326752041578861` and positive-definite
margin lower bound `1.94850297451582e-07`. This is registered-weight metadata,
not evidence about the untouched 24-prompt population.

Seventeen focused tests and all 453 repository tests pass. No EXP-083B prompt
activation, candidate, native output, result bundle, or scientific decision
exists at this point. After the exact source commit is pinned, the next action
is the single canonical run followed by independent verification. The first
valid unresolved row must stop and reject the primary Atlas path; it may not
change the proof, page, rank, prompt, or threshold.

Executable source is pinned at
`ecf753d7352bcca47767d0fe91c46d84cca66b59`; the frozen config SHA-256 is
`2c8bdf12535e18327f0a4116e8f9dc4b918f900208d405972cfa421764bdd5c1`.

## EXP-083B rejected -- search only for a materially new causal source

The one-shot Gate stopped on `legal_holdout_english_01`. Rank 16 and the
target-free page were valid, but the strict final certificate was unresolved;
one exact fallback therefore rejected the zero-fallback Gate. All controls
were clean and independent no-forward verification passed with deterministic
core `57e78fd4d7b1bdc6e97c705a5acb2e7e408a1023e38ae933799b00b8e3711ff4`.

The rejection is not an artifact that permits another Atlas sweep. The unread
common-spectral term alone was `16.3298572850` around a pre-RMSNorm center of
norm `10.7091120605`. Even an oracle ball containing the observed native
hidden state needs radius `38.9079080403`, versus an ideal top-two row-margin
limit `4.1978252811`. Backward layers, more positions, and E2 integration can
only add uncertainty and are closed for this mechanism.

Current state returns to `NO_SURVIVING_CANDIDATE`. The next Wayfinder ticket
must identify a different causal information source that changes the missing-
contribution premise and derive its complete E0 operations, traffic, state,
verification, miss, fallback, and compile equation before assigning another
experiment number. Correlated/direction-aware information is admissible only
if it is a concrete lossless source with a paid causal constructor, not a
post-hoc tighter bound on the rejected rows.

Do not run another Atlas rank/page/layer/prompt/tolerance variant, build E2,
contact the Ubuntu host, download a larger model, or begin kernel/hardware
work. The next unblocked map ticket is `Find a Post-Atlas Causal Information
Source`.

## Post-Atlas E0 audit closed -- no experiment; isolate the cross residual

The strongest distinct source computes signed decision functionals rather than
full intermediate vectors. Its exact form retains `r^T W u` outside cached
primal/dual spans. Forward pairs are therefore Atlas-normal-form information;
a prompt-dependent exact dual build costs `1.5625%` at the registered 64-token
service life, and a static final-slice vocabulary composite costs a favorable
`12.720703125 GiB` plus `6.765979992%` scan traffic. Both fail before a real
Gate. The zero-forward EXP-083B diagnostic also left all 248,319 competitors
unresolved under the obvious few-row norm screen.

No experiment number is assigned and no model, kernel, larger checkpoint,
Ubuntu command, or hardware action is authorized. The current classification
remains `NO_SURVIVING_CANDIDATE`.

The next research ticket may address only this exact question: can a
checkpoint-derived lossless query mechanism obtain `r^T W u` for arbitrary
prompt-dependent residual pairs with a finite, favorable 405B equation for
construction, storage, operations, traffic, verification, misses, and
fallback? It must avoid a dense forward/transpose constructor, full direction
enumeration, training, target-future leakage, external proof without a local
result source, and disguised dense discovery. First derive a scoped lower
bound or concrete exact query structure; do not start E1 until E0 closes.

The resulting map ticket was `Break the Bilinear Cross-Residual Barrier`.

## Bilinear cross-residual separable-code E0 closure

`Break the Bilinear Cross-Residual Barrier` derived a finite lower bound for
the strongest direct Atlas extension: arbitrary matrix-local left/right linear
covering codes with raw-coordinate repair only for their cross residual. With
all `8 GiB` granted to binary code images and every non-cross cost free, the
registered whole-population lower bound is `1.52104775688%`, or
`1.28338404487x` the complete p50 allowance. This closes larger bases,
covering-code bases, and separately coded row/column residuals under the
declared Cartesian coefficient-probe model.

It does **not** close the general static vector-matrix-vector problem. The
source-audited literature leaves a large gap for nonlinear cell-probe
structures, and this proof additionally assumes matrix-local advice plus
independently selectable query pairs across matrices. No universal
impossibility, actual Transformer reachability result, or word-to-hardware
mapping follows.

No experiment number, E1 run, model download, Ubuntu command, kernel, or
hardware action is authorized. The next E0 frontier must change the
information interface, not the code rate:

```text
joint/nonseparable cross-matrix advice with charged cancellation/probes
or
a causal query restriction derived from unchanged real-checkpoint execution
```

Any proposal must still provide a finite constructor, representation,
selector, operation, traffic, state, build, verification, miss, and fallback
equation below `1.185185185%`. A shared bit may not be divided across matrices
without proving how query-local answers cancel its other-matrix content.

The next unblocked map ticket is `Resolve Cross-Matrix Advice Locality`.

## Cross-matrix locality resolved -- certify the real causal query restriction

Global linear advice cannot donate its projection dimension to every matrix
for free. A block-local exact query may use only `U intersect V_i` without
outside probes; a fixed outside support `T` buys at most `|T|` additional local
dimensions. However, query-dependent supports prevent those local statements
from becoming a full direct sum. The finite global span-to-cover bound is only
`6,407,133` coefficient uses, `0.001586914271%` of dense and about `1/747` of
the complete p50 allowance.

Do not claim that globally mixed advice is impossible, build a global code
without its constructor/query equation, or repeat matrix-local covering codes.
The next high-value uncertainty is the arbitrary Cartesian query assumption.
Open one proof-first ticket to determine whether unchanged causal Transformer
execution restricts the Bilinear Cross Residual population enough to change
the bound.

That ticket must specify a leakage-free trace extractor on the already pinned
small checkpoint, the exact residual-pair object, build/evaluation separation,
GF(2) or finite-field rank/cover certificate, population/family thresholds,
and a 405B equation charging extraction, representation, query operations,
logical/address traffic, state, verification, misses, and fallback. First
derive a shape-only information threshold and the cheapest frozen E1 Gate. Do
not assign an experiment number or execute new model rows until that E0
contract proves that the measured causal statistic could cross the final
fraction.

No model download, private Ubuntu action, kernel, or hardware stage is
authorized. The next unblocked map ticket is
`Certify Causal Bilinear Query Restriction`.

## Causal bilinear restriction threshold resolved -- run only the frozen rank Gate

The shape-only question is now closed for one concrete query structure. A
factor-scanned Causal Bilinear Span Ledger stores the residual outer products
as left/right factors and combines cached scalar answers only after an exact
rational membership witness.

With a favorable two-byte factor grant, six-check verifier, full metadata, and
46 dense-equivalent build passes amortized over 20M service tokens, the
registered 405B equation permits at most span dimension 23:

```text
B=23 common traffic/operations             1.159413233% / 0.281295494%
B=23 component state                                2.104879502 GiB
B=24 traffic                                        above 1.185185185%
```

The frozen small-checkpoint Gate uses only the last Qwen3.5-0.8B `down_proj`.
It builds from six existing build prompts x four decode positions and evaluates
18 disjoint prompts x two positions. Prompt-only side bases have rank 16.
Every float32 residual factor is treated as an exact dyadic value; rank
increases under primes `65521/65519/65497` are certified independent arrivals,
while modular nonincrease receives no hit credit without an exact rational
witness.

For 36 held-out rows, the best post-hoc 23-dimensional subspace must miss at
least `rank-23`. Only four registered-down-equivalent fallbacks fit the
remaining traffic, so rank 28 is the canonical early rejection threshold. A
pass requires all controls, at most four exact misses, at least five hits in
each six-row family, and rank at most 27.

This is a favorable E1 structural Gate, not a Core Candidate. Pair extraction
and a local result source are free; the known general VJP is dense. A pass may
open only a paid extractor and native numerical-semantics Gate. A failure
closes this factor-scan ledger without a rank/prompt/prime sweep.

Authority: `docs/research/E0_CAUSAL_BILINEAR_QUERY_RESTRICTION.md`. No model
row or experiment number exists yet. The next unblocked map ticket is `Run the
Causal Bilinear Rank Gate`.

## EXP-084A rejected -- require a different exact query representation

The frozen Causal Bilinear Rank Gate is complete. After a zero-row batched
capture control failure was preserved, the sequential committed-prefix source
freeze passed all 114 controls. The 24 build queries had rank 24 under all
three primes, and the first 23 exact independent rows filled the ledger.

The first five held-out rows were all exact rational nonmembers. Zero hits and
five misses exceed the four-fallback allowance, so the preregistered stop
returned:

```text
REJECT_CAUSAL_BILINEAR_FACTOR_SPAN_LEDGER_AS_CORE
```

The separate held-out rank process reached only five because the miss stop
fired first. Do not report rank 28, retune the basis, select another build
subset, or sweep rank, side rank, primes, prompts, positions, or decision
directions. Those would be post-result rescue attempts, and rank 24 is already
outside the registered factor-scan traffic window.

There is no next model or hardware experiment. The next admissible ticket must
first invent a materially different exact Bilinear Cross Residual query code:
nonlinear, implicit, or otherwise not an every-token scan of stored factor
pairs. Before E1 it must close a complete finite E0 equation for construction,
representation, causal selection, query operations, logical/address traffic,
hot/cold state, verification, misses, and fallback. Another linear-span ledger
or Atlas variant is out of scope.

Current classification remains `NO_SURVIVING_CANDIDATE`. No Ubuntu action,
download, E2 integration, kernel, physical benchmark, 122B/405B, or E6/E7 is
authorized by this result. Authority: `results/exp_084a` and
`docs/research/EXPERIMENT_084A_CAUSAL_BILINEAR_RANK_GATE.md`.

## Trace-built code unions closed -- derive an implicit source before E1

Do not partition the EXP-084A build span, train a router, add more trace
leaves, or sweep leaf dimension. Even with a free perfect router, the
registered construction budget retains at most 87,958 independent directions,
only `0.43979%` of 20M independent queries, while full fallback requires
`99.999992826%` coverage. The frozen full 24-row calibration span also misses
all five stored evaluation rows under all three primes.

The next unblocked ticket is `Derive an Implicit Nonlinear Bilinear Query
Source`. It must begin with one concrete automatic checkpoint constructor and
exact query equation for `r^T W u`; naming a nonlinear index or assuming an
oracle is insufficient. Before any E1 run it must charge representation,
build, selector, probes, logical/address traffic, operations, hot/cold state,
verification, misses, and fallback, and explain where new exact answer
information comes from without materializing trace basis directions.

No model row, EXP-085 number, backend, kernel, download, private Ubuntu action,
physical benchmark, 122B/405B execution, or E2-E7 work is authorized. Current
classification remains `NO_SURVIVING_CANDIDATE`. Authority:
`docs/research/E0_QUERY_ADAPTIVE_CODE_UNION_BOUND.md`.

## Exact-field nonlinear path collapsed -- derive a finite-word source only

Do not propose rational/arithmetic branching, bounded-twin-width Hamming
traversal, rectangle decomposition, or grammar evaluation as the missing
source. A full-dimensional exact algebraic path computes the same bilinear
function globally, and Baur--Strassen maps it to static `W u` with
constant-factor arithmetic overhead. Those routes return to F-049/F-050 and
archived EXP-072B.

The next unblocked ticket is `Derive a Finite-Word Discontinuous Bilinear
Source`. It must begin with one non-exhaustive automatic checkpoint
constructor and exact finite-word query equation. The word size, native
rounding or integer surrogate, bitwise/modular/floor instructions, table
growth, address generation, every probe, traffic, operations, state,
verification, misses, fallback, and service-life build amortization must be
explicit. An unbounded word, free SIMD factor, exhaustive answer table, or
static field circuit is inadmissible.

No model row, EXP-085 number, backend, kernel, download, private Ubuntu action,
physical benchmark, 122B/405B execution, or E2-E7 work is authorized. Current
classification remains `NO_SURVIVING_CANDIDATE`. Authority:
`docs/research/E0_IMPLICIT_NONLINEAR_BILINEAR_SOURCE_AUDIT.md`.

## Explicit finite-word sources closed -- resolve the general probe gap at E0

Do not build a local rank-one truth table, sweep its block dimensions, count
one-bit payload instead of physical word traffic, omit construction and
address costs, rerun Mailman/Four-Russians, benchmark full-scan broadword
packing, import Boolean rectangle nonemptiness as a signed numerical result,
or call the native finite state space a free answer table.

The next unblocked ticket is Resolve the General Finite-Word Rank-One Probe
Gap. It must produce exactly one of two deliverables before any model or
hardware action:

1. a globally nonlocal nonlinear bounded-word constructor with a complete exact
   query equation and target-scale state/build/probe/traffic/operation/
   verification/fallback closure; or
2. a lower bound or finite adversarial certificate that actually covers that
   model under the global 8 GiB advice grant.

Current systematic linear, separable, direct-sum, and exact-field results are
insufficient for the second deliverable. Do not overstate them. No EXP-085,
model row, backend, kernel, download, Ubuntu action, hardware benchmark,
122B/405B execution, or E2-E7 work is authorized. Current classification
remains NO_SURVIVING_CANDIDATE. Authority:
docs/research/E0_FINITE_WORD_DISCONTINUOUS_BILINEAR_SOURCE_AUDIT.md.

## Boolean rectangles and limited-independence theorems closed -- keep E0

Do not lift Boolean all-zero rectangles into exact signed/Q4/BF16 arithmetic,
quote their hidden-constant big-O leading term as a target bound, or treat
free cell-probe computation as target-machine compute. Any rectangle summary
claimed to answer every exact intersection without raw probes must confront
the singleton injection lemma and pay the raw information content.

Do not apply Korten--Pitassi--Impagliazzo 2025 to the full rank-one family.
Its required limited independence is contradicted by the exact three-query
relation `q(u) XOR q(v) XOR q(u+v)=0`. Do not divide a whole-MatVec lower
bound by fewer than `n` scalar queries, and do not import the 2026 dynamic
Multiphase result into the static free-advice model.

The active ticket remains Resolve the General Finite-Word Rank-One Probe Gap.
Its next action must still be either a different globally nonlocal nonlinear
exact numerical constructor with a complete 405B equation, or a theorem that
covers arbitrary adaptive global advice and scalar rank-one output. No model,
EXP-085, backend, kernel, download, private Ubuntu action, hardware benchmark,
or E2-E7 action is authorized. Authority:
`docs/research/E0_GLOBAL_NONLINEAR_RANK_ONE_FRONTIER.md`.

## Finite-semiring preprocessing graph closed -- do not hide the catalog

Do not cite Williams' `O(n^2/log^2 n)` finite-semiring step count while
omitting the answer-pattern bytes carried by its adjacency lists or the
`K^b` catalog behind every input group. In the favorable physical expansion,
query payload falls only as `1/b` while persistent edge payload grows as
`K^b/b`. At theorem-parameter `b=14`, one Boolean registered square already
needs `36.616085 GiB` of edges, and the ideal model-wide one-bit sidecar is
`6,911.57x` the global 8 GiB grant. Q4 and BF16 single-query payloads also
miss the complete block budget.

Do not relabel BF16 symbols or FP32 accumulators as a finite semiring. Their
rounded addition is non-associative, and the published regrouping equation is
not the native reference equation. Do not multiply the paper's one-vector
claim into a 32-token amortization theorem; none is published.

The active ticket remains Resolve the General Finite-Word Rank-One Probe Gap.
The next admissible route must be scalar-specific and materially different:
either a globally coupled bounded-word numerical constructor with a complete
native-order 405B/32-token equation, or a direct scalar rank-one lower bound
covering arbitrary adaptive global advice. No EXP-085, model, backend,
kernel, download, private Ubuntu action, hardware benchmark, or E2-E7 action
is authorized. Authority:
`docs/research/E0_FINITE_SEMIRING_PREPROCESSING_FRONTIER.md`.

## Naive global-advice direct sum closed -- require a synergy-charging theorem

Do not divide the global 8 GiB advice by matrix, layer, or tile count. The
finite XOR witness shows why: one small jointly computed advice string can
carry full conditional information about every selected matrix after the
others are known. Do not sum per-tile CKL worst cases, set the hidden `Omega`
constant to one, use the displayed `n^2/64` proof coefficient outside its
regime, or assume independently hard tile queries compose into one causal
execution.

The active ticket remains Resolve the General Finite-Word Rank-One Probe Gap.
The next admissible theorem route must jointly charge arbitrary nonlinear
global advice and every cross-matrix probe needed to unlock its synergy for a
simultaneously composable scalar rank-one query family. The alternative is a
concrete globally coupled constructor with an exact native-order 405B and
32-token equation. No EXP-085, model, backend, kernel, download, private
Ubuntu action, hardware benchmark, or E2-E7 action is authorized. Authority:
`docs/research/E0_GLOBAL_ADVICE_SYNERGY_FRONTIER.md`.

## All-linear Fourier direct sum closed at the rank-one boundary

Retain the Fourier-fiber theorem as a valid guardrail: it charges arbitrary
nonlinear global advice and every adaptive cross-block raw probe jointly. Do
not cite its `26.2%` all-linear result for Transformer rank-one queries. The
spanning step needs every Walsh character, while 32 independently granted
rank-one tuples have a query-cardinality method ceiling at only
`0.028137293%` of registered coefficients.

The active ticket remains Resolve the General Finite-Word Rank-One Probe Gap.
The next theorem route must lower-bound the rank or another adaptive-tree
complexity measure of `H[F,Q_rank1]` for every sufficiently large nonlinear
advice fiber, at the finite registered scale. Query count alone is closed.
The alternative remains a concrete scalar-specific constructor with complete
native-order state, build, probe, compute, and 32-token equations. No EXP-085,
model, backend, kernel, download, private Ubuntu action, hardware benchmark,
or E2-E7 action is authorized. Authority:
`docs/research/E0_FOURIER_FIBER_DIRECT_SUM_FRONTIER.md`.

## Native-exact local shortcuts closed -- no EXP-085

The old `2.5%` hypothesis has been removed from candidate selection. A cheap
screen on already pinned evidence rejects rounding absorption, temporal value
reuse, run compression, exact product reuse/cancellation, and the direct
low-VC/Pollard exact-MatVec constructor. No additional model execution or
hardware work is justified.

Do not reopen these mechanisms under a new cache, dictionary, cancellation,
VC-dimension, or row-tree name. The next candidate must state where new exact
query-answer information comes from and must pass, before implementation, the
unchanged-result, 8 GiB state, and 20 ms/token equations. If its first required
premise is merely local repetition, full-sweep batching, randomized Boolean
correctness, or an added compute device, reject it without experiment.

The active ticket remains `Resolve the General Finite-Word Rank-One Probe
Gap`; it remains claimed because this audit supplies neither a target-feasible
global constructor nor a covering impossibility theorem. Authority:
`docs/research/E0_NATIVE_EXACT_SHORTCUT_FRONTIER.md`.

## Direct-design cycle after the local-shortcut screen

ΩROUNDLOCK was designed and screened without a new model run. It treats native
BF16 equality as a layerwise state firewall, but the free oracle leaves
`99.90234375%` of down-projection rows unlocked for the Atlas-plus-one-page
predictor. Do not build its selector or repair kernel.

ΩBACKCUT then tried to avoid state repair by certifying only signed logit
differences. Its candidate-winner Gate passes, but the exact source is the
already closed decision-dual `r^T W u` construction. Do not repeat EXP-083B
under that name.

The next admissible candidate must introduce a genuinely new, charged source
of exact finite-word rank-one answers. Its first screen remains only the final
contract-derived allowance, not `2.5%`; reject it immediately if it merely
wraps Atlas, a decision dual, local repetition, or a full sweep in a new
certificate vocabulary.

## Spiky and entrywise-power direct evaluators closed -- no EXP-085

Do not implement a sum-of-spiky-matrices runtime.  The direct evaluator's
resource is total active row/column factor incidence `L`, not merely the
number of named components. A finite sign-pattern count grants the complete
model-wide `L=4,800,000,000` allowance, arbitrary cross-matrix components,
and ignored invalid cross cells, yet describes at most
`2^184,958,474,578.07` of the `2^403,747,897,344` registered sign
checkpoints. A hidden sparse-support implementation and global budget
reallocation are already included in this count.

Do not reopen entrywise integer powers of a low-rank root as a distinct
execution class.  Their exact multinomial expansion is an ordinary static
rank-`binom(r+p-1,p)` evaluator; at most 96 terms fit the complete favorable
square-matrix allowance.

The next admissible candidate must be outside direct static masked-factor
decompositions and must supply either (1) a globally adaptive numerical query
algorithm with fully charged finite-word/native-order equations, or (2) a
covering lower bound that grants the global 8 GiB nonlinear advice and every
adaptive cross-matrix probe.  The active ticket remains claimed.  No model,
EXP-085, backend, kernel, download, private Ubuntu action, or hardware stage is
authorized.  Authority:
`docs/research/E0_SPIKY_POWER_RANK_ONE_FRONTIER.md`.

## Literal adaptive tables and trapdoor masking closed -- no EXP-085

Do not implement nearest-codeword or nearest-query routing backed by one
literal exact scalar per representative. Even after granting arbitrary joint
representative masks and free routing, a fixed-`u` packing forces more than
`2^13741.25` one-bit entries to repair every `N=16,384` query inside the
complete `8/675` coefficient budget. A direct table also needs a 13,742-bit
address. Linear-image compression is F-055; an actually succinct nonlinear
decoder must provide a new equation rather than hide this table.

Do not use a trapdoored random mask as though it compiled the checkpoint. The
identity `Wx=(W+R)x-Rx` makes `Rx` fast but leaves `(W+R)x` as an arbitrary
dense product. Reopening requires a supplied exact average-case solver whose
own complete target cost fits; the trapdoor paper does not provide one.

Continue the active ticket only through a nonliteral compressed adaptive
answer decoder with finite word/native-order accounting, or a joint lower
bound covering that decoder and global advice. No model, EXP-085, backend,
kernel, download, Ubuntu action, or hardware stage is authorized. Authority:
`docs/research/E0_ADAPTIVE_CODEBOOK_TRAPDOOR_FRONTIER.md`.

## Direct row lottery closed -- require a compressed oracle

Do not cite worst-case-to-average-case matrix error correction as the missing
execution source. OMEGA-XORLIFT's reduction is valid, but it assumes a
preprocessed oracle with above-random average-coordinate accuracy on uniform
finite-field matrices and vectors. The common-row Fourier Gate is stronger
than that premise and cannot reject it.

Do not implement OMEGA-ROWLOTTERY. Concentrating exact work on selected rows
does beat average guessing, but a rank-covering collection of direct encoded
row forms reads at least one full matrix coefficient source. At the registered
one-bit population that is `47.00244140625 GiB`, before repeated calls, Q4,
native arithmetic, decoding, or verification.

The active ticket remains `Resolve the General Finite-Word Rank-One Probe
Gap`. Its next constructor must supply a **compressed nonlinear** cold-backed
above-random oracle with every repeated probe/call charged, or use a materially
different non-oracle information source. The alternative remains a covering
lower bound.

## Functional-code names closed -- continue beyond linear recovery sets

Do not implement functional PIR, functional batch, or functional array-code
layouts. Their recovery sets contain stored linear forms and reduce to the
already screened sparse-span/covering model. The small exhaustive prototype
is a novelty check, not authority for a general nonlinear lower bound.

Continue only with a nonliteral **nonlinear** cold decoder whose query-time
equation includes address discovery and every raw probe, or with a theorem
that covers arbitrary nonlinear global advice and adaptive rank-one probes.
No model, EXP-085, backend, kernel, download, Ubuntu action, or hardware stage
is authorized. Authority:
`docs/research/E0_FUNCTIONAL_ARRAY_CODE_NOVELTY_GATE.md`.

## Small nonlinear fibers screened -- require a scalable mechanism

Do not spend another cycle exhaustively enlarging the `2 x 2` fiber search or
sampling nearby small sets. The complete toy optimum is affine covering, and
the `2 x 3` affine profile adds no nonlinear source. A continuation must now
provide either a scalable construction for high-dimensional nonlinear fibers
or a theorem controlling rank-one decision depth on every sufficiently large
fiber. Small-cardinality pattern matching alone cannot promote a candidate.

No model, EXP-085, backend, kernel, download, private Ubuntu action, or
hardware stage is authorized. Authority:
`docs/research/E0_NONLINEAR_FIBER_DECISION_DEPTH_SCREEN.md`.

## Lossless/cut/gauge routes screened -- require a cross-model source

Do not call a cut-value oracle an implementation, do not treat fused
decompression as removal of the checkpoint sweep, and do not continue
attention-only gauge algebra after the free-deletion ceiling. Fused lossless
execution may be composed with a future mechanism only after that mechanism
independently supplies at least 841 exact tokens per favorable 32 GB/s BF16
sweep or avoids the sweep altogether.

The next candidate must couple information outside attention alone and expose
one exact equation for MLP, embeddings, and state as well as weight traffic.
It must be materially different from linear circuits, local tables, Boolean
zero rectangles, subset-aggregate oracles, and compressed full sweeps. No
model, EXP-085, backend, kernel, download, private Ubuntu action, or hardware
stage is authorized. Authority:
`docs/research/E0_LOSSLESS_CUT_GAUGE_FRONTIER.md`.

## Fixed-linear disjoint-support frontier closed through shape 846

Do not construct or search the former `30 x 40` or `31 x 39` dictionaries.
The anti-flag spectrum forces representative-support collisions inside a
minimal rank decomposition, and exact XOR accounting turns those collisions
into a decisive determinantal-capacity rejection.  `30 x 44` also fails.

The first remaining fixed-linear parameter shape is `31 x 43`, `S=1,559`,
`t=15`.  Continue it only if a concrete atom equation and sub-dense decoder
are derived before search.  Its current ratio `3.409163670153519...` is slack,
not evidence of existence.  If no such equation emerges, change mechanism to
a genuinely adaptive finite-word decoder; do not spend another cycle on
nearby rectangle sweeps.

No EXP-085, model, backend, kernel, download, private Ubuntu action, or
hardware stage is authorized.  Authority:
`docs/research/E0_BIORTHOGONAL_DECOMPOSITION_CANCELLATION_GATE.md`.

## Fixed-linear overlap charge is recursive -- frontier closed through 856

Do not construct or search `31 x 43` or `32 x 42`. A positive spectral
internal-edge bound gives an overlapping adjacent anti-flag pair, not merely
an average completed decomposition. Peeling that pair and applying the same
theorem to the residual rank forces cumulative cancellation. `31 x 43`
fails at rank 22 with radius 324 and exact ratio
`0.4293893830152846...`.

The new first fixed-linear parameter survivor is `31 x 42`, `S=1,523`,
`t=15`, but its best ratio `38.10086306517199...` is only slack. Do not sweep
nearby shapes. Continue only with a concrete structural atom/decomposer
equation or, preferably, a genuinely adaptive finite-word decoder whose
address discovery, words, and output operation are explicit.

No EXP-085, model, backend, kernel, download, private Ubuntu action, or
hardware stage is authorized. Authority:
`docs/research/E0_RECURSIVE_BIORTHOGONAL_CANCELLATION_GATE.md`.

## Adaptive extension-field linear summaries are closed

Do not implement a value-adaptive extension-field linear dictionary. The
zero-transcript theorem already permits its strongest deterministic address
logic and arbitrary exact post-processing, then forces a sparse field-span
representation. Exact counting and the aggregate storage equation reject it
well before native execution.

The next constructor must change the information source, not only the
scheduler. It must expose either (a) explicit nonlinear stored cells with a
sub-dense address/recovery algorithm, or (b) a concrete cross-matrix mixed
cell whose joint causal query actually reuses one physical read. Charge its
source-dependent metadata and preprocessing immediately. Apply only the
cheapest semantic/counting Gate before any model or hardware action.

No new experiment number, model, backend, kernel, download, private Ubuntu
action, or hardware stage is authorized. Authority:
`docs/research/E0_EXTENSION_FIELD_ADAPTIVE_SUPPORT_GATE.md`.

## Packed-linear words closed; nonlinear word capacity isolated

Do not implement another linear word packer, field basis, or value-adaptive
linear address scheduler. The packed-linear Segre union Gate already grants
64 unrelated summaries per word and rejects the entire registered side-128
range.

The only block-local finite-word target surviving current capacity is
genuinely nonlinear. Its first exact traffic-line parameters are:

```text
source       arbitrary 25 x 108 binary matrix
storage      50 padded 64-bit words
query        every rank-one parity
budget       two adaptive word reads
```

This is not authorized for implementation from capacity alone. Continue only
if an explicit finite encoder, first address, value-dependent second address,
and exact decoder equation are derived. Use the `2 x 3`, seven-bit, two-probe
case as the cheapest algebraic seed, but do not rerun the systematic subclass
or the same 600-second unrestricted SMT query. If no equation emerges, change
to a global cross-matrix information source and immediately charge its decoder
description, joint causal query, and physical reads.

No experiment number, model, backend, kernel, download, private Ubuntu action,
or hardware stage is authorized. Authorities:
`docs/research/E0_ADAPTIVE_PACKED_LINEAR_WORD_GATE.md` and
`docs/research/E0_ADAPTIVE_NONLINEAR_PROBE_DEGREE_GATE.md`.

<!-- FIXED_PUBLIC_DYNAMIC_EXECUTOR_RESULT:3b9534800671e17eaef74869cf792186e683d6de -->
## Next constructor action after fixed-public hosted gate

Use the committed actual-checkpoint tensor audit and physical ledger to modify only the measured failing resource term. If G4 failed for both primitives, the next implementation must eliminate the observed artifact/residency/I/O/latency term rather than rename the same codec or compiler primitive. If G4 passed, move directly to multi-layer accounting and an exact TARGET-W artifact/8GiB ledger when gated tensor access is available.

## Fixed-public dynamic executor hosted result — commit `3b9534800671e17eaef74869cf792186e683d6de` / run `31982198413`

- Verdict: `REAL_CHECKPOINT_MECHANISM_CLASS_REJECTED_AT_G4`
- DEV-W: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; resolved SHA `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; official `LlamaForCausalLM.from_pretrained()` executed.
- TARGET-W: `meta-llama/Meta-Llama-3.1-405B-Instruct@f9801cba95a53242b3cc928a4a418d12571d1c5f`; metadata access=True; full TARGET-W weights/performance remain unevaluated unless the gated tensors were actually accessible.
- Official reference forward: executed=True.
- Frozen workload SHA-256: `f52dad873703170179b08bf2c48637da2256c2db53ffd245e724c6fa91bb40e6`; 128 decode steps per workload.
- Actual layer tensor audit: `9` parameter tensors, `7080192` reference bytes.
- `cold_packed_projection_sequential_materialization`: G2=True, G3=True, G4=False; artifact=4208296 B; reference-layer=7080192 B; compiled-layer-resident=1771776 B; baseline p50/p95=105.321 ms/202.285 ms; candidate p50/p95=137.679 ms/234.948 ms.
- `checkpoint_mlp_torchinductor_existing_isa`: G2=False, G3=True, G4=False; artifact=169700 B; reference-layer=7080192 B; compiled-layer-resident=7080192 B; baseline p50/p95=102.057 ms/102.057 ms; candidate p50/p95=2268.660 ms/2268.660 ms.
- Instruction/SASS counts are not synthesized: each mechanism reports its measured/not-measured status explicitly.
- G5/G6 remain gated by TARGET-W measurements; DEV-W results are not extrapolated into a 405B performance claim.

<!-- FIXED_PUBLIC_DYNAMIC_EXECUTOR_RESULT:7777eda66232c6a37c39f4e55f5b9ce56032fe92 -->
## Next constructor action after fixed-public hosted gate

Use the committed actual-checkpoint tensor audit and physical ledger to modify only the measured failing resource term. If G4 failed for all implemented primitives, the next implementation must eliminate the observed artifact/residency/I/O/latency term rather than rename the same codec, tiling choice, or compiler primitive. If G4 passed, move directly to multi-layer accounting and an exact TARGET-W artifact/8GiB ledger when gated tensor access is available.

## Fixed-public dynamic executor hosted result — commit `7777eda66232c6a37c39f4e55f5b9ce56032fe92` / run `32219219400`

- Verdict: `REAL_CHECKPOINT_DYNAMIC_EXECUTOR_SURVIVES_G4`
- DEV-W: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; resolved SHA `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; official `LlamaForCausalLM.from_pretrained()` executed.
- TARGET-W: `meta-llama/Meta-Llama-3.1-405B-Instruct@f9801cba95a53242b3cc928a4a418d12571d1c5f`; metadata access=True; full TARGET-W weights/performance remain unevaluated unless the gated tensors were actually accessible.
- Official reference forward: executed=True.
- Frozen workload SHA-256: `f52dad873703170179b08bf2c48637da2256c2db53ffd245e724c6fa91bb40e6`; 128 decode steps per workload.
- Actual layer tensor audit: `9` parameter tensors, `7080192` reference bytes.
- `checkpoint_mlp_output_row_streamed_lossless_existing_isa`: G2=True, G3=True, G4=True; output-row-tile=128; artifact=4223092 B; reference-layer=7080192 B; compiled-layer-resident=1771776 B; peak-hot=393216 B; baseline p50/p95=113.832 ms/203.051 ms; candidate p50/p95=151.764 ms/241.453 ms.
- Instruction/SASS counts are not synthesized: each mechanism reports its measured/not-measured status explicitly.
- G5/G6 remain gated by TARGET-W measurements; DEV-W results are not extrapolated into a 405B performance claim.

<!-- EXP085A_PREREGISTERED -->
## Active EXP-085A — Joint Exact SwiGLU Compiler Gate

The active cheapest decisive Gate compiles one actual public BF16 SwiGLU MLP as a sum of fused nonlinear macro-pages crossing gate, SiLU, multiplication, and down projection. It tests an impossible favorable oracle selector and a sound metadata selector on 24 real causal activations from `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

The frozen pass requires zero false accept/fallback, p50/p95 whole-model-equivalent fractions at or below `1.185185%/1.481481%`, and a projected complete metadata ledger within 8 GiB. The contract is `docs/research/EXPERIMENT_085A_JOINT_EXACT_SWIGLU_COMPILER_GATE.md`. No backend or complete-layer work is authorized before this Gate survives.

<!-- EXP086A_RESULT:376700afc276cb30b5eece61d3c1fb7e2e5b6ccd -->
## EXP-086A result — `376700afc276cb30b5eece61d3c1fb7e2e5b6ccd`

- Decision: `INVALID_CAUSAL_FUNCTIONAL_MICROPROGRAM_NUMERICAL_CONTROL_FAILURE`.
- Evaluation oracle FP32 vector exact: `NOT_AVAILABLE`.
- Evaluation router FP32 vector exact: `NOT_AVAILABLE`.
- Oracle row exact p50: `NOT_AVAILABLE`.
- Router row exact p50: `NOT_AVAILABLE`.
- Projected target sidecar: `2.06103515625 GiB`.
- Compiled MLP operation fraction: `0.00048828125`.
- Whole-model fraction with only MLP replaced: `NOT_AVAILABLE`.
- Complete Transformer layer, 405B, 8-GiB GPU, and physical p50/p95 remain `NOT TESTED`.

<!-- EXP086A_RESULT:152faa529ffafd2b84136dc9282bd237f6a8764a -->
## EXP-086A result — `152faa529ffafd2b84136dc9282bd237f6a8764a`

- Decision: `INVALID_CAUSAL_FUNCTIONAL_MICROPROGRAM_NUMERICAL_CONTROL_FAILURE`.
- Evaluation oracle FP32 vector exact: `NOT_AVAILABLE`.
- Evaluation router FP32 vector exact: `NOT_AVAILABLE`.
- Oracle row exact p50: `NOT_AVAILABLE`.
- Router row exact p50: `NOT_AVAILABLE`.
- Projected target sidecar: `2.06103515625 GiB`.
- Compiled MLP operation fraction: `0.00048828125`.
- Whole-model fraction with only MLP replaced: `NOT_AVAILABLE`.
- Complete Transformer layer, 405B, 8-GiB GPU, and physical p50/p95 remain `NOT TESTED`.

<!-- EXP086A_RESULT:06ab606ddf1fce6720aae3956f5bf70b823c6b05 -->
## EXP-086A result — `06ab606ddf1fce6720aae3956f5bf70b823c6b05`

- Decision: `REJECT_LOW_RANK_FUNCTIONAL_MICROPROGRAM_LIBRARY_AT_ORACLE_GATE`.
- Evaluation oracle FP32 vector exact: `0.0`.
- Evaluation router FP32 vector exact: `0.0`.
- Oracle row exact p50: `0.0052083334885537624`.
- Router row exact p50: `0.0052083334885537624`.
- Projected target sidecar: `2.06103515625 GiB`.
- Compiled MLP operation fraction: `0.00048828125`.
- Whole-model fraction with only MLP replaced: `0.1878463717678428`.
- Complete Transformer layer, 405B, 8-GiB GPU, and physical p50/p95 remain `NOT TESTED`.
