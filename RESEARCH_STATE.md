# VORTEX Research State

Last updated: 2026-08-09 Asia/Seoul

## Fixed final objective

Execute an arbitrary publicly released, unmodified Hugging Face dense Transformer by replacing only the runtime:

- real 405B-class dense flagship;
- total peak GPU VRAM <=8 GiB;
- no target retraining, distillation, fine-tuning, LoRA, or target-specific adapter;
- original declared ability/output contract preserved;
- p50 warm time/token <=1.2x native 4B Q4 and p95 <=1.5x on the same target machine;
- independently reproducible evidence.

The objective is unchanged. A small-checkpoint result, theorem, or projection is not target completion.

## Current environment truth

Available and measured:

```text
GitHub repository, GitHub Actions CPU, and a local Windows reference environment
Python/PyTorch
pinned downloadable small checkpoints
synthetic/reference controls
frozen machine-readable evidence
```

Unavailable and `NOT TESTED`:

```text
405B download/storage/execution
VORTEX inventory or benchmarks on the privately identified Ubuntu 8 GiB target
CUDA and physical kernels
PCIe and target SSD
TTFT and tokens/second
target peak VRAM and power
```

Phase D remains `NOT TESTED`. E6/E7 are not achieved.

## Evidence contract

- Phase A: theory and mechanism;
- Phase B: synthetic/reference controls;
- Phase C: small-real-model falsification or actual operation replacement;
- Phase D: target hardware.

Every result must distinguish `MEASURED`, `DERIVED`, `PROJECTED`, and `UNVERIFIED`. Estimates and asymptotic theorems may not be presented as measurements.

## Current scientific position

The pinned populations have repeatedly behaved like general dense computation under the registered exact tests:

```text
EXP-058 ordinary exact low rank: full rank
EXP-059 shift-displacement structure: full displacement rank
EXP-060 exact-zero Q4 sparsity: insufficient
EXP-061 exact-zero activation skipping: no observed zeros
EXP-062 non-mask attention zeros: negligible
EXP-063 exact cached K/KV equivalence: no duplicates
EXP-064 exact output-row reuse: population failure
EXP-065 Kronecker rearrangement rank: unfavorable full-rank cuts
EXP-066 exact TT/MPO: storage Gate failure
EXP-067 joint Q/K/V arithmetic: zero reusable rows
EXP-068 absolute-unread demand certificate: output-head-only p50 Gate failure
EXP-069 temporal exact span replay: mandatory p50/p90 full passes 100%
EXP-070 Q4 local-pattern tables: operation 88-91%; bytes/storage exceeded dense
EXP-071 known online-Mv lower bounds: insufficient for an 8 GiB model-wide impossibility claim
EXP-072A self-contained exact Q4 DAG: universal hot-artifact information-capacity failure
```

These results do not prove the final objective feasible or impossible. They close only the registered mechanisms or claims. Permanent restrictions are in `FAILED_APPROACHES.md` and `FAILED_APPROACHES_RECENT.md`; machine-readable authority is under `results/exp_NNN/`.

## Recent authoritative closures

### EXP-066 — TT/MPO bond-rank core

```text
144 dense projections
4,384 preregistered plans
operation p50/p90 3.8941% / 6.7788%
storage p50/p90 11.0524% / 22.9883%
```

Decision:

```text
REJECT_REAL_Q4_TT_MPO_BOND_RANK_AS_CORE_RETAIN_MPO_CERTIFIER_AUXILIARY
```

Authority: `results/exp_066/summary.json`.

### EXP-067 — Joint Q/K/V exact arithmetic

```text
24 complete Q/K/V groups
10,752 Q4 rows
exact reusable rows 0
operation p50/p90 100% / 100%
storage p50/p90 107.4142% / 114.1204%
```

Decision:

```text
REJECT_REAL_Q4_EXACT_JOINT_ROW_REUSE_AS_CORE_RETAIN_GROUP_CERTIFIER_AUXILIARY
```

Authority: `results/exp_067/summary.json`.

### EXP-068 — Absolute-unread demand certificates

After granting all preceding Transformer work, the winning output-head row, metadata, and an independently optimal reveal order for each competitor for free:

```text
output-head-only p50 weight/operation lower bound 13.7697%
p90 lower bound 19.2524%
```

Decision:

```text
REJECT_GLOBAL_DEMAND_CERTIFICATE_AS_CORE_RETAIN_BOUND_AUDITOR_AUXILIARY
```

Authority: `results/exp_068/summary.json`.

### EXP-069 — Causal exact temporal-span replay

```text
147 registered projections
833 warm projection traces
p50/p90 mandatory weight and operation fractions 100% / 100%
TinyStories-1M/3M/8M model p50 69.244% / 100% / 100%
verified exact replay hits 0
p50 basis cache / Q4 projection population 391.97%
```

Decision:

```text
REJECT_CAUSAL_EXACT_TEMPORAL_SPAN_REPLAY_AS_CORE
RETAIN_DYADIC_RANK_AUDITOR_AUXILIARY
```

Authority: `results/exp_069/summary.json`.

### EXP-070 — Exact Q4 local-pattern table circuits

```text
3 models
144 dense projections
3,024 plans
checksum/reconstruction/collision/control failures 0
operation p50/p90 88.4856% / 91.4423%
query-byte and static p50/p90 111.0294% / 112.7907%
minimum joint worst-axis fraction 105.4244%
```

Decision:

```text
REJECT_EXACT_Q4_LOCAL_PATTERN_TABLE_AS_CORE
RETAIN_BLOCK_PATTERN_ANALYZER_AUXILIARY
```

Authority: `results/exp_070/summary.json`.

### EXP-071 — Exact dense-runtime lower-bound applicability audit

EXP-071 directly audited CGL15 Theorem 3 and CKL18 Theorems 1.2/1.3 before allowing an impossibility claim.

Integrity:

```text
3 theorem statements registered
1,052,740 exhaustive binary reduction cases
4,164 direct float32 replay cases
reduction mismatches 0
control failures 0
9 Llama-405B tensor families
884 tensor instances
405,849,243,648 parameters
```

Applicability:

```text
largest valid square subproblem n          16,384
CKL18 maximum registered side state        8 MiB
VORTEX hot/side-state allowance            8 GiB
ratio                                      1,024x
covered tensor families                    0 / 9
required model-wide direct sum             not established
finite Omega constants                     unavailable
```

Decision:

```text
INSUFFICIENT_LOWER_BOUND_DO_NOT_CLAIM_IMPOSSIBILITY
```

This does not establish feasibility. It prohibits dividing the shared 8 GiB state by tensor count, summing per-matrix asymptotic bounds without a direct-sum theorem, or equating cell probes with hardware transactions.

Authority:

```text
results/exp_071/summary.json
results/exp_071/raw/theorem_hypotheses.jsonl
results/exp_071/raw/tensor_rows.jsonl
results/exp_071/processed/direct_sum_audit.json
workflow 30965323458
artifact 8914506737
artifact ZIP SHA-256 bc81e90e3b5a35935f893ad7396d4b41a13de46606ce14bccc53cf79e30e8ba4
```

### EXP-072A — Self-contained exact Q4 DAG information capacity

Exact output on standard-basis activations uniquely recovers every Q4 coefficient. Exhaustive finite-domain validation found 272 unique maps for 272 matrices with zero signature collision or control failure.

Registered target arithmetic:

```text
405B Q4 information                  1,623,396,974,592 bits = 188.98828125 GiB
8 GiB hot allowance                    68,719,476,736 bits = 4.2331%
worst-case required / hot allowance    23.62353515625x
former 10% static threshold            18.898828125 GiB, does not fit hot state
```

Decision:

```text
REJECT_SELF_CONTAINED_EXACT_Q4_DAG_AS_UNIVERSAL_CORE
RETAIN_RESTRICTED_SYNTHESIS_AUXILIARY
```

This closes only an exact artifact that contains all checkpoint-derived information needed by the query inside the hot envelope. A runtime that probes the original checkpoint or another lossless cold representation remains unruled-out and must charge every probe.

Authority: `results/exp_072a/summary.json`, source commit `468f297925e10bdc541fe48f19c2f72a1e3f5e14`, evidence commit `f9ac26befb01fd9a71c7c6e1efed4c4b4df31389`.

## Auxiliary infrastructure retained

- checksum and provenance tooling;
- exact modular-rank and witness certifiers;
- Kronecker/MPO and dyadic temporal-rank auditors;
- exact row/group/block-pattern analyzers;
- theorem-hypothesis and exact-reduction auditor;
- exact verifier and fail-closed fallback components;
- adversarial random, forced-unique, structured, recurrence, and late-flip controls.

Auxiliary classification does not mean the final runtime objective is achieved.

## Primary unresolved bottleneck

No tested mechanism supplies a universal exact way to avoid almost all dense weight information and arithmetic. EXP-072A now rules out a universally compact self-contained hot artifact, while EXP-071 still leaves cold-backed online query algorithms formally open.

EXP-073 Stage 1 now fixes part of the physical contract. The sanitized target inventory measured one Quadro M5000 with 8,192 MiB total VRAM, compute capability 5.2, 8,058 MiB free at the inventory snapshot, and a maximum exposed PCIe Gen2 x16 link. Host RAM is 23.4983 GiB. The root filesystem has 97.6183 GiB available on one non-rotational ATA block device. Python 3.12.3 and Ollama 0.30.6 are present, Ollama is active, and fio/nvcc are not available.

These are inventory facts, not performance measurements. Root free capacity is `91.3700 GiB` below the registered `188.9883 GiB` packed 405B-Q4 information size before scales or other overhead. A favorable PCIe Gen2 x16 signaling calculation gives at most `7.4506 GiB/s`, so one full packed-Q4-equivalent host-to-device transfer would take at least `25.3656 s`; actual transfer behavior remains unmeasured.

## Current frontier

`EXP-073 — Private Ubuntu target resource-contract calibration`, specified in `NEXT_EXPERIMENT.md`.

Stage 1 read-only inventory is complete with zero saved private identifier and no authorized mutation. Authority: `results/exp_073/summary.json`; source `d3b1d2e4dd08e73781c969814cb4d181377a054d`; checkout-stable evidence `4e35afb9648dc0c513c3a90c32d604f4c0b0fd21`; deterministic core SHA-256 `aa9cae0457a6b92fcb75da35fedc1a2a2a9f3da115808d341a5a498ca4722da2`.

Stage 2 is not authorized by Stage 1 success. It remains a separately approved same-machine storage, transfer, and native 4B Q4 baseline measurement. No model download, package installation, service restart, benchmark file allocation, inference, or 405B allocation occurred in Stage 1.

The measured capacity/link envelope already replaces the corresponding proxy numbers in future cold-backed E0 Gates. Storage bandwidth, H2D bandwidth, native 4B Q4 latency, loaded-link behavior, peak process VRAM/RSS, Phase-D runtime validation, actual operation replacement, 405B execution, and E6/E7 remain `NOT TESTED`.

## EXP-074 authoritative no-download budget Gate

EXP-074 evaluated checkpoint-native MTP plus expert-stationary block verification
for the proposed Qwen3.5-122B-A10B surrogate without downloading weights or
contacting the target server.

Registered external model facts were `122B` total, `10B` activated, 48 layers,
256 experts, and `8 routed + 1 shared` experts per token. Deterministic reference
accounting produced 108 scenario rows, 18 minimum searches, and seven passing
controls.

Derived p50 perfect-acceptance requirements:

```text
fixed experts, zero-cost proposal                  9 tokens
fixed experts, 0.8B-equivalent proposal           25 tokens
fixed experts, 1.0B-equivalent proposal           50 tokens
independent-uniform expected experts, zero draft  98 tokens
maximally distinct experts, zero draft           102 tokens
```

MTP-1 plus expert paging required `10.0x` the native 1B weight-equivalent
traffic even with a free proposal and perfect fixed expert reuse, so it failed
the `1.2x` p50 Gate. The long-block hypothesis remains only a revised candidate
because the optimistic fixed-route/free-proposal ceiling reaches the Gate at
nine perfectly accepted tokens. No real proposal acceptance or expert route
locality has been measured.

The 81 GB decimal Ollama artifact nominally fits the measured root free space,
but leaves only `22.1812 GiB`, below the registered `30 GiB` safe-workspace Gate.

Decision:

```text
REVISE_MTP1_AND_EXPERT_PAGING_INSUFFICIENT_REQUIRE_LONG_CAUSAL_PROPOSAL_AND_ROUTING_LOCALITY_GATES
```

Authority: `results/exp_074/summary.json`; source
`8abc06e73c884b839927cf41d5f4fa6cbb8fc051`; evidence
`c1d778af011672ec7fadfa66935ba2548de8e115`; deterministic core SHA-256
`404b43088448eaafc3f3d9631cdc3271dc9ebc16b635e27c9211cf9aa0459a65`.

Current frontier: audit whether an unchanged small Qwen3.5 checkpoint actually
contains a usable causal MTP proposal surface, then measure accepted-prefix
length before any 35B/122B download or page scheduler. The MoE surrogate may
screen infrastructure and scheduling ideas but cannot validate the arbitrary
dense 405B mission. EXP-073 Stage 2 remains separately authorized and not run.

## EXP-075 authoritative native-MTP surface audit

EXP-075 replaced the assumed MTP surface with pinned public metadata from
`Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17` and vLLM revision
`a07086e4032e66aacae60ac2fc01e738096e9569`.

The bounded audit fetched six UTF-8 metadata/source files totaling 255,779
bytes. The checkpoint config declares one MTP hidden layer, its safetensors
index contains the exact registered set of 15 `mtp.*` tensors, and the pinned
vLLM source contains the registered model mapping, loader remap, class registry,
quantization inheritance, and recursive speculative-step surface. Five controls
passed with zero failure.

Decision:

```text
PROMOTE_TO_PINNED_QWEN35_08B_ACCEPTED_PREFIX_GATE
```

This is E1 static-interface evidence only. The full checkpoint index declares
`1,746,882,752` bytes (`1.6269113421 GiB`), but no payload was downloaded. No
proposal token, target verification, quantized path, model execution, server
command, latency, or hardware metric was produced.

Authority: `results/exp_075/summary.json`; source
`f85ac583a129070247992987d1b3c63634e6447f`; evidence
`2fcf7315cf9da491a5ca361536eb0f07e325e74c`; deterministic core SHA-256
`2515bb53a0e2967cfd23e15d18720142337c7d25dc658db057e86e1aa45b5674`.

Current frontier: preregister and run the smallest unchanged checkpoint
accepted-prefix Gate. It must measure causal recursive MTP proposals on exact
prefixes, charge MTP/LM-head/target verification/rejection/fallback work, and
report accepted-prefix p05/p50 rather than citing configured draft length. The
35B/122B checkpoints and expert scheduler remain prohibited. The Qwen-specific
branch remains auxiliary to the arbitrary dense 405B mission.

## EXP-076 authoritative native-MTP accepted-prefix Gate

EXP-076 loaded the pinned unchanged Qwen3.5-0.8B BF16 checkpoint and executed
a causal CPU reference of the registered native MTP equations. Six build
prompts selected `K=4`; the disjoint 18-prompt evaluation population produced
accepted-prefix p05/p50/p95 of `0/4/4`, including two zero-accept cases.

The shape audit charged 274,732,544 proposal parameters per position, including
the tied LM head. The necessary population minima were p50 `11` and p05 `9`.
The observed p50 was therefore insufficient even before the fail-closed
traffic result from zero acceptance. English, Korean, code, structured JSON,
math, and adversarial families did not all pass.

Execution integrity was clean: zero future-target reads, wrong accepts,
committed-prefix mutations, commit-replay mismatches, and rollback-recompute
mismatches. This validates the reference measurement controls, not vLLM or GPU
runtime equivalence.

Decision:

```text
REJECT_NATIVE_MTP_LONG_BLOCK_AS_SURROGATE_CORE
```

Authority: `results/exp_076/summary.json`; source
`5e331137f8e03250cc74aa796abbf69f49ef87a5`; evidence
`55b79937c1f21887ae76b7e56ad61ba7dde8322a`; deterministic core SHA-256
`199db6f8fc0dedd32d7b38be8ff1d05c29aced3388a87ddecccd1235038bd22d`.

The 35B-A3B route-union trace, 35B/122B downloads, expert scheduler, and target
server work are not authorized by this result. No current core candidate has
survived the `1.185185%` target-equivalent E0/Gate sequence. The only already
defined physical next measurement is separately authorized EXP-073 Stage 2;
it is calibration, not a solution. Phase D, E6, and E7 remain not achieved.

## EXP-077A pre-result activation-informed Fractal MLP contract

EXP-077A admitted a new E0 candidate under the following frozen contract. It uses
the current causal token's nonzero SwiGLU intermediate contribution as an
input-conditioned information source and asks whether the best 10% favorable
oracle subset preserves unchanged-target logits on the pinned Qwen3.5-0.8B
population.

This is only the necessary inner `10B activated -> 1B` surrogate premise for
Qwen3.5-122B-A10B. The oracle reads the complete MLP intermediate and receives
selector/full gate-up cost for free. Attention, DeltaNet, LM head, fallback,
cold storage, traffic, VRAM, and large-model scaling remain unverified. No
larger checkpoint or target-server action is authorized. The pre-result Gate
is frozen in `docs/research/EXPERIMENT_077A_ORACLE_FRACTAL_MLP_GATE.md`.

## EXP-077A authoritative Fractal MLP rejection

The corrected unchanged-checkpoint causal-cache replay passed its control with
zero mismatch across 192 registered positions. The favorable oracle kept 358
of 3,584 intermediate channels (`9.988839%`) but held-out top-1 agreement was
`71.5278%`, mean KL `0.884161`, and p95 KL `3.080752`. Every required family
missed its `95%` top-1 Gate. The 20% arm reached only `83.3333%` top-1.

Decision:

```text
REJECT_ACTIVATION_NORM_FRACTAL_MLP_10PCT_PATH
```

Authority: `results/exp_077a/summary.json`; source
`33fed17ed6abed7c8c14eec543efc620e4fe537d`; evidence
`0970c6626ff848c5026b684b3e2d1bb479e96603`; deterministic core
`e25083693c6db21a0da16c9305816958b149865f2fcfde0bd9b7e95c43022411`.

This closes activation-norm individual-channel fracturing under a free,
non-deployable oracle. It does not prove all dynamic sparsity impossible, but
no fraction/layer/router/larger-model rescue of this score is authorized. No
current core candidate survives. Phase D/E2-E7 and dense-405B execution remain
not achieved.

## EXP-078A preregistered frozen tangent-macroblock Gate

EXP-078A tests a different execution dependency: the complete input-conditioned
linear map induced by the last exact prompt token is reused on later causal
activations. It neither chooses MLP channels nor replays an earlier output.

The corrected E0 cost equation charges direct construction of
`W_down diag(c_anchor) W_up`. For the 0.8B checkpoint, the hot map is
`9.52381%` of exact MLP MACs but direct construction costs `341.3333` exact-MLP
token-equivalents. The registered p50/p95 allowances therefore require at least
`13,821/6,249` hot tokens. For the nine-path 122B surrogate screen, the hot map
is `11.1111%`, direct construction costs `1,024` token-equivalents, and the
requirements are `115,299/26,354`.

Before any long trace or physical matrix construction, the unchanged 0.8B
checkpoint will reuse an exact causal anchor for the next seven frozen EXP-076
positions. An early population/family/top-1/KL failure rejects this frozen-map
form. Perfect seven-token survival is right-censored and cannot promote it.

Status: PREREGISTERED E1 FAVORABLE LIFETIME GATE; NO EXP-078A MODEL RESULT;
NO TARGET SERVER, LARGE CHECKPOINT, PHYSICAL KERNEL, E2-E7, OR DENSE-405B CLAIM.

## EXP-078A authoritative frozen tangent-macroblock rejection

The unchanged checkpoint reproduced all 192 registered baseline decisions.
Freezing every layer's complete MLP operator at the exact last prompt token then
matched zero of 126 held-out later-token top-1 decisions. Every one of the 18
evaluation cases failed at its first reuse position; mean/p95 KL was
`14.898423/25.335417`, and MLP output relative-L2 p50/p95 was
`0.354193/0.511419`.

Decision:

```text
REJECT_FROZEN_TANGENT_MACROBLOCK_REUSE_PATH
```

Authority: `results/exp_078a/summary.json`; source
`cc368190031d87db92f7a476dd741c18224239c1`; evidence `fe60819`;
deterministic core `c743ae14748effaad3a034def7d8d92e67fa09abf65eeb931bef3e8a26368667`.

This closes the unchanged prior-token operator, not every causal operator-update
scheme. No core candidate survives. A next candidate must introduce a cheap
causal delta/correction dependency and pass its construction-amortization E0
Gate before code. Phase D/E2-E7 and dense-405B execution remain not achieved.

## EXP-079A preregistered causal proof-state Gate

EXP-079A admits a cold-backed execution dependency at E0. Four procedural DCT
directions per small-checkpoint MLP down projection form the hot center; exact
unread residuals remain row-block L2 proof balls and can be resolved through
charged original row reads. No future target token, training, adapter, or
checkpoint modification is used.

The Gate is deliberately more favorable than a runtime: every non-down
operation, the complete residual used by a quality-optimal selector, a separate
proof-radius-optimal selector, nonlinear propagation, and failed fallback are
free. The authoritative p50 arm still must meet the final `1.185185%` logical
fraction plus the frozen 99%/95%-family/KL/local-radius thresholds on the
unchanged Qwen3.5-0.8B trace.

Status: PREREGISTERED E1 FAVORABLE-ORACLE GATE; NO EXP-079A MODEL RESULT. NO
CORE CANDIDATE IS PROMOTED. TARGET SERVER, DOWNLOAD, CUDA, PHYSICAL 8 GIB,
WALL-CLOCK, 122B/405B, AND E2-E7 ARE NOT AUTHORIZED OR TESTED.

## EXP-079A authoritative causal proof-state rejection

The unchanged checkpoint matched all 192 frozen decisions. The p50 plan used 23
of 256 row blocks in every MLP down projection and stayed inside the logical
traffic/operation fractions. With every complete current residual exposed to a
free selector, held-out top-1 was still only `6/144`, mean KL `7.544862`, and
p95 KL `12.616226`.

The independent proof-radius oracle left a p50/p95 minimum sound enclosure
`48.663918x/57.778748x` the exact MLP-output signal. The p95 allowance reached
only `14/144` top-1 and a `46.993979x` median radius.

Decision:

```text
REJECT_DCT_BLOCK_ZONOTOPE_CAUSAL_PROOF_PATH
```

This closes the registered fixed pilot and row-block L2 enclosure, not every
cold-backed proof system. No core candidate survives. No target server, model
download, physical Q4/CUDA/8 GiB/speed run, 122B/405B execution, or E2-E7
evidence was produced.

Authority: `results/exp_079a/summary.json`; source `e0c661e`; evidence
`38bfd6c`; deterministic core
`c8d794bea8f17b31e40b9f667836d2198ce80137e23080260f81ea6866112eeb`.

## EXP-080A Hyperblock arithmetic Gate

EXP-080A does not reopen the rejected Jacobi, target-only fixed-point,
external-draft, or native-MTP proposal families. It grants a non-deployable
perfect block of future activations and tests one new asymptotic mechanism:
exact rectangular fast matrix multiplication across token columns.

The frozen Llama-405B tensor-shape population is tiled at block lengths
`32..16,384`. A constructive Strassen arm charges every scalar multiplication,
addition, padding tile, and cross-inner-block accumulation while granting the
best classical leaf width for free. A unit-constant `omega=2.371552` arm is
reported separately as a non-constructive theoretical control.

Promotion requires one constructive block length to meet both the final p50
traffic and arithmetic fraction under an already impossible zero-cost perfect
future oracle, fit a favorable 8 GiB workspace equation, and pass exact signed-
integer controls. Passing only the exponent oracle cannot promote the path.

All 80 signed-integer controls matched. No constructive joint pass existed for
any registered `K=32..16,384`. At the best constructive row, `K=16,384`, one
perfect target sweep reduced logical weight traffic to `0.006103516%` and the
favorable workspace equation to `7.539063 GiB`, but fully charged standard
Strassen arithmetic remained `36.111580%` of dense execution. The final p50
allowance is `1.185185%`, leaving a `30.469145x` arithmetic miss even before
layout, kernel, quantized reduction-order, or causal-production costs.

The unit-constant `omega=2.371552` diagnostic passed at `K=512`, or `K=8,192`
after charging a streamed 4B draft. It is not a constructive algorithm and does
not supply future causal activations. Prior causal evidence reaches at most six
target-only fixed-point positions or three external-draft positions.

Decision:

```text
REJECT_STANDARD_STRASSEN_HYPERBLOCK_AS_CORE_RETAIN_COST_MODEL_AUXILIARY
```

This closes standard recursive Strassen under the registered Hyperblock
interface, not every possible exact low-constant rectangular algorithm. No core
candidate survives. No target server, model download, physical Q4/CUDA/8 GiB
peak/speed run, 122B/405B execution, or E2-E7 evidence was produced.

Authority: `results/exp_080a/summary.json`; source `7f1c661`; evidence
`98e9089`; deterministic core
`7578c4c9f463da8135f3c320df9d7fb920ffc172d31fdd2f60b30af9778280ce`.

## EXP-081A authoritative syndrome-recovered lookup rejection

The finite-field reference passed 321/321 in-code, fault-injection, replay, and
singular-state controls. Metadata-complete 405B shape accounting was favorable
before fallback: `0.2148841982%` logical operations, `0.9266579409%` logical
traffic, and `3.978126 GiB` sidecar storage.

The required population premise failed. Across six unchanged
Qwen3.5-0.8B projections and 622 held-out causal positions per projection,
weighted exact residual-code coverage was `8.68167203%`, versus the registered
`99.75%`. Every projection had the same `54/622` fraction; family coverage was
only `6.7164%-10.7143%`. Even a favorable rank-eight SVD correction left
relative-L2 p50/p95 `0.351269/1.213828`, versus `0.01/0.05`.

Observed fallback raises derived logical traffic/operations to
`92.244986%/91.533212%` of dense, about `77.83x/77.23x` the final p50 allowance.

Decision:

```text
REJECT_SYNDROME_RECOVERED_LOOKUP_RESIDUAL_CODE_PATH
```

Authority: `results/exp_081a/summary.json`; source `1d3e91f`; evidence
`ee9573d`; deterministic core
`8621f6357536b6fc3396872668484c52103e28d2af8291b96575bc4d007c2ccc`.
An independent run reproduced the core and all seven scientific payload hashes.

This rejects the frozen nonlinear-lookup plus tiny exact residual-code path,
not every distribution-sensitive online data structure. Structurally valid
conditions were established. Large-model performance remains unverified. No
model-wide operation replacement, target-server action, Phase D, or E2-E7
evidence exists.

## Current frontier after EXP-081A

No core candidate survives. The next admissible mechanism must avoid assuming
that a complete local `W x` residual is tiny. The highest-upside untested axis
is an end-to-end, fail-closed **decision certificate** that adaptively reads
cold pages until the original greedy or fixed-RNG sampled token is uniquely
determined, then falls back when it cannot certify. Before an experiment number
or implementation, a favorable exact-activation oracle must show that the
minimum page set needed to certify held-out final token decisions can approach
the `1.185185%` equation after selector, bounds, verification, and fallback.
This is a research direction, not a promoted candidate or feasibility claim.

## E0 correction to the decision-certificate frontier

The direct output-head route is not a new core candidate. EXP-068 already
rejects absolute unread-contribution certificates, while the complete
registered 405B `lm_head` is only `0.520459999%` of non-embedding coefficient
uses. Eliminating it for free leaves `99.479540001%` of dense linear work.

Proof-carrying traces make verification cheap but do not produce the trace. A
six-challenge favorable field verifier is `0.058120260%` logical operations,
`0.266838924%` logical traffic, and `0.427185 GiB` sidecar on the frozen shapes.
The same-machine dense proposer raises the totals to `100.058120260%` and
`100.266838924%`. External computation violates the no-added-hardware scope;
an independent sublinear local proposer would itself be the missing executor.

Decision:

```text
REJECT_OUTPUT_HEAD_ONLY_DECISION_CERTIFICATE_AS_CORE
REJECT_PROOF_CARRYING_TRACE_AS_A_STANDALONE_LOCAL_CORE
```

All values are E0 `DERIVED`; no model or hardware was run.

## EXP-082A preregistered differential spanning-tree Gate

EXP-082A reopens exact row/column differences with a new asymptotic mechanism.
Unlike EXP-064's at-most-32 fixed prototypes, every row or column may be a
parent in an exact Hamming-difference spanning tree. The complete tree computes
`W @ x` for every causal `x`; no selector, future token, or approximate output
is needed. Low VC/Pollard dimension provides a published subquadratic
structural route, while random Q4 matrices remain the strongest counterexample.

The cheapest Stage 1 does not construct a tree. Exact 32-coefficient block
patterns give a collision-free per-edge Hamming lower bound, and the sum of
per-vertex nearest-neighbor bounds divided by two lower-bounds every spanning
tree. Both row and column orientations are granted and the cheaper bound is
used. If weighted coefficient work alone exceeds `1.185185185%`, the path stops
before MST, runtime, kernel, larger model, or hardware work.

Status: E0 PREREGISTERED; NO RESULT, CORE PROMOTION, OPERATION REPLACEMENT,
TARGET-SERVER ACTION, PHASE D, OR E2-E7 EVIDENCE.

## EXP-082A authoritative closure -- differential trees fail before construction

The preregistered Stage-1 lower bound was executed on 21 pinned Qwen3.5-0.8B
Q4 projections from layers 3, 11, and 23. All 72 exact controls passed. Even
after granting one operation for every differing 32-coefficient block, free
tree metadata, free delta values, free activation access, free construction,
and the better of row and column orientations, the weighted coefficient lower
bound was:

```text
certified lower bound       860,087 / 55,050,240 = 1.562367394%
registered final target                            1.185185185%
gap                                                1.318247489x
p50 / p90 matrix bounds                            1.562935965% / 1.564025879%
```

This is a lower bound on every exact terminal-only Hamming spanning tree, not
the cost of one unlucky tree. Stage 2 therefore stopped as preregistered.

Decision:

```text
REJECT_DIFFERENTIAL_SPANNING_TREE_FROM_CERTIFIED_LOWER_BOUND
```

No exact tree compiler/runtime, CUDA kernel, larger model, or target server was
run. E1 structural evidence only; no core candidate survives and Phase D/E2-E7
remain not achieved.

## Persistent research effort after EXP-082A

The user requested continuous research until a positive, evidence-backed
milestone exists. The milestone is not another negative closure: a materially
new Core Candidate must pass E0 resource closure and E1 real-weight
falsification, then reach actual fail-closed operation replacement at E2 on the
pinned small checkpoint. The unchanged E7 mission remains the final objective.

The versioned wayfinding map is
`.scratch/vortex-certain-milestone/map.md`. Its first frontier ticket audits
whether Synthetic Intermediates are genuinely new relative to EXP-053/054/072
before any EXP-083 number or implementation is permitted.

## E0 closure -- static synthetic intermediates are not a new core

The active map's first ticket audited exact Hamming/Steiner trees containing
compiler-created coefficient vectors. A tree edge `u -> v` evaluates
`z_v = z_u + (v-u).x`; expanding its delta is a static exact linear
straight-line program. General shared linear DAGs are more permissive, and
archived EXP-072B already defined those synthetic forms. Storing the same
program cold changes residency but supplies no new query-time information.

EXP-082A alone does not reject Steiner trees: the generic metric inequality
`SMT >= MST/2` turns its favorable terminal-tree certificate into only
`0.781183697%`, below the registered `1.185185185%`. Independent theory is
nevertheless adverse: almost-all random hypercube terminal sets have Steiner
cost near one third of dense, and almost-all binary linear maps require
`Theta(n^2/log n)` fan-in-two linear circuits. These are distributional and
asymptotic E0 evidence, not a finite real-checkpoint certificate.

Decision:

```text
REJECT_STATIC_SYNTHETIC_INTERMEDIATE_TREE_OR_DAG_AS_NEW_CORE
KEEP_QUERY_ADAPTIVE_COLD_BACKED_INFORMATION_SOURCE_OPEN_AT_E0
```

Authority: `docs/research/E0_SYNTHETIC_INTERMEDIATE_CIRCUIT_AUDIT.md`.

No Core Candidate currently survives. The remaining distinct interface is a
Query-Adaptive Cold Source: current causal activation selects a small subset of
lossless cold information while selection, probes, misses, verification,
fallback, traffic, and state are fully charged. It has no algorithm or
resource closure yet. No EXP-083, E1 run, model download, hardware action,
operation replacement, or E2-E7 evidence is authorized by this E0 audit.

## E0 query-adaptive cold-backed feasibility frontier

The complete branch-aware operation/traffic equation is now fixed. For either
normalized resource, common cost `g`, hit cost `h`, miss work `m`, build cost
`kappa`, service life `N`, and exact fast-path coverage `rho` give:

```text
R = g + kappa/N + rho*h + (1-rho)*(m+1).
```

On the registered `403,747,897,344` non-embedding coefficients and
`201,873,948,672` packed-Q4 bytes, the p50 allowance is exactly `8/675`, or
`1.185185185%`. Even a zero-cost path therefore needs `98.814814815%` exact
coverage under the mean conservation Gate. Reserving the retained six-check
verifier traffic raises the minimum to `99.081653739%`. EXP-081A's independent
pre-fallback value reproduces `99.741472756%`.

The p50 payload ceiling is `2.228263889 GiB/token`. With all other costs free,
that permits at most `584,126` 4-KiB pages, `36,507` 64-KiB pages, or `2,281`
1-MiB pages. After the known verifier the limits fall to `452,612`, `28,288`,
and `1,768`. If one equal cold call per 883 matrix instance is conditionally
required, average payload is at most `2.584080 MiB`, or `2.002286 MiB` after
that verifier.

Raw Q4 page omission is not an information source: without a checkpoint-
derived code or certificate for unread coefficients, an indistinguishable
unread weight change can alter `W*x`. A viable source must instead provide at
least `84.375x` useful information amplification in the impossible zero-cost
limit, or `108.891389x` after the known verifier, while meeting the coverage,
state, build, and fallback equations.

Decision:

```text
DERIVE_QUERY_ADAPTIVE_COLD_BACKED_FEASIBILITY_FRONTIER
REJECT_RAW_Q4_PAGE_SELECTION_WITHOUT_OMITTED_CONTRIBUTION_SOURCE
KEEP_CODED_CAUSAL_COLD_SOURCE_OPEN_FOR_INFORMATION_SOURCE_SEARCH
```

Authority: `docs/research/E0_QUERY_ADAPTIVE_COLD_EQUATION.md`. The equation is
E0 derived accounting, not a Core Candidate or execution result. The next map
ticket must find a causal coded information source; no EXP-083, E1, model,
server, hardware, operation replacement, or E2-E7 work is yet authorized.

## E0 causal information source -- Causal Residual Atlas

The committed prefix supplies a concrete coded source that raw page selection
lacked. For exact prefix pairs `(x_s, W x_s)`, causal orthogonalization forms
`Q` and `Z = WQ`. A current activation decomposes exactly as

```text
x = Q a + u
W x = Z a + W u.
```

The new **Causal Residual Atlas** never treats `u` as an approximate hit. It
reveals query-scored cold column pages and encloses every unread block by
`beta_G ||u_G||`, with exact completion or dense fallback when the declared
output cannot be certified. Synthetic/reference controls establish the linear
identity, safe residual bound, corrupt/non-finite rejection, all-page exact
completion, and strict top-1 margin rule.

Exact table enumeration and Boolean cell-probe saturation do not supply the
numerical Transformer source. The finite-semiring table needed for an
`84.375x` operation factor at `n=16,384` has at least a `4.624453296`
preprocessing exponent even under the unrealistically small `K=16` grant.
The deterministic Boolean cell-probe result gives only about `1.414214x`
additional traffic improvement over an already packed 64-bit Boolean matrix
at that finite shape, and its OR witness does not recover signed sums.

One favorable registered screen uses rank 16, two-byte capsule scalars,
64-column pages, a requested `0.2%` cold fraction, and 64 service tokens. Page
rounding raises cold work to `0.349720584%` over `1,009` pages/token. The
capsule is `0.980995178 GiB` and is read as `1.366004944 GiB/token` because
`Q` is needed twice; 64-token amortized logical traffic/operations are
`1.085025716%/0.557958575%`. This creates a narrow nonempty arithmetic window
but requires at least `99.899840530%` traffic-governed certificate coverage.

Decision:

```text
IDENTIFY_CAUSAL_RESIDUAL_ATLAS_AS_A_CONCRETE_CODED_CAUSAL_COLD_SOURCE
AUTHORIZE_ONLY_A_CHEAP_REAL_WEIGHT_RESIDUAL_CERTIFICATE_GATE
NO_SURVIVING_CANDIDATE_OR_POSITIVE_MILESTONE_YET
```

The decisive premise is unmeasured. Orthogonal arrivals, short prompts,
nonlinear bound growth, numerical enclosure, and `1,009` sequential page
requests can each reject it. Authority:
`docs/research/E0_CAUSAL_RESIDUAL_ATLAS_SOURCE.md`. No EXP-083, model download,
Ubuntu command, physical runtime, or E2-E7 work is authorized. The next map
ticket may only preregister the cheapest pinned real-weight favorable-oracle
Gate.

## Preregistered Causal Residual Atlas cheapest Gate

The cheapest real-weight falsification is now frozen without assigning an
experiment number or running the checkpoint. It uses the existing pinned
Qwen3.5-0.8B evaluation split, the first genuine post-prefill decode call
(teacher index 1), and only layer-11 `q_proj` and `down_proj`.

For each of 18 prompts, a prompt-only top-16 SVD basis is fixed before the
current decode activation. Every contiguous 64-column page is enumerated: 16
for `q_proj` and 56 for `down_proj`, or 1,296 exact-reference candidates in
total. The evaluator grants a native-anchored dense center, changes only one
projection at a time, and selects the top-1-preserving page with minimum
target-to-candidate KL. This is an impossible favorable oracle, not a selector.

The traffic equation requires `99.899840530%` token coverage. On 18 token
states that means 18/18 successes, 3/3 in every family, zero failure, and
top-1 preservation in all 36 projection branches. Mean/p95 KL must also be at
most `0.02/0.05`. One token failure leaves only `94.444444%` coverage and
rejects the registered rank-16/page-64 path before bound propagation.

Decision state:

```text
PREREGISTER_CAUSAL_RESIDUAL_ATLAS_CHEAPEST_REAL_WEIGHT_GATE
NO_MODEL_RESULT_OR_EXP_083_NUMBER
```

Authority: `docs/research/CAUSAL_RESIDUAL_ATLAS_CHEAPEST_GATE.md`. Seven pure
contract/oracle tests pass. The next map ticket may execute only this frozen
Gate with the already present payload. No Surviving Candidate, E2 operation
replacement, hardware action, or E3-E7 evidence exists.

## EXP-083A -- favorable real-weight Atlas Gate passed and reproduced

The frozen first-decode Gate executed on the unchanged pinned
Qwen3.5-0.8B payload from source commit
`1c7dd78097beaa7bc159a8f3459451b876a1338e`. The primary and separate
reproduction runs both returned:

```text
PROMOTE_CAUSAL_RESIDUAL_ATLAS_TO_CAUSAL_PAIR_AND_BOUND_GATE
```

MEASURED: all 18 token states, all six families at 3/3, and all 36 layer-11
`q_proj`/`down_proj` branches preserved native top-1. The selected mean/p95 KL
was `0.007225545020063708/0.037660752986209814` against limits `0.02/0.05`.
All 1,296 page candidates ran; 234 controls passed; control, leakage,
malformed-state, and baseline-trace failures were zero. Independent verification
recomputed both bundles and the identical deterministic-core SHA-256
`ab96e6114f44a1c02a01d080848c4ec643a747b63ebfee30c77fda45f3954845`.

DERIVED FROM MEASURED ROWS: 1,234/1,296 (`95.216049%`) individual page
candidates preserved top-1, and every branch had at least two preserving pages.
The q/down branch minima were 8/2. This is the first positive real-weight
necessary-condition result for the Causal Residual Atlas source; it resolves
the question of favorable one-page existence on the registered population.

It does not resolve causal selection. A post-hoc minimum-certified-radius rule
managed only 34/36 branches and 16/18 tokens, with p95 KL `0.075846638542797`.
Even the tightest current selected radius was `19.852750x` the exact unread
error. This audit is exploratory and cannot alter the preregistered pass.

The pass remains an E1 favorable-oracle ceiling. The center uses the native
dense current output, the selector enumerates every page with exact-reference
logits, only one projection is patched at a time, and all remaining work is
dense. Legal pair-built `Z=WQ`, a target-free selector, outward numerical
certification, simultaneous composition, dense-operation replacement,
physical speed/VRAM, 122B/405B, and E2-E7 remain unverified. The next map ticket
may preregister only the legal causal-pair and bound Gate; no Ubuntu or hardware
action is authorized.

Authority: `docs/research/EXPERIMENT_083A_CAUSAL_RESIDUAL_ATLAS_GATE.md`,
`results/exp_083a`, and `results/exp_083a_reproduction`.

## Legal pair and outward-bound Gate preregistered

The executable follow-up is now frozen without assigning an experiment number
or running another checkpoint row. The already observed EXP-083A evaluation
logits are quarantined from method choice. A new SHA-pinned 24-prompt
population, four prompts in each of six families, is the only scientific
population.

The cheapest branch is the first post-prefill call to layer-23 `down_proj`.
Two-pass causal Gram-Schmidt constructs BF16 `Q_hat/Z_hat` from committed
prefix input/image pairs only. The selector receives only the current residual
and chooses the 64-column page with maximum residual energy. A verified static
`beta_W >= ||W||_2` gives

```text
||W[:, unread] u[unread]||_2 <= beta_W ||u[unread]||_2.
```

Pair-image, native FP32-accumulate/BF16-output, selected-page, cast, residual-
addition, final RMSNorm, and LM-head row-margin errors are all outward and
strict. An unresolved margin executes immutable-branch dense completion or
aborts; it cannot commit.

The favorable 405B component equation now charges pair construction, proof
metadata, a full vocabulary bound scan, and verified spectral compilation
amortized over an explicit 20,000,000 checkpoint-service tokens:

```text
charged traffic                         1.093706271797%
charged operations                      0.928746620379%
registered target                       1.185185185185%
traffic-governed minimum coverage      99.908521086612%
component capsule + metadata            0.983090482652 GiB
verified compile work               69,479.621549 dense-token equivalents
```

The finite Gate therefore requires 24/24 tokens, 4/4 per family, zero false
accept, zero fallback, and mean/p95 KL at most `0.02/0.05`. The 7.0169 GiB
unallocated remainder still omits KV, workspaces, page buffers, fallback
overlap, allocator headroom, and physical runtime state; it is not an 8 GiB
peak pass.

Decision state:

```text
PREREGISTER_CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_GATE
NO_NEW_MODEL_RESULT_OR_EXPERIMENT_NUMBER
```

Ten focused reference tests pass. Authority:
`docs/research/CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_GATE.md`. The next map
ticket may execute only this frozen last-down Gate. A pass permits backward
layer/position expansion only; a failure rejects the legal Atlas primary path.
E2-E7, hardware, 122B/405B, and the Fixed Mission remain unachieved.

## EXP-083B -- legal pair/outward Atlas Gate rejected on the first row

The frozen Gate executed exactly once from source HEAD
`336d59b580104af327a466ad57d7b5c2af9e7a37` on the unchanged pinned
Qwen3.5-0.8B payload. It stopped at the first new evaluation prompt, as the
preregistered cheapest-failure rule required.

MEASURED: causal two-pass MGS reached rank 16, the target-free selector chose
page 0, and the candidate/native greedy winner was the same token `21461`.
However, the strict outward certificate was unresolved. Exact dense completion
replayed bitwise, so the row incurred one valid fallback and immediately
violated the required zero-fallback population Gate. Candidate/native KL was
`0.05158216424853682`. All 19 controls passed; control failures, leakage,
malformed states, false accepts, and dense-replay mismatches were zero.

The frozen projection radius was `23.420520066618046`, led by unread residual
radius `16.329857285002372` and pair-image radius `6.237663356197239`; the
observed projection error was `3.5125591928982423`. The candidate pre-RMSNorm
norm was only `10.709112060498093`, so the unread term alone made the sound
ball cross the RMSNorm near-zero region.

DERIVED POST-HOC, NOT USED FOR THE DECISION: the observed final-hidden
difference was `38.90790804031119`, while the candidate top-two margin and
static row norms permit an ideal hidden-ball radius below
`4.197825281093514` even if all numerical rounding is deleted. Thus tightening
only the RMSNorm implementation envelope cannot rescue this row under the same
global L2-ball/LM-row-norm certificate.

The independent verifier performed zero model forwards, passed all evidence
checks, and rebuilt deterministic-core SHA-256
`57e78fd4d7b1bdc6e97c705a5acb2e7e408a1023e38ae933799b00b8e3711ff4`.
Authority: `results/exp_083b` and
`docs/research/EXPERIMENT_083B_LEGAL_PAIR_OUTWARD_LAST_DOWN_GATE.md`.

Decision:

```text
REJECT_CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_PATH
CLOSE_BACKWARD_LAYER_POSITION_EXPANSION_AND_E2_FOR_THIS_MECHANISM
RETURN_TO_NO_SURVIVING_CANDIDATE
```

Do not reopen with another rank, page width, selector score, layer, prompt
subset, spectral slack, or tolerance. A continuation needs a materially new
causal information source and a new fully charged E0 equation. No Ubuntu,
hardware, large-model, E2-E7, or Fixed-Mission success follows from this
negative E1 result.

<!-- E0-POST-ATLAS-CAUSAL-SOURCE-AUDIT -->
## Post-Atlas E0 causal information-source audit

The strongest materially distinct continuation was made explicit as a
**Decision-Directional Source**: compute only signed decision functionals
`v^T W x` instead of reconstructing every `W x`. With orthonormal primal and
dual bases `Q/P`, paid images `WQ/W^T P`, and residuals `u/r`, the exact normal
form is:

```text
v^T W x = v^T(WQ)a + (W^T P b)^T u + r^T W u.
```

The **Bilinear Cross Residual** `r^T W u` is genuine missing information.
The perturbation `Delta W=lambda r u^T` leaves both cached images unchanged
while changing the query, so forward pairs alone reduce to the rejected Atlas
normal form and a dual cache cannot silently drop the cross term.

Two concrete constructors fail the registered finite equation. One fresh
full-model dual direction costs `1/64 = 1.5625%` at the registered 64-token
service life, already above `8/675 = 1.185185185%` with all other work free;
it needs at least 85 tokens with zero common cost, 109 with the retained
verifier, or 999 when added to registered Atlas traffic. A static vocabulary
composite for only one last `down_proj` has `6,829,375,488` coefficients and
is `12.720703125 GiB` even under a favorable two-byte grant; scanning it costs
`6.765979992%`, `5.708795618x` the target. Across 126 downs the grant is
`1.565242767 TiB` and `8.525134790x` complete Q4 traffic.

A read-only, zero-forward diagnostic on the already observed EXP-083B row
also left all `248,319/248,319` competitors unresolved under the obvious
few-exact-directions plus row-norm screen, even when illegally granted the
observed `3.5124788056` candidate/native distance. This is post-hoc evidence,
not a population Gate or a universal exact-MIPS lower bound.

Decision:

```text
REJECT_FORWARD_TRACE_ONLY_POST_ATLAS_SOURCE_AS_ATLAS_NORMAL_FORM
REJECT_DYNAMIC_CAUSAL_DECISION_DUAL_CODE_UNDER_REGISTERED_64_TOKEN_CONSTRUCTOR
REJECT_STATIC_FULL_VOCABULARY_DUAL_SCAN_AS_CORE
KEEP_LOSSLESS_SUBDENSE_BILINEAR_CROSS_RESIDUAL_SOURCE_OPEN_WITH_NO_CONSTRUCTION
RETURN_TO_NO_SURVIVING_CANDIDATE
```

No experiment number, model run, download, server command, hardware stage, or
E2 claim follows. The only open invention frontier is a checkpoint-derived,
lossless, sub-dense source for prompt-dependent `r^T W u` with complete finite
operations, traffic, state, build, verification, miss, and fallback accounting.
Authority: `docs/research/E0_POST_ATLAS_CAUSAL_INFORMATION_SOURCE_AUDIT.md`.

## E0 matrix-local separable cross-residual bound

The next E0 Gate generalized the primal/dual Atlas bases to arbitrary
matrix-local binary linear covering codes. For a matrix `W_i`, left/right code
dimensions `a_i,b_i` require exactly
`a_i*n_i + m_i*b_i - a_i*b_i` independent image bits, while the only granted
raw work is the coordinate cross residual.

A finite sphere-covering witness uses code-rate threshold `3/10` and radius
`3/16`, with `0.3 + H_2(3/16) = 0.9962122601 < 1`. Even granting the entire
`8 GiB` hot target to one favorable binary bit plane, coefficient-weighted
Markov accounting forces at least `43.2653584179%` of the registered
population below that side rate. Its arbitrary Cartesian query pairs then
require at least:

```text
raw cross probes                    6,141,198,336
fraction of dense                   1.52104775688%
complete p50 allowance              1.18518518519%
bound / allowance                   1.28338404487x
hot point where witness stops       9.34710279225 GiB
```

The last quantity is not sufficient state; it is only where this conservative
witness stops rejecting. The bound grants code lookup, decoding, all cached
terms, image reads, verification, fallback, KV/runtime state, and physical
movement for free.

Decision:

```text
REJECT_MATRIX_LOCAL_SEPARABLE_LINEAR_RESIDUAL_CODE_AS_CORE
DO_NOT_REOPEN_ATLAS_WITH_LARGER_OR_COVERING_BASES
KEEP_NONSEPARABLE_GLOBAL_OR_CAUSALLY_RESTRICTED_SOURCE_OPEN
RETURN_TO_NO_SURVIVING_CANDIDATE
```

This is a scoped coefficient-probe theorem. It assumes matrix-local side
images and independently selectable cross-matrix query tuples. It does not
rule out nonlinear or globally shared advice, word-packed probes, or a causal
query restriction demonstrated on real Transformer traces. No model,
checkpoint, server, GPU, E1, or E2 action occurred. Authority:
`docs/research/E0_BILINEAR_CROSS_RESIDUAL_SEPARABLE_CODE_BOUND.md` and
`results/e0_bilinear_cross_residual_frontier`.

## E0 cross-matrix joint-advice localization closure

The next Gate allowed one arbitrary global binary linear advice space over all
883 non-embedding matrices. Exact query algebra now proves **Joint Advice
Localization**: for a block-local query `q_i`, every outside-block component
of the chosen advice vector appears identically in the raw residual and must
be canceled. Zero-outside-probe advice is exactly `U intersect V_i`; those
shortened dimensions sum to at most the global `8 GiB` dimension. For a fixed
outside set `T`, the exposed local dimension is at most
`dim(U intersect V_i)+|T|`.

Projection dimensions themselves cannot be summed. The counterexample
`U={(x,x)}` projects fully to two blocks while having no nonzero shortened
vector in either. Query-dependent outside supports also form a union of local
spaces, not one fixed subspace, so the localization theorem must not be
inflated into a target-sized direct sum.

A separate global span-to-cover theorem gives the strongest certified finite
consequence. The `13/50` covering-radius witness has
`hot_rate+H_2(13/50)=0.996950297239<1`. Since an arbitrary registered block
tuple is a sum of at most `16,384` aligned rank-one tuples, the worst-case
registered query tuple requires at least:

```text
global covering radius              104,974,453,310 coefficients
rank-one tuple span length                      16,384
total coefficient-use lower bound             6,407,133
fraction of dense                    0.00158691427055%
complete p50 allowance               1.18518518519%
allowance / bound                    746.848904934x
```

Nine focused GF(2) controls and all 476 repository tests pass. The result
rejects free reuse of cross-matrix projection rank but is far too weak to
reject a global linear code, and no concrete code closes construction, query,
traffic, state, verification, miss, fallback, or latency. Nonlinear and
data-dependent structures remain outside scope.

Decision:

```text
ESTABLISH_GLOBAL_LINEAR_ADVICE_LOCALIZATION
DO_NOT_CLAIM_TARGET_SIZED_DIRECT_SUM_OR_GENERAL_IMPOSSIBILITY
KEEP_NO_SURVIVING_CANDIDATE
MOVE_NEXT_TO_A_CAUSAL_QUERY_RESTRICTION_CERTIFICATE
```

No experiment number, model/checkpoint execution, Ubuntu action, kernel, or
hardware work occurred. Authority:
`docs/research/E0_CROSS_MATRIX_ADVICE_LOCALITY.md` and
`results/e0_cross_matrix_advice_locality`.

## E0 causal bilinear query-restriction threshold

The next Gate now separates the population question from the still-missing
pair/result source. A **Causal Bilinear Span Ledger** stores aligned
rank-one query factors `q_i=r_i tensor u_i` and cached scalar answers; an exact
rational witness is mandatory for a hit and every other row falls back.

The rank-to-miss theorem is finite and favorable: if `N` held-out query
tensors have certified rank `R`, then even the best future-aware
`B`-dimensional subspace misses at least `max(0,R-B)` rows. This does not
repeat EXP-069, which ranked only activation inputs.

Registered 405B shape accounting gives:

```text
aligned pair coordinates over 883 matrices        39,109,888
favorable factor bytes per coordinate                        2
largest every-token factor-scan span                         23
B=23 common traffic/operations             1.159413233% / 0.281295494%
B=23 component state                                2.104879502 GiB
B=24 traffic                                        above 1.185185185%
```

The equation charges factor scans, Gram/pivot/answer/fingerprint metadata, six
verification checks, and 46 dense-equivalent build passes amortized over 20M
checkpoint-service tokens. It still grants pair extraction, a local
trace/proposal source, native numerical repair, KV/workspace, and fallback
overlap for free. The known general decision-dual VJP is dense, so no Core
Candidate is promoted.

The frozen cheapest E1 screen uses the pinned Qwen3.5-0.8B last `down_proj`,
six build prompts at four decode positions, and 18 held-out prompts at two
positions, for 24/36 rows. Prompt-only primal/dual side rank is 16. A miss is
weighted as one registered 405B down instance. Only four of 36 such misses fit
the remaining traffic, so certified held-out rank `>=28` rejects every
`B<=23` factor-scan ledger even after granting the best post-hoc subspace.

Decision state:

```text
DERIVE_B23_SHAPE_LIMIT_FOR_A_FULL_FACTOR_SCAN
PREREGISTER_LAST_DOWN_CAUSAL_BILINEAR_RANK_GATE_ONLY
KEEP_PAIR_EXTRACTION_AND_NATIVE_NUMERICAL_RECONSTRUCTION_UNSOLVED
KEEP_NO_SURVIVING_CANDIDATE
```

No model row, experiment number, server command, download, kernel, or hardware
action occurred. Authority:
`docs/research/E0_CAUSAL_BILINEAR_QUERY_RESTRICTION.md` and
`results/e0_causal_bilinear_query_restriction`.

## EXP-084A -- causal bilinear factor-span ledger rejected

EXP-084A executed the frozen last-`down_proj` Gate on the unchanged pinned
Qwen3.5-0.8B checkpoint. The first batched-capture attempt correctly failed
closed before any scientific query row because one prompt position did not
bit-match its causal-prefix replay. Its zero-row artifact is preserved. A new
source freeze used only sequential single-token committed-prefix states and
changed no scientific threshold.

The authoritative run passed all `114/114` controls. All 24 build queries were
independent modulo each registered prime, and the automatic ledger retained
the first 23 exact-rationally independent rows. The first five held-out rows
then produced zero exact hits and five exact misses. Every row raised the
fixed-ledger modular rank from 23 to 24 under all three primes; the fifth miss
exceeded the four-fallback allowance and triggered the frozen stop.

Independent verification used zero model forwards and rebuilt deterministic
core `1e79550fb66fe050338b2eedaf069728fd959583052dee2032fdd57f5cd0a7c4`.

Decision:

```text
REJECT_CAUSAL_BILINEAR_FACTOR_SPAN_LEDGER_AS_CORE
DO_NOT_SWEEP_RANK_SIDE_RANK_PRIMES_PROMPTS_OR_POSITIONS
DO_NOT_CLAIM_FULL_HELDOUT_RANK_28; EXECUTION STOPPED AT FIVE MISSES
KEEP_NONLINEAR_OR_IMPLICIT_EXACT_QUERY_CODES_LOGICALLY_OPEN
KEEP_NO_SURVIVING_CANDIDATE
```

This closes the frozen full-factor scanned calibration ledger, not every
possible nonlinear or post-hoc subspace. No pair-extractor, native numerical
semantics, E2 integration, hardware stage, scale claim, or Fixed-Mission
progress is promoted. Authority: `results/exp_084a` and
`docs/research/EXPERIMENT_084A_CAUSAL_BILINEAR_RANK_GATE.md`.

## E0 query-adaptive exact code union rejected as a primary core

The strongest favorable rescue of the factor ledger was analyzed as a
**Trace-Built Bilinear Code Union**. A perfect causal router chooses one small
exact linear leaf for free, so the hit set is nonlinear and only the selected
leaf is scanned. Construction and persistent state for every independent
answer-bearing direction remain charged.

For a globally independent finite query population, leaves of total dimension
`A` can hit at most `A` members. Registered build amortization therefore limits
the most favorable leaf size `b=1` to `A=87,958`, only `0.43979%` of the 20M
service population. Full-fallback coverage must be `99.999992826%`, while the
traffic at that maximum-capacity point is `1.0074538801` dense, or `85.0039x`
the complete p50 target. A one-pass online grant still pays at least one dense
execution per never-repeated independent query.

A zero-forward replay of the frozen EXP-084A arrays also closes partitioning
as a local rescue. The complete 24-row build span has rank 24 under all three
registered primes; every one of the five stored evaluation rows raises it to
25. Any union of leaves generated only from that span therefore has `0/5`
hits.

Decision:

```text
REJECT_TRACE_BUILT_QUERY_ADAPTIVE_LINEAR_CODE_UNION_AS_CORE
RETAIN_REPEATED_QUERY_CODE_UNIONS_AS_AUXILIARY_ONLY
DO_NOT_CLAIM_A_GENERAL_NONLINEAR_OR_CELL_PROBE_LOWER_BOUND
REQUIRE_AN_IMPLICIT_CHECKPOINT_DERIVED_NONLINEAR_SOURCE_NEXT
KEEP_NO_SURVIVING_CANDIDATE
```

This is an E0 conservation result plus a read-only frozen diagnostic. It does
not establish independence of all real causal queries or reject an implicit
nonlinear checkpoint data structure. No new model forward, experiment number,
Ubuntu action, hardware stage, scale claim, or E2-E7 evidence occurred.
Authority: `docs/research/E0_QUERY_ADAPTIVE_CODE_UNION_BOUND.md` and
`results/e0_query_adaptive_code_union`.

## E0 exact-field nonlinear bilinear source class collapsed

The next audit separated algebraic nonlinearity from genuine finite-word
discontinuity. For any bounded exact algebraic program computing
`f_W(r,u)=r^T W u` on an open domain, one full-dimensional branch/probe path
can be fixed. Its rational expression agrees with `f_W` on an open set and is
therefore identical. Baur--Strassen then turns that scalar circuit into a
static circuit for `gradient_r f_W=W u` with at most four times the unit-cost
arithmetic.

At the registered p50 fraction `8/675`, the reduction gives a static MatVec
operation ceiling `32/675 = 4.740740740...%`. This is a novelty containment
result, not a finite impossibility theorem or proof that a static circuit
meets the target. It prevents exact-field branching from being counted as the
new information source demanded after EXP-084A/F-058.

The current bounded-twin-width literature supplies no reopening premise. The
general algorithm is predecessor-row Hamming traversal already covered by
EXP-082A/F-049; ordered rectangle decomposition and lossless grammar MatVec
are static exact schedules already covered by F-050/EXP-072B.

Sixty-four exact rational reference cases across two deliberately nonlinear
cancellation paths reproduced both the scalar answer and `W u` gradient. Five
focused standard-library tests passed. No Transformer forward, checkpoint
mutation, EXP-085 number, Ubuntu command, or hardware action occurred.

Decision:

```text
REJECT_EXACT_FIELD_NONLINEAR_BILINEAR_ARITHMETIC_AS_DISTINCT_CORE_CLASS
KEEP_FINITE_WORD_DISCONTINUOUS_QUERY_SOURCES_OPEN
KEEP_NO_SURVIVING_CANDIDATE
```

Authority:
`docs/research/E0_IMPLICIT_NONLINEAR_BILINEAR_SOURCE_AUDIT.md` and
`results/e0_nonlinear_bilinear_source`.
