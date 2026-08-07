# VORTEX Research State

Last updated: 2026-08-07 Asia/Seoul

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
