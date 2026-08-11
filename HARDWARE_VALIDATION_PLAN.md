# Hardware Validation Plan

## Status

Phase D is **NOT TESTED**.

A private Ubuntu host with an 8 GiB-class NVIDIA GPU has been identified outside this repository, but VORTEX has not inventoried or benchmarked it. Identification is not evidence. Connection details and private paths must not be committed or uploaded.

No current result measures real 405B execution, total <=8 GiB GPU state, CUDA kernels, PCIe traffic, target SSD, TTFT, tokens/second, power, or physical skipped-layer traffic.

EXP-047R, EXP-048, EXP-049, and EXP-050 core candidates were rejected before hardware promotion. Their correctness/verifier/reference components remain auxiliary.

## Required ladder

```text
1B–3B real operation-replacement gate
7B–8B representative developer gate
30B–34B
70B
405B dense flagship
```

Minimum developer hardware: CUDA GPU constrained to <=8 GiB usable VRAM, measured NVMe, sufficient host RAM/storage, pinned software, and reproducible profilers. Do not substitute MoE for the dense flagship.

## Pinning

Record target model/revision/license/tokenizer/file hashes, VORTEX commit, OS/kernel, driver/CUDA, Python lock, baseline runtime, cache/storage paths, profiler versions, thermal state, and power telemetry.

## Current fail-closed entry point

```bash
bash experiments/exp_050/future_gpu_run.sh
```

Expected: state that no Phase-D external-draft backend exists and exit nonzero.

EXP-051 may receive a hardware runner only after:

- suffix-stable layer oracle survives;
- a sound causal tail certificate is committed;
- actual target blocks are skipped during complete generation;
- exact target output is preserved;
- a full hot-state plan fits <=8 GiB symbolically.

## Same-machine baselines

```text
native 4B Q4
standard exact target runtime
VORTEX exact sequential target
VORTEX certified tail-skip candidate
full-depth target with intermediate probe instrumentation
```

Prompt, tokenizer, context, decode contract, batch, and cache state must match.

## Required MEASURED metrics

- cold/warm TTFT;
- p50/p95/p99 time/token and tokens/second;
- exact token/logit agreement;
- peak allocated/reserved VRAM;
- target KV/work/probe/fallback bytes;
- host RSS/page faults;
- disk/runtime bytes;
- SSD/H2D/D2H traffic;
- executed and skipped block counts;
- layer-weight physical bytes;
- LM-head probe bytes and time;
- selector/certificate cost;
- fallback completion bytes;
- kernel time/occupancy;
- energy/power.

## EXP-051 hardware obligations

Before E4+:

1. prove intermediate-depth hidden/logit alignment against final target;
2. separate non-deployable suffix-stable oracle from real selector;
3. execute omitted target blocks zero times on certified tokens;
4. charge full LM-head probe and every selector/certificate operation;
5. measure whether layer weights are actually not transferred/read;
6. include fallback full-tail execution;
7. report depth/traffic distributions by model/family;
8. compare physical bytes against 1.185185% target-equivalent allowance;
9. preserve late-decision adversarial claim boundary;
10. reject intermediate multi-layer stability as a certificate without a sound omitted-tail bound.

PROJECTED reference:

```text
405B Q4 full stream 188.592821 GiB
4B Q4 baseline 1.862645 GiB
1.2x allowance 2.235174 GiB/token
required target-equivalent fraction 0.01185185185
```

These are not hardware measurements.

## Storage/bandwidth characterization

```bash
fio --name=seqread --filename=/path/to/testfile --rw=read --bs=1M --iodepth=32 --direct=1 --size=32G
fio --name=randread4k --filename=/path/to/testfile --rw=randread --bs=4k --iodepth=64 --direct=1 --size=32G
fio --name=randread64k --filename=/path/to/testfile --rw=randread --bs=64k --iodepth=32 --direct=1 --size=32G
```

Record filesystem, mount options, queue depth, cache/compression, and thermal state.

## Evidence gates

### E5

Same certified operation-replacement protocol passes medium/large targets with quality, physical bytes, fallback, memory, and scaling compatible with target equations.

### E6

Real target model executes end-to-end with total peak VRAM <=8 GiB and reproducible hashes.

### E7

Real dense 405B, <=8 GiB, original contract preserved, p50 <=1.2x and p95 <=1.5x native 4B Q4, with raw profiler evidence.

## Stop conditions

Stop and record failure for VRAM >8 GiB, storage failure, selector/probe/fallback dominance, exact mismatch, future/reference leakage, target modification/training, invalid baseline, logical savings not realized physically, thermal contamination, or unreproducible command.

## Result layout

```text
results/exp_xxx/raw/
results/exp_xxx/processed/
results/exp_xxx/summary.json
results/exp_xxx/logs/
results/exp_xxx/artifacts/
results/exp_xxx/checksums.sha256
```

<!-- EXP-052-AUTHORITATIVE-FINAL -->
## EXP-052/053 hardware boundary

EXP-052 has no Phase-D core route. EXP-053 hardware work is forbidden until a real small-checkpoint operation is replaced exactly and circuit bytes/query work close the 8 GiB and 1.185185% equations. Phase D remains NOT TESTED.

<!-- EXP-053-AUTHORITATIVE-FINAL -->
## EXP-053/054 hardware boundary

EXP-053 has no Phase-D promotion route as core. EXP-054 hardware work is forbidden until a real small-checkpoint operation is replaced exactly and both decision-diagram storage and path probes close the 8 GiB and 1.185185% equations. Phase D remains NOT TESTED.

<!-- EXP-054-AUTHORITATIVE-FINAL -->
## EXP-054/055 hardware boundary

EXP-054 has no Phase-D route as core. EXP-055 hardware work is forbidden until a real small-checkpoint linear operation is replaced exactly and grouped signature bytes/operations close the 8 GiB and 1.185185% equations. Phase D remains NOT TESTED.

<!-- EXP-058-AUTHORITATIVE-FINAL -->
## EXP-058 hardware status

No factor kernel was promoted because the favorable structural lower bound is already 2.0x on every measured matrix. CUDA, factor bytes, PCIe, SSD, TTFT, tokens/sec, power, and 8 GiB residency are NOT TESTED. EXP-059 remains a CPU structural Gate unless its exact displacement-rank thresholds survive.

<!-- EXP-059-AUTHORITATIVE-FINAL -->
## EXP-059 hardware status

No transform kernel was promoted because favorable query work already equals dense work and generator storage exceeds dense storage. FFT/NTT execution, CUDA, PCIe, SSD, TTFT, tokens/sec, power, and 8 GiB residency remain NOT TESTED.

<!-- EXP-060-AUTHORITATIVE-FINAL -->
## EXP-060 hardware status

No sparse GPU kernel was promoted because logical work remained above 69% even for the best matrix and metadata exceeded dense Q4 bytes. CUDA sparse kernels, PCIe, SSD, TTFT, tokens/sec, power, and 8 GiB residency remain NOT TESTED.

<!-- EXP-061-AUTHORITATIVE-FINAL -->
## EXP-061 hardware status

No activation-sparse kernel was promoted because exact-zero density was zero and logical accounting exceeded dense execution. CUDA sparse projection kernels, PCIe, SSD, TTFT, tokens/sec, power, 405B activation statistics, and 8 GiB residency remain NOT TESTED.

<!-- EXP-062-AUTHORITATIVE-FINAL -->
## EXP-062 hardware status

No attention-sparse kernel was promoted because whole-model logical work and bytes exceeded baseline. CUDA attention kernels, physical cache traffic, PCIe, SSD, TTFT, tokens/sec, power, 405B attention statistics, and 8 GiB residency remain NOT TESTED.

<!-- EXP-063-AUTHORITATIVE-FINAL -->
## EXP-063 hardware status

No grouped-attention kernel was promoted. CUDA kernels, physical KV traffic, PCIe, SSD, TTFT, tokens/sec, 405B statistics and 8 GiB residency remain NOT TESTED.

<!-- EXP-064-AUTHORITATIVE-FINAL -->
## EXP-064 hardware status

No output-row kernel was promoted. Q4 output preservation, CUDA implementation, physical memory traffic, PCIe, SSD, TTFT, tokens/sec, 405B execution and 8 GiB residency remain NOT TESTED.

<!-- EXP-065-AUTHORITATIVE-FINAL -->
## EXP-065 hardware status

No Kronecker kernel or exact factor reconstruction was promoted. Q4 output preservation, CUDA, physical traffic, PCIe, SSD, TTFT, tokens/sec, 405B execution and 8 GiB residency remain NOT TESTED.

<!-- EXP-072A-AUTHORITATIVE-FINAL -->
## EXP-072A hardware boundary

No circuit synthesizer or kernel is promoted. The self-contained exact hot-artifact class fails the universal capacity Gate before hardware work: `188.98828125 GiB` worst-case Q4 information versus 8 GiB hot state. This is a derived resource result, not a measured GPU result. Cold-backed online execution remains outside the Gate.

## EXP-073 sanitized target calibration

Stage 1 is read-only and records sanitized inventory only: OS/kernel, CPU, host RAM, GPU/VRAM, driver/runtime, PCIe exposure, block devices/filesystems, free capacity, existing runtimes, and profiler availability. It must not install packages, download models, restart services, stop workloads, or write benchmark files.

Stage 1 completed successfully:

```text
GPU                         Quadro M5000, 8,192 MiB, compute 5.2
snapshot free VRAM          8,058 MiB
driver CUDA API             12.2
PCIe current / maximum      Gen1 x16 / Gen2 x16
host RAM                    23.4983 GiB total, 21.8865 GiB available
local block device          238.4749 GiB, non-rotational ATA
root filesystem             ext4, 233.6702 GiB total, 97.6183 GiB free
Python / Ollama             3.12.3 / 0.30.6, service active
fio / nvcc                  NOT_AVAILABLE / NOT_AVAILABLE
telemetry interfaces        power, temperature, clock available
```

Authority is `results/exp_073/summary.json`. This does not establish CUDA-library compatibility or benchmark stability. The current root free capacity is insufficient for the registered packed 405B-Q4 information by 91.3700 GiB before overhead. Do not download or allocate that checkpoint on this filesystem.

After Stage 1 review and separate authorization, Stage 2 may measure:

```text
native 4B Q4 cold/warm TTFT and p50/p95/p99 time per token
peak VRAM and host RSS
sequential/random local-storage reads using a bounded dedicated test file
host-to-device transfer bandwidth
page faults, power, clock, and thermal state
```

Use the same prompt, tokenizer, context, batch, decode contract, and cache state across comparisons. Sanitize hostnames, addresses, usernames, internal mount names, and keys from all evidence. EXP-073 is calibration only; 405B execution, operation replacement, and E6/E7 remain NOT TESTED.

Stage 2 is still not authorized. fio is absent, so any Stage 2 storage method must be preregistered using already present read-only/runtime facilities or separately approved installation; Stage 1 does not authorize either choice.

## EXP-074 hardware boundary

EXP-074 executed no target-server command and downloaded no checkpoint. MTP-1
plus expert paging failed the optimistic logical Gate before hardware work, so
no CUDA kernel, expert-page scheduler, 35B/122B run, or inference benchmark is
promoted.

The 81 GB decimal Q4 artifact nominally fits current root free capacity but
would leave only 22.1812 GiB, below the registered 30 GiB safe-workspace Gate.
Do not pull it on the current root filesystem.

Before hardware work on the revised block candidate:

1. verify native MTP tensor presence and runtime exposure without weights when
   possible;
2. pass a small-checkpoint causal accepted-prefix Gate;
3. pass a middle-rung MoE router-union trace Gate;
4. complete separately authorized EXP-073 Stage 2 baselines;
5. preregister peak VRAM, rollback/KV, physical bytes, compute, and thermal
   measurements.

Phase-D runtime validation, E4-E7, 122B execution, and dense 405B remain NOT
TESTED.

## EXP-075 hardware boundary

EXP-075 downloaded only 255,779 bytes of public metadata/source text. The
checkpoint index declares `1.6269113421 GiB`, but the payload was not fetched.
No target-server command, runtime installation, inference, CUDA call, or
physical measurement occurred.

EXP-076 may begin on an isolated developer environment only after pinning the
checkpoint file manifest and compatible dependency hashes. It must first use
the smallest unchanged 0.8B checkpoint and publish exact causal proposal,
verification, rollback, RSS, and CPU accounting. A CPU reference result is not
E4 hardware evidence.

The private Quadro M5000 host is not automatically authorized for the newest
vLLM/SGLang stack, and its compute-capability compatibility remains unverified.
Do not install or change services there under EXP-075/076 authority. The 35B
and 122B downloads, expert-route traces, page scheduler, CUDA backend, and
EXP-073 Stage 2 remain separately gated. Phase D/E4-E7 are unchanged.

## EXP-076 hardware boundary

EXP-076 ran only on the developer Windows CPU with the pinned 1.65 GiB-class
BF16 payload. It recorded logical parameter accounting and CPU time; peak RSS
was unavailable, and no CUDA, VRAM, PCIe, SSD, H2D, power, thermal, TTFT, or
tokens-per-second claim was produced.

Because the accepted-prefix Gate failed, no 35B-A3B router trace, 35B/122B
download, expert pager, or GPU speculative runtime is promoted. The private
Ubuntu host was not contacted. EXP-073 Stage 2 remains the only defined target-
hardware measurement and still requires separate authorization and
preregistration. It would establish baselines only, not validate VORTEX.

## EXP-077A hardware boundary

EXP-077A ran only in the developer Windows CPU reference environment using the
already present pinned Qwen3.5-0.8B BF16 payload. It contacted no Ubuntu host,
downloaded no new checkpoint, and measured no CUDA, VRAM, SSD, PCIe, H2D,
power, thermal, TTFT, or token-throughput quantity.

The favorable 10% MLP oracle failed quality before physical implementation.
Therefore no sparse kernel, 35B/122B download, expert router, or target-server
trial is promoted. EXP-073 Stage 2 remains the only specified target-hardware
calibration and still needs separate explicit authorization. Phase D/E4-E7 are
unchanged.

## EXP-078A hardware boundary

EXP-078A ran only in the developer Windows CPU reference environment with the
already present pinned Qwen3.5-0.8B BF16 payload. It downloaded nothing and did
not contact the Ubuntu host. The reported direct macro construction and hot
application values are logical MAC equations, not measured latency, bandwidth,
VRAM, SSD, PCIe, H2D, power, thermal, or token throughput.

Frozen tangent reuse failed at the first later token for every held-out case, so
no dense macro materializer, rank kernel, sentinel, cache-repair runtime,
35B/122B download, or target-server trial is promoted. EXP-073 Stage 2 remains
separate hardware calibration and Phase D/E4-E7 remain unchanged.

## EXP-080A hardware boundary

EXP-080A is a CPU arithmetic-count prototype only. A perfect future activation
block and best Strassen leaf width are favorable grants. No CUDA/Strassen/
bit-sliced kernel, PCIe or SSD transfer, target-server command, checkpoint
download, TTFT, token latency, power, or measured peak VRAM is authorized.

Only a constructive joint arithmetic/traffic pass could authorize a separate
exact packed-kernel micro-Gate. Even then, causal future-block production and
online output latency require independent proof before Phase D. Current status
remains `NOT TESTED`.

### EXP-080A closure

The constructive arithmetic Gate failed: the best registered standard
Strassen row remained `30.469145x` above the final p50 allowance. Therefore no
CUDA kernel, bit-sliced/quantized reduction study, physical 8 GiB peak test,
storage/H2D run, checkpoint download, or Ubuntu-host command is promoted.

The `7.539063 GiB` workspace value is a favorable shape equation only; it omits
model-resident state, KV cache, layout buffers, runtime overhead, and causal
block production. It is not a measured VRAM result. EXP-073 Stage 2 remains the
separate baseline-calibration route and Phase D/E4-E7 remain unchanged.

## EXP-081A hardware boundary

EXP-081A may run only finite-field controls and a CPU small-checkpoint residual
Gate. Its `3.976903 GiB` sidecar and `0.9265250%` traffic values are logical
equations, not allocated VRAM or measured bandwidth. No lookup kernel, GPU
fingerprint, SSD/H2D scheduler, Ubuntu command, or large checkpoint is
authorized until the residual-code coverage Gate passes. Phase D remains
`NOT TESTED`.

### EXP-081A closure

The coverage Gate failed at `8.681672%` versus `99.75%`. Metadata-complete
shape values were `3.978126 GiB` sidecar and `0.92665794%` fast-path traffic,
but observed fallback raises derived traffic to `92.244986%`. Therefore no
lookup, field-arithmetic, fingerprint, storage, H2D, CUDA, VRAM, or target-
Ubuntu measurement is promoted. No command was executed on the private server.
EXP-073 Stage 2 remains a separate calibration option; Phase D/E4-E7 are
unchanged and `NOT TESTED`.

## E0 proof-trace and EXP-082A hardware boundary

The proof-carrying-trace numbers are equations only. No prover, network worker,
GPU, proof library, or Ubuntu command was used. External proving is outside the
no-added-hardware target and cannot be counted as local acceleration.

EXP-082A initially authorizes only CPU inspection of the already present pinned
0.8B weight payload. Stage 1 must stop at a certified structural lower bound;
it may not run CUDA, allocate a physical 8 GiB tree sidecar, benchmark SSD/H2D,
contact the Ubuntu host, or download 35B/122B/405B. Hardware work remains gated
behind exact-tree, fully charged logical, output-contract, and operation-
replacement passes. Phase D/E4-E7 remain `NOT TESTED`.

### EXP-082A closure

The pinned CPU structural Gate returned a `1.562367394%` favorable coefficient
lower bound versus the `1.185185185%` target. The preregistered stop rule fired
before tree construction. No CUDA, GPU allocation, physical VRAM/bandwidth,
SSD/H2D, Ubuntu server, larger checkpoint, or 122B/405B action occurred.
Hardware validation is not authorized for this rejected mechanism; Phase D and
E4-E7 remain `NOT TESTED`.

## Static synthetic-circuit E0 boundary

The synthetic-intermediate audit is class reasoning only. A static synthetic
tree/DAG is already covered by archived EXP-072B; moving its artifact to disk
does not authorize SSD/H2D benchmarks, GPU allocation, a CUDA interpreter, the
private Ubuntu host, or a larger model. The recorded `97.6183 GiB` root free
capacity is below the `188.98828125 GiB` favorable lossless Q4 information
floor before circuit metadata. A storage upgrade would remove that capacity
fact only, not establish per-token traffic or latency.

A future Query-Adaptive Cold Source must first pass logical operation, traffic,
state, verification, miss, and fallback Gates. Phase D and E4-E7 remain
`NOT TESTED`; no target-server command is authorized.

## Query-adaptive cold equation hardware boundary

The E0 equation converts the p50 fraction to `2.228263889 GiB/token` over the
registered non-embedding Q4 population. At the favorable EXP-073 PCIe Gen2 x16
signaling ceiling (`7.450580597 GiB/s`), moving that entire allowance across
the link has a `0.299072517 s/token` serialization floor. This is `DERIVED`,
not measured, and is not a rejection because native-4B latency, useful request
parallelism, protocol efficiency, storage bandwidth, and overlap remain
unknown.

The complete Q4 checkpoint still needs `188.988281250 GiB` cold capacity;
EXP-073 root free capacity was `97.6183 GiB`. No storage allocation, page read,
GPU buffer, SSD/H2D probe, native-4B run, CUDA action, download, or private
Ubuntu command occurred. A concrete coded causal information source must first
pass E0 and E1 before hardware promotion; Phase D/E4-E7 remain `NOT TESTED`.

## Causal Residual Atlas hardware boundary

The new source is not authorized for physical validation. Its favorable
rank-16 screen logically reads `1,009` cold column pages and about
`0.657508850 GiB` of rounded Q4 payload per token. The stored two-byte capsule
is `0.980995178 GiB`, but its logical HBM reads are `1.366004944 GiB/token`
because `Q` is used for both coordinates and residuals. These are derived logical
quantities, not SSD, RAM, H2D, or HBM measurements.

Before hardware work, the pinned real-weight Gate must show at least
`99.899840530%` fully declared certificate coverage and a deployable numerical
enclosure. A later plan must then measure request latency/IOPS, page coalescing,
layer-sequential dependency, capsule HBM traffic, fallback tails, complete KV
and workspace peak, and same-machine native-4B p50/p95. No target-server
command, storage allocation, model download, CUDA path, or E4-E7 claim is
authorized by source identification.

## Causal Residual Atlas cheapest-Gate hardware boundary

The preregistered first-decode Gate is a Windows CPU exact-reference
falsification using the already present 0.8B payload. Its 1,296 page candidates
are logical oracle variants, not SSD requests, H2D transfers, GPU page reads,
or measured latency.

Even a complete Gate pass may not contact the Ubuntu host or authorize a cold
runtime. It first requires legal committed-pair `Z` construction, outward
native numerical bounds, broader positions/depths, simultaneous composition,
and fully charged fallback/state equations. The stored capsule, `1,009`
target-page requests, `0.657508850 GiB` target cold payload, native-4B latency,
complete peak VRAM, and physical serialization remain unmeasured. Phase D and
E4-E7 remain `NOT TESTED`.

### EXP-083A execution closure

The Windows CPU favorable oracle passed and reproduced. External lifetime
working-set counters observed `3,513,356,288` bytes for the primary run and
`3,516,215,296` bytes for the clean reproduction. The original internal RSS
field was null because of a Windows ctypes signature defect, now corrected.
These values describe the entire CPU process and do not prove GPU VRAM fit.

Primary/reproduction wall times were `2,153.684/1,290.715` seconds, but the
primary briefly contended with a stale prior attempt and both runs exhaustively
execute all dense oracle candidates. They are explicitly not native-4B or
deployment latency measurements. No SSD request, H2D transfer, GPU allocation,
Ubuntu command, model download, or 122B/405B action occurred.

Hardware remains gated behind legal pair construction, target-free selection,
outward certification, simultaneous operation replacement, and complete
branch-state accounting. Phase D and E4-E7 remain `NOT TESTED`.

## Legal pair/outward Gate hardware boundary

The new preregistration is a CPU/reference correctness Gate only. Its favorable
target equation charges `0.983090483 GiB` of capsule/proof metadata,
`1.093706272%` logical traffic, and `0.928746620%` logical operations after an
explicit 20M-token static-compile amortization. None is a measured GPU, SSD,
PCIe, latency, or peak-memory value.

The remaining `7.016909517 GiB` is not free verified capacity: KV, model/runtime
state, workspaces, selected-page staging, fallback overlap, allocator headroom,
and physical proof execution are absent. The verified spectral compiler's
`69,479.621549` dense-token-equivalent logical work is not measured wall time.

No target Ubuntu command, model download, storage allocation, CUDA action, or
EXP-073 Stage-2 measurement is authorized. Even a 24/24 pass permits only
backward correctness expansion; hardware remains blocked behind simultaneous
E2 operation replacement and a complete peak-state plan. Phase D/E4-E7 remain
`NOT TESTED`.

### EXP-083B hardware closure

The prerequisite correctness Gate failed on its first untouched row with one
valid fallback. Consequently the Causal Residual Atlas path does not advance
to simultaneous E2 replacement, target-server calibration, SSD/H2D profiling,
CUDA implementation, or model-size scaling. The measured Windows CPU peak RSS
was `3,513,516,032` bytes and wall time was `257.4760863` seconds, but these
describe a reference proof/model process and are not GPU VRAM or deployment
latency evidence.

No Ubuntu command, download, storage allocation, GPU action, or EXP-073 Stage-2
measurement occurred. Phase D and E4-E7 remain `NOT TESTED`. A future hardware
plan can reopen only after a materially new causal information source passes
its own E0/E1 correctness Gates and demonstrates actual fail-closed E2
operation replacement.

## Post-Atlas causal-source hardware boundary

The Decision-Directional Source audit is algebraic/reference E0 only. Its six
focused controls and read-only EXP-083B direction screen used CPU arrays and
zero Transformer forwards. They are not latency, throughput, SSD, H2D, GPU
kernel, peak-VRAM, or energy evidence.

Both constructible routes fail before hardware promotion: one dynamic exact
dual direction is already `1.5625%` of dense work/traffic over 64 tokens, and
one favorable static last-down vocabulary table is `12.720703125 GiB` before
runtime state. Consequently no CUDA prototype, target-server calibration,
model download, EXP-073 Stage 2 action, 122B/405B run, or Phase D/E4-E7 test is
authorized. Hardware can reopen only after a concrete lossless Bilinear Cross
Residual source passes E0 and then E1/E2 correctness boundaries.

## Separable cross-residual lower-bound hardware boundary

The new result is a binary coefficient-probe theorem and calculator only. It
does not measure word packing, cache lines, SSD IOPS, H2D/HBM traffic, CUDA,
latency, power, or peak VRAM. The `8 GiB` value is used as an unrealistically
favorable all-side-information grant; no allocation occurred.

Because the scoped logical operation bound is already `1.52104775688%`, no
kernel, target-server calibration, storage upgrade, checkpoint download, or
Phase D work is authorized for matrix-local separable residual codes. Hardware
may reopen only for a materially nonseparable/global or causally restricted
source that first passes its own complete E0 equation, E1 falsification, and
actual fail-closed E2 replacement.

## Cross-matrix advice-locality hardware boundary

Joint Advice Localization is finite binary linear algebra, not hardware
evidence. Its `6,407,133` coefficient-use bound maps to `100,112` ideal 64-bit
words or `12,514` ideal 512-bit lines only if every useful bit is perfectly
packed and co-located. The resulting `800,896` bytes are a payload floor, not
measured address traffic, cache-line reuse, SSD IOPS, PCIe/H2D/HBM movement,
latency, energy, or peak VRAM.

The bound is far below the p50 allowance, while no concrete global code
provides a physical layout or query schedule. Therefore it authorizes neither
a kernel nor target-server calibration. No Ubuntu command, download, storage
mutation, GPU allocation, EXP-073 Stage 2 action, 122B/405B run, or Phase D
measurement occurred. Hardware can reopen only after a causal restriction or
concrete global source passes E0/E1 and actual fail-closed E2 replacement.

## Causal bilinear query-restriction hardware boundary

The span-23 result is a logical E0 calculator and preregistration only. Its
`1,799,054,848` factor bytes/token, `40,618` vector segments,
`28,110,232` ideal 64-byte lines, and `442,129` vector-rounded 4-KiB pages are
address-accounting values, not SSD IOPS, PCIe/H2D/HBM traffic, latency, power,
or peak VRAM measurements.

The `2.104879502 GiB` component state omits KV, runtime/model state, workspaces,
fallback overlap, and allocator headroom. Pair extraction and the local result
source are free favorable oracles; the known general VJP is dense. Therefore
no CUDA prototype, storage allocation, target-server command, EXP-073 Stage 2,
larger download, or Phase D/E4-E7 work is authorized. Hardware can reopen only
after the frozen E1 rank Gate and a paid extractor/native-semantics Gate both
survive actual fail-closed E2 replacement.

## Query-adaptive code-union hardware closure

The code-union result is an exact E0 calculator plus a zero-forward replay of
frozen EXP-084A arrays. Its explicit persistent-state figures range from
`0.185625 TiB` at leaf dimension 23 to `6.259769 TiB` at dimension 1; these
are logical representations, not GPU allocations or measured SSD/PCIe/HBM
traffic. The perfect router, lookup, request latency, pair extractor, native
repair, and fallback overlap were free favorable grants.

Because the maximum-capacity independent-stream point is `85.0039x` the p50
target, even the favorable online one-pass lower bound is `84.6328x`, and the
frozen build-span partition has `0/5` hits, this class is not promoted to a
kernel or physical benchmark. No target Ubuntu command, storage mutation,
download, CUDA action, EXP-073 Stage 2 measurement, GPU allocation, 122B/405B
run, or Phase D/E4-E7 work occurred. Hardware may reopen only after a
materially implicit nonlinear checkpoint-derived source passes its own
complete E0 equation, untouched E1 Gate, and actual fail-closed E2 replacement
with a full peak-state plan.

## Exact-field nonlinear source hardware closure

The path-collapse result is algebraic containment, not a physical benchmark.
It allocates no GPU/CPU/SSD state and measures no bandwidth, latency, power,
or VRAM. Exact-field branching returns to the static arithmetic-DAG family and
therefore authorizes no kernel or target calibration.

No target Ubuntu command, storage mutation, download, CUDA action, EXP-073
Stage 2 measurement, GPU allocation, 122B/405B run, or Phase D/E4-E7 work
occurred. Hardware can reopen only after a concrete bounded-word discontinuous
source passes a complete E0 equation, untouched E1 falsification, and actual
fail-closed E2 replacement with native numerical semantics and a full
peak-state plan.

## Finite-word source audit: no hardware Gate is authorized

The local Block Rank-One Truth Table fails persistent state, build
amortization, and address width before hardware mapping. Its 2.5% query-only
layout would require about 1.628309058482239e14 GiB of table storage and a
78-bit address; the tighter registered layout requires about
6.485398215045125e20 GiB and a 100-bit address. These are representation
failures, not unknown PCIe or GPU constants.

Do not benchmark random reads, allocate a table fragment, build a kernel, or
use a small truncated table as evidence for the arbitrary-checkpoint claim.
No target-server calibration can repair the missing artifact. Hardware work
remains closed until a different finite-word rank-one source passes E0 with a
real 8 GiB peak-state and shared-link plan.

## Global nonlinear theorem audit: no hardware Gate is authorized

The Larsen--Williams leading substitutions (`2 MiB` query traffic and `2 MiB`
redundancy for one `16,384 x 16,384` one-bit square at `w=64`) are asymptotic
cell-probe monomials with hidden constants and free computation. They are not
SSD, PCIe, HBM, latency, power, or peak-VRAM measurements, and their Boolean
nonemptiness output is not the native numerical result needed by the runtime.

The limited-independence, singleton-injection, scalarization, and dynamic-model
results are proof boundaries only. They provide no representation to allocate
and no query kernel to benchmark. Cross-request 40-way full-sweep batching is
both outside the single-stream contract and 32x above its `1/1280` per-token
weight allowance; it cannot authorize a throughput benchmark as a substitute.

No target-server command, storage mutation, download, CUDA action, GPU
allocation, EXP-073 Stage 2, 122B/405B run, or Phase D/E4-E7 work is
authorized. Hardware may reopen only after the active general nonlinear
rank-one ticket produces one of its two required E0 deliverables.

## Finite-semiring preprocessing graph: no hardware Gate is authorized

The favorable physical expansion of Williams' graph already fails persistent
representation, query payload, and numerical semantics before any device
mapping. One registered Boolean square needs `36.616085 GiB` of minimum edge
values; the model-wide one-bit sidecar is `55,292.57 GiB`. Q4 and BF16
single-query values exceed the entire DFloat `1/40` block budget at `b=14`.
These figures omit global addresses, adjacency offsets, counters, second-layer
state, operations, cache lines, pages, and shared-link movement.

The theorem's semiring regrouping is also not a bit-exact BF16/FP32 execution
contract, and no 32-query reuse theorem supplies a hardware schedule. Random
read benchmarking, graph-fragment allocation, or a one-bit toy kernel cannot
repair these E0 failures and would not test the user's claim.

No target-server command, storage mutation, download, CUDA action, GPU
allocation, EXP-073 Stage 2, 122B/405B run, or Phase D/E4-E7 work is
authorized. Hardware may reopen only for a different constructor after its
complete native-order scalar and 32-token equation survives E0 and an
untouched E1 Gate.

## Global-advice synergy audit: no hardware Gate is authorized

The XOR witness, full-square substitution, and 6-by-6 pigeonhole screen are
finite information-theory calculations. They allocate no 8 GiB artifact and
measure no word packing, random access, SSD, PCIe/H2D/HBM traffic, CUDA,
latency, power, or peak VRAM. The invalid per-tile sums are explicitly not a
physical workload or a certified lower bound.

The audit leaves no constructor to benchmark and no covering impossibility
theorem. Random-read tests or a per-matrix CKL prototype would test the
rejected division rather than the open global model. No target-server command,
storage mutation, download, CUDA action, GPU allocation, EXP-073 Stage 2,
122B/405B run, or Phase D/E4-E7 work is authorized. Hardware may reopen only
after a globally coupled constructor or a synergy-charging theorem supplies a
complete E0 deliverable and the subsequent E1/E2 Gates survive.

## Fourier-fiber direct-sum audit: no hardware Gate is authorized

The 26.2% all-linear result is a systematic bit-probe lower bound with free
advice reads, computation, query generation, addressing, packing, and physical
movement. It is not a measured cache line, SSD, PCIe, HBM, CUDA, latency,
power, or VRAM result. Its all-linear query description is also not a standard
Transformer activation interface.

After the rank-one restriction, the finite number is only a ceiling on one
proof method and supplies no executable query schedule. There is therefore no
artifact or kernel to benchmark. No target-server command, storage mutation,
download, CUDA action, GPU allocation, EXP-073 Stage 2, 122B/405B run, or
Phase D/E4-E7 work is authorized. Hardware may reopen only after the active
ticket produces a target-scale restricted-rank theorem or a complete
constructor and survives the later E1/E2 Gates.

## Spiky / entrywise-power audit: no hardware Gate is authorized

The SpikyCut result is a finite description-count rejection with factor
interactions, block reductions, metadata, storage, traffic, addressing,
native arithmetic, and verification granted free.  PowerFold is rejected
before implementation because its exact expansion is a static low-rank
normal form.  Neither result supplies an executable artifact or a physical
performance prediction.

No target-server command, storage mutation, download, CUDA action, GPU
allocation, EXP-073 Stage 2, 122B/405B run, or Phase D/E4-E7 action is
authorized.  Hardware remains downstream of a genuinely new globally
adaptive exact constructor that first survives its complete E0 and E1 Gates.

## Adaptive codebook / trapdoor audit: no hardware Gate is authorized

NearestPair fails a finite storage and address-width Gate before a physical
layout exists. TrapShift leaves one full arbitrary dense product and therefore
defines no executable savings to benchmark. Neither result predicts target
bandwidth, latency, VRAM, or kernel behavior.

No target-server command, storage mutation, download, CUDA action, GPU
allocation, EXP-073 Stage 2, 122B/405B run, or Phase D/E4-E7 action is
authorized. Hardware remains downstream of a nonliteral compressed decoder
that first survives complete E0 and E1 accounting.
