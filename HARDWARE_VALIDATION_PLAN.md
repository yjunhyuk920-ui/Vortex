# Hardware Validation Plan

## Status

Phase D is **NOT TESTED**.

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
