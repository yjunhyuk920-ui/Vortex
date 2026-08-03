# Hardware Validation Plan

## Status

Phase D is **NOT TESTED**.

No current result measures a real 405B checkpoint, <=8 GiB GPU execution, CUDA kernels, PCIe traffic, target SSD behavior, TTFT, tokens/second, power, physical block weight reuse, or combined target/draft peak VRAM.

EXP-047R range CPTC, EXP-048 hard Jacobi/partial-layer draft, and EXP-049 target-only continuous fixed-point generation were rejected before Phase-D promotion. Their correctness/verifier/solver components remain auxiliary only.

## Required hardware ladder

```text
1B–3B small real operation-replacement gate
7B–8B representative developer gate
30B–34B
70B
405B dense flagship
```

Minimum developer hardware must include a CUDA-capable GPU constrained to <=8 GiB usable VRAM, measured NVMe storage, sufficient host RAM/storage, and reproducible profiler tooling.

Do not substitute MoE for the dense flagship.

## Checkpoint and software pinning

Before execution record:

- target and draft model IDs;
- exact revisions and license/access state;
- tokenizer revision;
- file manifests and SHA-256;
- VORTEX commit;
- OS/kernel, driver, CUDA, Python lock;
- baseline runtime versions;
- cache/storage paths;
- `nvidia-smi`, Nsight or equivalent, `fio`, `iostat`, `pidstat`, `/usr/bin/time -v`;
- thermal and power state.

Moving `main` is never authoritative.

## Current entry points

```bash
bash experiments/exp_049/future_gpu_run.sh
```

Expected behavior: fail closed and state that no Phase-D backend exists.

EXP-050 may receive a hardware runner only after a fixed external-draft pool survives its small-checkpoint Gate, a causal deployable selector is implemented, and a complete generation loop replaces sequential target decoding.

## Required same-machine baselines

```text
native 4B Q4
standard exact runtime for every tested target
VORTEX exact sequential target baseline
VORTEX external-draft proposal + exact target verification
future-aware exact proposal oracle, labeled non-deployable
```

Prompt, tokenizer, context, decode contract, batch size, and cache state must match.

## Required MEASURED metrics

- cold/warm TTFT;
- p50/p95/p99 time/token and tokens/second;
- exact token agreement or declared quality contract;
- target/draft peak allocated and reserved VRAM;
- target and draft KV bytes;
- host RSS/page faults;
- target/draft/runtime storage bytes;
- SSD bytes, IOPS, latency, queue depth;
- H2D/D2H bytes;
- kernel time/occupancy;
- draft forward count and weight bytes;
- target verification count and weight bytes;
- proposal block length and exact-prefix distribution;
- rejected positions and correction bytes;
- selector cost;
- physical target-weight reuse across block positions;
- energy/power.

## EXP-050-specific obligations

Before any E5/E6/E7 claim:

1. prove the external draft uses no target future token, target-specific training, or hidden reference continuation;
2. charge one complete draft computation per proposed token;
3. measure whether draft weights remain resident and include them in the <=8 GiB total;
4. include both target and draft KV caches, verification buffers, and fallback state;
5. measure exact target verification and correction traffic;
6. use a causal selector fixed before evaluation;
7. report exact-prefix distributions for every family/model, not only selected successes;
8. compare against the 4B-draft/405B-target dynamic requirement;
9. preserve the universal first-token counterexample boundary;
10. reject a restricted-family result as evidence for the arbitrary-model mission.

PROJECTED reference:

```text
405B Q4 target stream: 188.592821 GiB
4B Q4 draft stream: 1.862645 GiB
1.2x 4B allowance: 2.235174 GiB/token
required target-equivalent fraction: 0.01185185185
perfect 4B-draft proposal minimum length: 507 tokens
```

These are parameter-count projections, not hardware measurements.

## Storage/bandwidth characterization

```bash
fio --name=seqread --filename=/path/to/testfile --rw=read --bs=1M --iodepth=32 --direct=1 --size=32G
fio --name=randread4k --filename=/path/to/testfile --rw=randread --bs=4k --iodepth=64 --direct=1 --size=32G
fio --name=randread64k --filename=/path/to/testfile --rw=randread --bs=64k --iodepth=32 --direct=1 --size=32G
```

Record filesystem, mount options, queue depth, cache state, compression, and thermal state.

## Evidence gates

### E5

Same protocol passes medium/large targets with measured quality, prefix distribution, target/draft bytes, correction, selector, memory, and non-degrading scaling compatible with target equations.

### E6

Real target model executes end-to-end with total peak VRAM <=8 GiB and reproducible hashes.

### E7

Real dense 405B, <=8 GiB, original contract preserved, p50 <=1.2x and p95 <=1.5x native 4B Q4, with raw profiler evidence.

## Stop conditions

Stop and record failure for:

- total VRAM >8 GiB;
- target/draft storage failure;
- draft, verification, correction, selector, KV, or cold reads dominate;
- exact output mismatch outside contract;
- target-future/reference leakage;
- target modification/training violation;
- invalid baseline;
- logical stream amortization not realized physically;
- thermal/cache contamination;
- unreproducible command.

## Result layout

```text
results/exp_xxx/raw/
results/exp_xxx/processed/
results/exp_xxx/summary.json
results/exp_xxx/logs/
results/exp_xxx/artifacts/
results/exp_xxx/checksums.sha256
```
