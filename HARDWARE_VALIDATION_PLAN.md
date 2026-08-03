# Hardware Validation Plan

## Status

Phase D is **NOT TESTED** in the current environment.

EXP-047/047R produced E1 correctness and negative architecture evidence only. Range-based CPTC was rejected as the core path, so it has no active Phase-D promotion route. No current result measures target GPU, 405B, CUDA, PCIe, SSD, TTFT, tokens/second, power, or peak VRAM.

## Required hardware ladder

### Minimum developer gate

- CUDA-capable GPU constrained to <=8 GiB usable VRAM;
- 64–128 GiB system RAM preferred;
- measured NVMe SSD;
- enough storage for checkpoint, runtime format, logs, and profiler output.

### Scaling gates

- 1B–3B: small-real-checkpoint operation-replacement gate;
- 7B–8B: >=20 GiB working storage;
- 30B–34B: >=100 GiB;
- 70B: >=250 GiB;
- 405B Q4-class checkpoint/runtime/profiler scratch: provision >=1 TiB until measured requirements replace this conservative value.

Do not substitute MoE for the dense flagship.

## Checkpoint requirements

Pin before execution:

- model ID;
- exact revision;
- license/access state;
- tokenizer revision;
- file manifest;
- SHA-256 checksums;
- cache/storage path.

No moving `main` revision is authoritative.

## Required software inventory

- OS/kernel;
- NVIDIA driver/CUDA;
- Python lock/environment hash;
- VORTEX commit;
- baseline runtime versions;
- `nvidia-smi`, Nsight Systems/Compute or equivalent;
- `fio`, `iostat`, `pidstat`, `/usr/bin/time -v`;
- power telemetry where available.

## Installation skeleton

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pytest -q
python scripts/run_validation.py
```

A pinned lockfile must be committed before the first Phase-D run.

## Current experiment entry points

EXP-047R deliberately fails closed for unavailable Phase D:

```bash
bash experiments/exp_047r/future_gpu_run.sh
```

Expected behavior: explain that no real-operation backend exists, retain `Phase D: NOT TESTED`, and exit nonzero rather than fabricate evidence.

EXP-048 may receive a real hardware runner only after its B3 partial-layer self-draft survives the small-checkpoint early rejection Gate and replaces sequential target decoding under exact output verification.

## Storage/bandwidth characterization

```bash
fio --name=seqread --filename=/path/to/testfile --rw=read --bs=1M --iodepth=32 --direct=1 --size=32G
fio --name=randread4k --filename=/path/to/testfile --rw=randread --bs=4k --iodepth=64 --direct=1 --size=32G
fio --name=randread64k --filename=/path/to/testfile --rw=randread --bs=64k --iodepth=32 --direct=1 --size=32G
```

Record filesystem, mount options, queue depth, cache state, thermal state, compressed/uncompressed representation, and page-cache policy.

## Baselines

Same machine, checkpoint revision, prompt set, tokenizer, decoding contract, context length, batch size, and cache state:

```text
native 4B Q4
exact/standard runtime for each tested target size
VORTEX exact sequential B0
VORTEX block-verification oracle B1, explicitly non-deployable
VORTEX charged Jacobi B2
VORTEX causal partial-layer self-draft B3, only after Phase-C promotion
```

Future-aware B1 cannot be reported as deployable performance.

## Required MEASURED metrics

- cold/warm TTFT;
- p50/p95/p99 time/token and tokens/second;
- exact token agreement and declared quality contract;
- peak allocated/reserved VRAM;
- host RSS/page faults;
- disk/runtime bytes;
- SSD bytes/IOPS/latency/queue depth;
- H2D/D2H bytes;
- kernel time/occupancy;
- target full streams and bytes;
- partial-draft layer streams and bytes;
- proposal block length;
- accepted-prefix length;
- rejected scored positions;
- correction/fallback passes and bytes;
- KV cache rebuild/copy bytes;
- target-equivalent streams per accepted token;
- energy/power where available.

## EXP-048-specific hardware obligations

Before any E5/E6/E7 claim:

- prove exact greedy output equality for every committed token;
- verify the deployable proposal path uses no future generated token or reference continuation;
- measure draft generation serial latency separately from target block verification;
- account for all early-layer and output-head reads during draft generation;
- account for all full-target passes, mismatches, corrections, rejected positions, and KV state reconstruction;
- measure whether a single target weight stream is physically reused across the proposed block rather than logically counted once while reread per position;
- report achieved accepted tokens per target stream and target-equivalent stream fraction;
- compare observed traffic against the PROJECTED requirement `<=0.01185185` before claiming target compatibility;
- preserve the exact sequential B0 baseline on the same machine and checkpoint;
- reject the architecture if block verification loses its logical amortization through kernel launches, memory layout, KV traffic, or proposal latency.

Reference projection only:

```text
405B Q4 full stream: 188.592821 GiB
1.2x 4B Q4 allowance: 2.235174 GiB/token
required target-equivalent stream fraction: 1.185185%
zero-cost perfect-proposal minimum: 85 accepted tokens/full target stream
```

Real draft cost increases the required accepted block length. These are not hardware measurements.

## Retired CPTC hardware obligations

CPTC selector/tile-access profiling is no longer a core Phase-D requirement because EXP-047R rejected the range family before operation replacement. It may be profiled only if reused as an auxiliary guard inside a different successful mechanism, with its complete cost charged.

## Profiler skeleton

```bash
/usr/bin/time -v bash experiments/exp_xxx/future_gpu_run.sh
nvidia-smi --query-compute-apps=pid,used_memory --format=csv -lms 100
nsys profile --trace=cuda,nvtx,osrt --stats=true \
  -o results/exp_xxx/raw/nsys_report \
  bash experiments/exp_xxx/future_gpu_run.sh
```

## Evidence gates

### E5

Same exact protocol passes at medium/large sizes with measured quality, accepted-block distribution, physical bytes, correction cost, memory, and non-degrading scaling compatible with target equations.

### E6

Real target model executes end-to-end with peak VRAM <=8 GiB and reproducible hashes.

### E7

Real dense 405B, <=8 GiB, original contract preserved, p50 <=1.2x and p95 <=1.5x native 4B Q4, with raw profiler evidence.

## Stop conditions

Stop and record failure for:

- VRAM >8 GiB;
- storage/capacity failure;
- draft, verification, correction, KV, or cold reads dominate;
- exact output mismatch;
- future-token/reference leakage;
- checkpoint modification/training violation;
- invalid baseline;
- logical stream amortization not realized as physical traffic reduction;
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
