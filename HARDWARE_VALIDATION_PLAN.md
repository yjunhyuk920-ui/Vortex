# Hardware Validation Plan

## Status

Phase D is **NOT TESTED** in the current environment.

This plan is prepared so target hardware can immediately produce evidence instead of requiring a new design cycle.

## Required hardware ladder

### Minimum developer gate

- one CUDA-capable GPU with exactly or at most 8 GiB usable VRAM for the constrained run;
- 64–128 GiB system RAM preferred;
- NVMe SSD with measured sustained and random-read behavior;
- enough disk for runtime, logs, and at least a small/7B checkpoint.

### Scaling gates

- 7B/8B checkpoint: >=20 GiB free disk including temporary formats;
- 30B/34B: >=100 GiB free disk;
- 70B: >=250 GiB free disk;
- 405B Q4-class checkpoint and runtime artifacts: provision at least 1 TiB free NVMe until measured formats establish a lower requirement;
- host RAM targets must be recorded, not assumed.

## Checkpoints

The exact model IDs, revisions, license acceptance, file list, and SHA-256 hashes must be pinned before Phase D execution.

Candidate ladder, subject to availability and license:

```text
1B–3B: TinyLlama or another public dense causal LM
7B–8B: one public Llama/Qwen/Mistral-class dense checkpoint
30B–34B: one public dense checkpoint
70B: one public dense checkpoint
405B: one public dense 405B-class checkpoint
```

Do not substitute a MoE checkpoint for the dense flagship.

## Required software

- pinned Linux distribution and kernel;
- pinned NVIDIA driver, CUDA toolkit, and runtime;
- Python lockfile/environment hash;
- `nvidia-smi`, CUDA profiler, Nsight Systems/Compute where available;
- `fio`, `iostat`, `pidstat`, `/usr/bin/time -v`, and power telemetry;
- baseline runtimes pinned by commit/version;
- VORTEX commit SHA and unmodified checkpoint hashes.

## Installation skeleton

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pytest -q
python scripts/run_validation.py
```

Exact package versions must be generated and committed before the first hardware run.

## Storage and bandwidth characterization

Before model execution:

```bash
fio --name=seqread --filename=/path/to/testfile --rw=read --bs=1M --iodepth=32 --direct=1 --size=32G
fio --name=randread4k --filename=/path/to/testfile --rw=randread --bs=4k --iodepth=64 --direct=1 --size=32G
fio --name=randread64k --filename=/path/to/testfile --rw=randread --bs=64k --iodepth=32 --direct=1 --size=32G
```

Record filesystem, mount options, queue depth, thermal state, and whether the file is cached.

## Baseline commands

Place exact runnable commands in per-experiment `future_gpu_run.sh` files. The final baseline suite must include:

```text
native 4B Q4 runtime on the same machine
exact/standard runtime for each tested target size
VORTEX strict fallback mode
VORTEX probabilistic certified mode, if retained
```

All commands must use identical prompt sets, context limits, decoding policy, and stop rules.

## Required metrics

MEASURED only:

- cold and warm TTFT;
- p50/p95/p99 time/token;
- tokens/second;
- exact token/logit agreement or declared quality metric;
- peak allocated and reserved VRAM;
- host RSS and page faults;
- model/runtime-format disk bytes;
- SSD read bytes, IOPS, queue depth, and latency;
- host-to-device and device-to-host bytes;
- kernel time and occupancy;
- fallback count and bytes;
- certificate accept/reject/wrong-accept counts;
- model forward, layer, and tile execution counts;
- energy/power where available.

## Profiler command skeleton

```bash
/usr/bin/time -v bash experiments/exp_xxx/future_gpu_run.sh
nvidia-smi --query-compute-apps=pid,used_memory --format=csv -lms 100
nsys profile --trace=cuda,nvtx,osrt --stats=true -o results/exp_xxx/raw/nsys_report \
  bash experiments/exp_xxx/future_gpu_run.sh
```

Add target-specific CUPTI/Nsight counters only after the basic run is stable.

## Success criteria

### E5

- same protocol passes at 30B/70B with measured scaling compatible with the target equations;
- no hidden model modification or retraining;
- quality contract maintained;
- all bytes and fallback charged.

### E6

- real target model runs end-to-end with peak VRAM <=8 GiB;
- independently reproducible checkpoint and runtime hashes;
- no unreported OOM retry or alternate hardware path.

### E7

- real 405B dense model;
- peak VRAM <=8 GiB;
- original declared ability/quality preserved;
- p50 warm time/token <=1.2x native 4B Q4 baseline on the same machine;
- p95 <=1.5x baseline;
- TTFT and user-perceived responsiveness meet the committed acceptance threshold;
- raw profiler evidence published.

## Failure and stop conditions

Stop and record failure when:

- peak VRAM exceeds 8 GiB after declared allocator reserve;
- checkpoint/runtime data do not fit provisioned storage;
- fallback or cold reads dominate the target budget;
- wrong certified accepts occur beyond the declared contract;
- quality agreement fails;
- the run uses future tokens or an unmodified-checkpoint violation;
- thermal throttling or cache contamination invalidates measurement;
- the baseline or VORTEX command is not reproducible.

Do not tune on the held-out evaluation prompts after a failure; create a new pre-registered experiment.

## Result locations

```text
results/exp_xxx/raw/
results/exp_xxx/processed/
results/exp_xxx/summary.json
results/exp_xxx/logs/
results/exp_xxx/artifacts/
results/exp_xxx/checksums.sha256
```

Every hardware result must include machine inventory, command line, environment lock, checkpoint hashes, and provenance labels.
