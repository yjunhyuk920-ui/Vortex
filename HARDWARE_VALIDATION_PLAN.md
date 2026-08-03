# Hardware Validation Plan

## Status

Phase D is **NOT TESTED** in the current environment.

EXP-047's E1 correctness result does not alter this status. Current measured CPU fallback and tile fractions are negative architecture evidence, not target hardware data.

## Required hardware ladder

### Minimum developer gate

- CUDA-capable GPU constrained to <=8 GiB usable VRAM;
- 64–128 GiB system RAM preferred;
- measured NVMe SSD;
- enough storage for checkpoint, runtime format, logs, and profiler output.

### Scaling gates

- 1B–3B: current small-checkpoint falsification;
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

## EXP-047 current future-hardware entry point

```bash
VORTEX_MODEL_PATH=/absolute/path/to/checkpoint \
VORTEX_MODEL_REVISION=<pinned-revision> \
VORTEX_CHECKPOINT_MANIFEST=/absolute/path/to/checksums.sha256 \
bash experiments/exp_047/future_gpu_run.sh
```

Current behavior:

- captures strict preflight inventory;
- writes `results/exp_047/raw/future_gpu_preflight.json`;
- exits without pretending to run a model because the real-operation runner is not implemented;
- records Phase D as `NOT TESTED`.

This is intentional. A Phase-D command must fail rather than fabricate a result.

## Storage/bandwidth characterization

```bash
fio --name=seqread --filename=/path/to/testfile --rw=read --bs=1M --iodepth=32 --direct=1 --size=32G
fio --name=randread4k --filename=/path/to/testfile --rw=randread --bs=4k --iodepth=64 --direct=1 --size=32G
fio --name=randread64k --filename=/path/to/testfile --rw=randread --bs=64k --iodepth=32 --direct=1 --size=32G
```

Record filesystem, mount options, queue depth, cache state, and thermal state.

## Baselines

Same machine and prompt/decode protocol:

```text
native 4B Q4
exact/standard runtime for each target size
VORTEX strict fallback mode
VORTEX probabilistic certified mode, if retained
```

## Required MEASURED metrics

- cold/warm TTFT;
- p50/p95/p99 time/token and tokens/second;
- token/logit/quality agreement;
- peak allocated/reserved VRAM;
- host RSS/page faults;
- disk/runtime bytes;
- SSD bytes/IOPS/latency/queue depth;
- H2D/D2H bytes;
- kernel time/occupancy;
- fallback count and bytes;
- certificate accepts/rejects/wrong accepts;
- forward/layer/tile counts;
- energy/power where available.

## EXP-047-specific hardware obligations

If CPTC survives EXP-047R and real small-model replacement:

- compare certificate selector time against actual dense tile kernel time;
- measure random versus coalesced tile access;
- include static bound metadata in VRAM/RAM/SSD totals;
- account for RNG/permutation state and union-budget bookkeeping;
- report fallback stream overlap separately from actual elapsed latency;
- reject if evaluated tile fraction, fallback, or selector overhead cannot plausibly approach the 1.185% pre-overhead traffic fraction.

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

Same protocol passes at medium/large sizes with measured quality, bytes, fallback, and scaling compatible with target equations.

### E6

Real target model executes end-to-end with peak VRAM <=8 GiB and reproducible hashes.

### E7

Real dense 405B, <=8 GiB, original contract preserved, p50 <=1.2x and p95 <=1.5x native 4B Q4, with raw profiler evidence.

## Stop conditions

Stop and record failure for:

- VRAM >8 GiB;
- storage/capacity failure;
- fallback/cold reads dominate;
- wrong certified accepts beyond contract;
- quality failure;
- future-token leakage;
- checkpoint modification/training violation;
- invalid baseline;
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
