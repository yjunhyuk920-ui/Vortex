# EXP-073 — Private Ubuntu Target Resource-Contract Calibration

## Question

Can the privately identified Ubuntu target provide a complete, sanitized, reproducible resource inventory before VORTEX defines another cold-backed online execution Gate?

EXP-073 is calibration, not a core runtime candidate. It does not test operation replacement, 405B execution, or the final latency target. Its purpose is to replace proxy hardware assumptions with measured target facts while keeping every private connection detail out of the public repository.

## Registered outcomes

Stage 1 has exactly two outcomes:

```text
COMPLETE_SANITIZED_READ_ONLY_TARGET_INVENTORY
INFRASTRUCTURE_LIMITATION_NO_SCIENTIFIC_DECISION
```

Neither outcome promotes a VORTEX execution mechanism. A complete inventory authorizes review only. It does not authorize Stage 2 benchmarks.

## Authorization boundary

Stage 1 may execute only read-only operating-system and runtime queries over an already configured SSH connection. It may read:

- OS name, kernel release, and machine architecture;
- CPU model and logical CPU count;
- total and currently available host RAM;
- GPU count, model, driver, compute capability, total/free VRAM, and exposed PCIe generation/width;
- aggregate root-filesystem and block-device capacity/type information without device or mount names;
- installed Python, Ollama, fio, and CUDA compiler versions or `NOT_AVAILABLE`;
- availability of GPU power, temperature, and clock telemetry;
- Ollama service state when the existing service manager exposes it.

Stage 1 must not:

```text
install or upgrade packages
download a model or checkpoint
start, stop, restart, or reconfigure a service
create a remote file or benchmark allocation
run fio or another load generator
run inference or enumerate private model names
kill, pause, or evict a workload
collect hostname, address, username, SSH key, serial, UUID, device name, or mount path
```

The collector is transmitted over SSH standard input and executed in memory by the existing shell. It creates no remote script. The SSH alias is a runtime argument and is never written to the evidence bundle, log, configuration, or command transcript.

## Sanitization contract

The public evidence schema is an allowlist. Unknown keys are rejected. String values are normalized to one line and rejected if they contain:

- an IPv4 or IPv6 address;
- an email/user-at-host form;
- a private-key marker;
- a filesystem path;
- a hostname, username, device identifier, serial, UUID, or SSH-related field name.

Storage evidence is aggregated locally. Root filesystem type, total bytes, and available bytes are retained; root mount name and backing-device name are discarded. Block devices retain only counts, total capacity, rotational class, and transport-class counts. Raw `nvidia-smi -L`, `lsblk`, `df`, SSH diagnostics, and shell transcripts are not saved.

If sanitization fails, no candidate evidence is committed. The result is an infrastructure limitation until the collector or environment is corrected.

## Registered inventory schema

```text
platform.os_pretty_name
platform.kernel_release
platform.machine_architecture
cpu.model_name
cpu.logical_cpu_count
memory.total_bytes
memory.available_bytes
gpu.count
gpu.models
gpu.driver_versions
gpu.cuda_driver_api_versions
gpu.compute_capabilities
gpu.total_vram_mib
gpu.free_vram_mib
gpu.pcie_current_generation
gpu.pcie_current_width
gpu.pcie_max_generation
gpu.pcie_max_width
storage.root_filesystem_type
storage.root_total_bytes
storage.root_available_bytes
storage.block_device_count
storage.block_total_bytes
storage.rotational_device_count
storage.transport_counts
runtimes.python3_version
runtimes.ollama_version
runtimes.fio_version
runtimes.nvcc_version
runtimes.ollama_service_state
telemetry.gpu_power_available
telemetry.gpu_temperature_available
telemetry.gpu_clock_available
```

Missing optional capability fields are recorded as `NOT_AVAILABLE`; missing required platform, memory, GPU, or storage fields makes Stage 1 incomplete.

## Required controls

- mock collection proves the SSH target is not serialized;
- unknown or forbidden output keys fail closed;
- IP addresses, paths, user-at-host strings, UUIDs, serials, and key markers fail sanitization;
- multiline/control-character values fail or normalize deterministically;
- multi-GPU rows and aggregate block-device rows parse deterministically;
- missing optional tools become `NOT_AVAILABLE` without installation;
- a nonzero SSH exit produces infrastructure limitation and no scientific decision;
- deterministic fixture replay produces the same evidence-core SHA-256.

## Completion Gate

Stage 1 is complete only when all of the following hold:

```text
SSH command exit status = 0
all required inventory fields present and type-valid
all public strings pass the sanitizer
no forbidden field name or private identifier in saved evidence
remote mutation count = 0 by construction of the allowlisted command program
experiment-specific tests pass
repository validation passes
```

The output must distinguish `MEASURED` inventory values from `NOT_AVAILABLE` capability fields. No bandwidth, latency, throughput, power draw, thermal load, or VORTEX improvement may be inferred from inventory alone.

## Stage 2 boundary

Stage 2 remains separately authorized only after the Stage 1 evidence is reviewed. It may then use a bounded dedicated test file and an already present 4B Q4 model to measure storage reads, host-to-device transfer, native baseline TTFT/time-per-token distributions, peak VRAM/RSS, faults, power, and thermal state.

Until that authorization, the following remain prohibited:

```text
fio execution
storage write allocation
CUDA transfer benchmark
Ollama inference
model download
package installation
service restart
405B allocation or execution
```

## Evidence and claim boundary

Target inventory values are `MEASURED` only after a successful Stage 1 run. Completeness decisions and deterministic hashes are `DERIVED`. Runtime performance, target feasibility, and 405B behavior remain `UNVERIFIED`.

This is a Phase-D calibration prerequisite, not Phase-D runtime validation. E6 and E7 remain **NOT ACHIEVED**. The result cannot change the status of any previously rejected mechanism family.

## Stop rule

Stop immediately after the sanitized inventory bundle is frozen and reviewed. Do not proceed to Stage 2 merely because Stage 1 succeeds. If SSH, permissions, or required read-only interfaces fail, record the infrastructure limitation; do not repair the host during this experiment.

## Stage 1 authoritative result

```text
decision                         COMPLETE_SANITIZED_READ_ONLY_TARGET_INVENTORY
GPU                              Quadro M5000, 8,192 MiB, compute 5.2
free VRAM snapshot               8,058 MiB
driver / CUDA API                535.309.01 / 12.2
PCIe current / maximum           Gen1 x16 / Gen2 x16
host RAM total / available       23.4983 / 21.8865 GiB
local block storage              238.4749 GiB, non-rotational ATA
root filesystem total / free     233.6702 / 97.6183 GiB, ext4
Python / Ollama                  3.12.3 / 0.30.6, service active
fio / nvcc                       NOT_AVAILABLE / NOT_AVAILABLE
telemetry interfaces             power, temperature, clock available
validation failures              0
```

Derived consequences:

```text
packed 405B Q4 information              188.9883 GiB
root free-capacity deficit               91.3700 GiB before overhead
favorable PCIe Gen2 x16 ceiling           7.4506 GiB/s
full packed-Q4-equivalent transfer floor 25.3656 s before overhead
```

The current PCIe Gen1 state is an idle snapshot and may upshift; no loaded-link measurement was made. The Gen2 number is a favorable signaling ceiling, not measured bandwidth. Likewise, telemetry availability is not a power or thermal measurement.

Authority:

```text
results/exp_073/summary.json
source commit       d3b1d2e4dd08e73781c969814cb4d181377a054d
evidence commit     4e35afb9648dc0c513c3a90c32d604f4c0b0fd21
config SHA-256      d7867c68a135bd69e5cfc8b733b62f7c5c4ea50b9190b231d60c0653f3d4f0d0
core SHA-256        aa9cae0457a6b92fcb75da35fedc1a2a2a9f3da115808d341a5a498ca4722da2
checksums           4 / 4 verified
private scan        PASS
EXP-073 tests        14 / 14 passed
repository tests     344 / 344 passed
validation/current   PASS / PASS (offline fixture only)
workflow/artifact   NOT RUN
```

Stage 1 is complete. Stage 2 remains separately authorized and was not started.
