# VORTEX

VORTEX is a research runtime for executing very large Hugging Face models under a small GPU-memory budget while preserving the behavior of the original model.

## Non-negotiable target

The project target is fixed:

> Run an arbitrary Hugging Face model—including a 405B-class dense model—on an 8GB VRAM machine with no user-side training, distillation, fine-tuning, calibration workflow, or model-specific manual conversion, while preserving the original model's quality and approaching the wall-clock experience of running a native 4B model on the same machine.

The intended user experience is eventually:

```bash
vortex run meta-llama/Llama-3.1-405B-Instruct
```

Everything else—graph inspection, shard discovery, runtime-format generation, tiling, progressive execution, certification, fallback, and caching—must be automatic.

## Repository status

This repository contains an **executable first-stage prototype**, not only a design document.

Implemented and tested:

- Hugging Face `config.json` and safetensors shard/index auto-discovery.
- Individual tensor, tensor-slice, and layer loading without constructing the full model.
- Byte-accurate LRU budgeting for a fixed tensor/VRAM tile window.
- A dependency-light, layer-streamed Llama reference decoder with KV caching.
- Automatic first-use conversion of a linear matrix into a low-bit base plus lossless residual tiles.
- Exact progressive greedy-token certification: unread residual tiles are skipped only when bounds prove they cannot change the dense argmax.
- Disk-backed residual refinement using safetensors slices.
- Exact causal Jacobi block decoding with equality checks against sequential greedy decoding.
- A Llama 3.1 405B memory planner based on its published tensor dimensions.
- Seven automated tests covering cache bounds, streamed generation, progressive bounds, disk refinement, and Jacobi equality.

Current verified checkpoint:

```text
7 tests passed
Progressive LM-head exact certification: 100% in the recorded synthetic runs
Disk-backed progressive LM-head exact certification: 100% in the recorded tiny-checkpoint runs
Jacobi output equality against sequential greedy: 100% in the recorded tiny-checkpoint runs
```

The latest recorded measurements are in [`validation_results.json`](validation_results.json) and are explained in [`VALIDATION.md`](VALIDATION.md).

## What is not completed yet

The runtime does **not yet** achieve the final 405B-on-8GB-at-4B-speed target. The current bottleneck is internal Transformer execution: Q/K/V/O and gate/up/down projections still require exact streamed evaluation. The next milestone is to extend progressive, decision-directed refinement from the LM head into those internal projections and measure how much target weight traffic and compute can actually be skipped.

This distinction must remain explicit in every future session: working primitives are recorded as working; the final target is only declared achieved after the wall-clock acceptance gates in [`docs/VALIDATION_PROTOCOL.md`](docs/VALIDATION_PROTOCOL.md) pass on real hardware.

## Quick start

```bash
git clone https://github.com/yjunhyuk920-ui/Vortex.git
cd Vortex

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -e .
pip install pytest

python -m pytest -q
python scripts/run_validation.py
```

Run the tiny streamed-Llama demo:

```bash
python -m vortex_runtime.cli demo --tokens 8 --budget-mb 2
```

Inspect a local Hugging Face safetensors model:

```bash
python -m vortex_runtime.cli inspect /path/to/model
```

Benchmark in-memory progressive LM-head certification:

```bash
python -m vortex_runtime.cli certify \
  --vocab 4096 \
  --hidden 1024 \
  --trials 32 \
  --base-bits 6
```

## Repository map

```text
vortex_runtime/
  hf_loader.py       HF config/shard discovery and tensor slicing
  tile_cache.py      byte-budgeted LRU tensor cache
  llama.py           streamed Llama reference runtime and Jacobi decoder
  progressive.py     in-memory progressive linear operator and certificates
  vtx_linear.py      disk-backed VTX linear format and refinement
  planner.py         large-model tensor and memory planning
  toy_model.py       deterministic tiny HF checkpoint generator
  cli.py             prototype CLI

scripts/
  run_validation.py  reproducible validation suite and JSON report

tests/               automated unit/integration tests
docs/                project context, architecture, roadmap, protocol, handoff
AGENTS.md             mandatory context for future AI coding sessions
```

## Start here in a new session

Read these files in order:

1. [`AGENTS.md`](AGENTS.md)
2. [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md)
3. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
4. [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md)
5. [`docs/ROADMAP.md`](docs/ROADMAP.md)
6. [`docs/VALIDATION_PROTOCOL.md`](docs/VALIDATION_PROTOCOL.md)

Then run:

```bash
python -m pytest -q
python scripts/run_validation.py
```

Do not replace the fixed target with a smaller model, model retraining, distillation, or a manual per-model preparation workflow. Those are outside the project definition.
