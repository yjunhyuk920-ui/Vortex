# Architecture Gate 0 certificate — corrected VORTEX-WAVE family

Evidence level: **E0 architecture + E1 observed mechanism inputs**

This certificate evaluates the fixed 405B/8GiB/4B-speed target before native backend work. The machine-readable source of truth is `architecture_gate0_budget.json`.

```bash
python scripts/run_architecture_gate0.py
```

## Fixed target

- unmodified 405B-class dense Hugging Face model;
- peak GPU VRAM at or below 8 GiB;
- no user training, distillation, fine-tuning, or manual model-specific conversion;
- declared original-model quality preserved;
- p50 warm-decode time/token at or below 1.2x a native 4B Q4 baseline on the same machine.

The analytic baseline is only a Gate 0 bound. Same-machine hardware measurement is mandatory at E3/E4.

## Corrected equations

Let:

- `A` be committed causal-prefix tokens from one shared repair set;
- `rho` be the selected exact repair fraction;
- `B_cold` be bytes in one full exact cold pass;
- `C_cold` be arithmetic in one full exact target pass.

```text
B/token = B_hot + rho * B_cold / A
C/token = C_hot + rho * C_cold
```

Storage traffic can be shared across the block. Exact arithmetic with selected weights is performed for every token and is not divided by `A`.

## Fixed analytic envelope

```text
memory estimate:                 4.304970 GiB
memory limit:                    8.000000 GiB
hot traffic:                     1.292485 GiB/token
traffic limit:                   2.835174 GiB/token
hot compute:                     3.531515 GFLOP/token
compute limit:                  12.132735 GFLOP/token
full exact repair traffic:     757.921875 GiB
full exact repair compute:     845.521355 GFLOP/token
required traffic efficiency E: 491.299160
maximum exact repair fraction:   1.017268%
```

## Original VORTEX-WAVE-1 point — rejected

The original point assumed:

```text
rho = 25%
A   = 160
E   = 640
```

Its traffic projection fits, but corrected compute is:

```text
214.911854 GFLOP/token
```

Therefore the original 25%-repair design point is rejected.

## Per-token repair family — rejected

The strongest per-token exact-target adjoint oracle on TinyLlama 1.1B reached:

```text
E:                         8.195999
repair fraction:          12.201075%
traffic shortfall:        59.94x
compute excess:           11.99x
```

This rejects exact-span, layer-suffix, row-tile, residual-energy tile, and per-token adjoint repair as steady-state paths.

## Block-shared logical oracle — survives E1 budget

Workflow evidence:

```text
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
managed operations: all O/down projections
build prompts: four mixed English/Korean task prompts
evaluation: disjoint Korean prompt
proposed continuation: 64 tokens
selector: exact target tokens plus teacher-forced gradients
```

Observed zero-repair prefix:

```text
1 exact token
```

Best repaired prefix inside the corrected combined envelope:

```text
selected tiles:                    128
selected exact bytes:            8 MiB
rho:                           0.190642%
committed prefix A:                 2 tokens
incremental prefix:                 1 token
E = A/rho:                    1049.087891
projected traffic:              2.014943 GiB/token
projected compute:              5.143432 GFLOP/token
```

Both analytic traffic and compute limits pass at this observed point.

This does **not** establish feasibility. It proves only that the rank-32 block-shared family is not eliminated by the logical byte/compute budget when an oracle chooses the repair set.

Source:

- `results/tinyllama_1_1b_block_shared_combined_gate.json`
- workflow run `30738817896`
- artifact digest `sha256:26eb56a58911aec98e714d0433e5b078343acf54df2cd35b9f30fa33891e2832`

## Why this remains E1

The passing selector used information unavailable to a real runtime:

- exact future target tokens;
- teacher-forced gradients;
- exact target logit margins.

Additional limitations:

- only O/down projections were replaced;
- only one evaluation prompt was measured;
- the repair improved the prefix by one token;
- no sound certificate can decide online that the prefix is safe;
- physical storage traffic, peak VRAM, and wall-clock were not measured.

## Active Gate 0 continuation

`scripts/run_block_shared_residual_selector.py` removes exact target and gradient information from selection.

Allowed selector inputs:

- approximate autoregressive activation residuals;
- precomputed weight-tile Frobenius norms.

Exact output is used only afterward to evaluate the resulting prefix.

The selector advances only when:

```text
incremental exact prefix > 0
rho <= 0.01017268
A / rho >= 491.299160
traffic pass = true
compute pass = true
```

After that, the next mandatory gate is a sound causal-prefix certificate that does not know the exact continuation.

## Current Gate result

| Gate | Result |
|---|---|
| Analytic memory | Pass |
| Original 25% traffic point | Pass |
| Original 25% compute point | **Fail** |
| Per-token repair family | **Fail** |
| Exact-target block-shared logical oracle | **Pass at E1** |
| Target-independent selector | Pending experiment |
| Sound commit certificate | Not implemented |
| Model-wide real-operation replacement | Not implemented |
| Architecture Gate 0 overall | **Blocked, not rejected** |

No CUDA/NVMe production backend is justified until target-independent selection and certification preserve the combined envelope.
