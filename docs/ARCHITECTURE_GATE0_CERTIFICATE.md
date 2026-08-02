# Architecture Gate 0 certificate — corrected VORTEX-WAVE-1

Evidence level: **E0 architecture + E1 observed mechanism inputs**

This certificate evaluates one complete candidate against the fixed 405B/8GiB/4B-speed target before native backend work. The machine-readable source of truth is `architecture_gate0_budget.json`.

Generate it with:

```bash
python scripts/run_architecture_gate0.py
```

## Fixed target

Target envelope:

- 405,849,243,648 parameters;
- 126 layers;
- hidden size 16,384;
- intermediate size 53,248;
- 128 attention heads and 8 KV heads;
- 128,256 vocabulary;
- 4,096-token decode context;
- BF16 exact cold weights and KV.

Comparison envelope:

- native 4B-class model;
- Q4 weights;
- BF16 KV;
- same-machine wall-clock measurement required at E3/E4.

The current baseline is analytic. It is a Gate 0 bound, not final performance evidence.

## Candidate summary

`VORTEX-WAVE-1` combines:

1. rank-32 INT8 session capsules for model-wide linear operators;
2. rank-64 old-context attention summaries;
3. a 128-token exact recent KV window;
4. block proposals;
5. selective exact weight/KV repair;
6. exact final token certification and full fallback.

The quality contract remains:

```text
hot path proposes
certificate commits
cold exact state resolves uncertainty
full exact fallback remains authoritative
```

## Corrected resource equations

Let:

- `A` be committed causal-prefix tokens produced from one shared repair set;
- `rho` be the selected exact repair fraction of a full target pass;
- `B_cold` be bytes in one full exact cold pass;
- `C_cold` be arithmetic in one full exact target pass.

Storage traffic can be shared:

```text
B/token = B_hot + rho * B_cold / A
```

Exact arithmetic cannot be shared merely because the weights stay resident:

```text
C/token = C_hot + rho * C_cold
```

The previous equation `C_hot + C_cold / E`, where `E=A/rho`, was incorrect for selected exact weights applied to every token. It is no longer used by the generator or tests.

## Memory envelope

| Component | GiB |
|---|---:|
| Rank-32 INT8 linear capsules | 1.16557 |
| Rank-64 attention summaries | 0.01538 |
| Exact recent KV | 0.06152 |
| Embedding cache | 0.06250 |
| Workspace | 1.25000 |
| Repair window | 0.50000 |
| Allocator reserve | 1.00000 |
| Certificate state | 0.25000 |
| **Total** | **4.30497** |
| **Limit** | **8.00000** |

Analytic memory result: **pass**.

This remains an estimate until measured with a real CUDA allocator.

## Traffic envelope

```text
hot traffic:      1.292485 GiB/token
4B baseline:      2.362645 GiB/token
1.2x limit:       2.835174 GiB/token
full cold repair: 757.921875 GiB
required E:       491.299160
```

For the original design point:

```text
rho = 0.25
A   = 160
E   = 640
```

Projected traffic is:

```text
2.476738 GiB/token
```

Analytic traffic result: **pass**.

## Corrected compute envelope

```text
hot compute:             3.531515 GFLOP/token
4B baseline:            10.110613 GFLOP/token
1.2x limit:             12.132735 GFLOP/token
full exact repair:     845.521355 GFLOP/token
maximum allowed rho:     0.01017268
```

The compute-limited exact fraction is approximately:

```text
1.01727%
```

For the original 25% design point:

```text
C/token = 3.531515 + 0.25 * 845.521355
        = 214.911854 GFLOP/token
```

Analytic compute result: **fail**.

Therefore the original `VORTEX-WAVE-1` design point is rejected before backend implementation.

## Strongest observed E1 result

The exact-target adjoint tile oracle on TinyLlama 1.1B required:

```text
observed E:                  8.195999
required E:                491.299160
observed repair fraction:    0.12201075
maximum compute fraction:    0.01017268
```

Consequences:

- traffic efficiency is short by about 59.94x;
- exact arithmetic is about 11.99x above the compute allowance;
- the per-token rank-32 local-repair family is rejected even under an optimistic oracle that knows exact target tokens and gradients.

## Rejected paths

The following are not active steady-state solutions:

- exact-span Atlas warm decode;
- exact layer-suffix repair;
- output-row tile repair;
- residual-energy-ranked 2D tiles;
- exact-target adjoint 2D tiles applied per token;
- the original 25%-repair `VORTEX-WAVE-1` design point.

## Active candidate

The remaining experiment is block-shared combined traffic/compute repair.

A fixed selected subset may advance only when:

```text
rho <= 0.01017268
A / rho >= 491.299160
exact causal prefix committed > 0
traffic gate = pass
compute gate = pass
```

Passing the oracle is still only E1 because exact target tokens and gradients may select the subset. A deployable mechanism must later provide:

- target-independent selection;
- sound token commit certification;
- disjoint-prompt generalization;
- real operation replacement;
- measured device memory and wall-clock.

## Gate result

| Gate | Result |
|---|---|
| Analytic memory | Pass |
| Analytic traffic at original design point | Pass |
| Analytic compute at original design point | **Fail** |
| Observed repair traffic | **Fail** |
| Observed repair compute | **Fail** |
| Architecture Gate 0 | **Rejected for current design point** |

The project does not proceed to physical NVMe/CUDA streaming for this repair family unless the combined oracle finds a subset inside the corrected compute and traffic envelope.
