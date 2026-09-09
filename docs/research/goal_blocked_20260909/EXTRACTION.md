# VORTEX resume action extract

Evidence read: `vortex-independent-audit/AGENTS.md`, `NEXT_EXPERIMENT.md`, and the three requested handoffs. All requested files are present. No shell/Python/test execution command is specified in these files; the actionable item is a construction contract and prerequisite reading/workflow.

## Explicit still-requested construction

The root next-action block requires constructing the following finite, paid, causal execution object (`NEXT_EXPERIMENT.md:87-95`):

```text
G = Compile(checkpoint_all)                          finite, fully paid
addresses = Address(G, current_input, exact_state)  finite causal algorithm
words = Read(G, addresses)                          paid physical payload
(effect,state') = Decode(words,input,state)         native finite-word exact
```

The same requirement says it must support arbitrary unchanged checkpoint matrices, globally mixed advice without a per-matrix direct-sum assumption, exact ordered Q4/BF16/FP32 dense effect and required successor state, while meeting the 8-GiB and native-4B-Q4 traffic/arithmetic line (`NEXT_EXPERIMENT.md:97-100`). The candidate must directly construct a nonlinear implicit GF(2) or native finite-word representation/address/decoder, with source-dependent program bits not fetched at roughly one bit per original weight and without a complete stored image atom per query component (`NEXT_EXPERIMENT.md:110-118`).

The alternative still-requested construction is an exact dynamic nonlinear state whose update is subdense and explicitly paid even when legal right factors change in every coordinate (`NEXT_EXPERIMENT.md:120-133`; implicit-carrier handoff `:58-65`). A valid alternative must aggregate those dense changes with a finite native-exact update rule; literal `y'=y+W(s'-s)` is closed.

## Stated prerequisites and fixed constraints

Before a core experiment, specify a finite checkpoint compiler, causal addresses, physically read words, native ordered decoder, and successor-state relation (`NEXT_EXPERIMENT.md:34-40`). Preserve arbitrary unmodified HF dense 405B scope, batch 1, total peak GPU VRAM <=8 GiB, exact output/logits/RNG/next state/KV for all legal continuations, and all CPU/RAM/SSD/PCIe/HBM/metadata/compile/decode/verification/recovery/fallback costs (`AGENTS.md:5-10,44-50`; `NEXT_EXPERIMENT.md:97-100`). No arbitrary per-matrix division of global advice (`NEXT_EXPERIMENT.md:97-100,141-146`). Privacy/permissions/evaluation independence remain unchanged; no private host or credential discovery is requested. No 405B hardware is available (`NEXT_EXPERIMENT.md:41-42`; native-residual handoff `:13`; FMM handoff `:31-32`).

## Explicit do-not-rerun / exclusions

Do not rerun or grow the residual two-query lemma (`codex_native_residual_lift_20260909/HANDOFF.md:3-7`), the FMM catalog/integrity gate (`codex_fmm_integrity_20260909/HANDOFF.md:26-36`), the implicit-carrier static routes as a new general claim (`implicit_program_carrier_20260909/HANDOFF.md:21-43`), the sparse fixture/scalar sweep/trivial third-coordinate variation (`NEXT_EXPERIMENT.md:26-31`), the native-fiber collision box (`NEXT_EXPERIMENT.md:44-53`), or gate-only butterfly/replay evidence (`NEXT_EXPERIMENT.md:65-70`). Existing exclusions include source reconstruction, free rounding witness/decoder, response catalogs, full bitset scans, quadratic row-operation replay, literal all-coordinate dense-delta updates, and hidden/unpaid decoder work (`NEXT_EXPERIMENT.md:100-108,120-124,141-146`; residual handoff `:9-13`).

No theorem, feasibility, goal-attainment, hardware, or runtime-model conclusion is made here. Requested model metadata is not exposed by these files.
