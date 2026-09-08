# Research state — 2026-09-08

Fixed mission and CTC unchanged. [Current frontier theorem](experiments/nonlinear_router_frontier_20260908/REPORT.md).
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false
README_CURRENT=true

HARDWARE_STATUS concerns the target; no latency benchmark in this round. Actual
pinned BF16 HF CPU generation now ran under a guarded supported API.

## Latest constructive frontier — nonlinear adaptive word routing

Three materially different information-flow principles were compared before
selection: globally nonlinear checkpoint words, 32-query factor-envelope
restriction generation, and exact native accumulator transition-map
composition. The latter two still hide their generator if promoted. The first
had a concrete old capacity target and was attacked directly.

A new exact largest-fiber/Segre cover Gate applies to **arbitrary nonlinear
stored cells, arbitrary value-adaptive addresses at every probe depth, and an
arbitrary deterministic final decoder**. Any depth-`t`, `S`-cell, `w`-bit exact
linear-query router implies a cover by at most `S^t` coefficient subspaces of
dimension at most `t*w`.

Consequences now registered:

```text
25x108 / 50 words / 2 probes       REJECTED, coverage upper 0.000074505808...
all side<=128 target 64-bit cases  8,256 checked, 0 unclosed
first local unclosed area          5,400 bits
first local shapes                 24x225, 25x216
storage / target probes            99 padded words / 4 probes
nonadaptive 4-probe                REJECTED on both shapes
one-value-stage adaptive 4-probe   REJECTED on both shapes
multi-stage / strong32 interface   REJECTED: >=32 distinct words, 64/675 = 8x target
causal reachability of hard tuple  OPEN
```

The 5,400-bit single-query points are **not constructions**. A stronger
independently-selectable 32-query interface forces at least 32 distinct words
on some tuple: 2,048 bits, `64/675`, exactly 8x the registered `8/675` line.
This does not prove that one legal batch-1 Transformer continuation realizes
that hard tuple. Global cross-matrix encoding and native numerical lifting also
remain outside this Gate. O1-O5 remain OPEN; O6 is partial E0 evidence only.
405B/CUDA/<=8GiB/native4BQ4/TTFT remain NOT TESTED.

## Prior bounded native execution evidence (unchanged)

Current: native graph constructor/JSON loader A, full byte/layout cache codec B,
and TOP/singleton demand executor C. Each candidate's12generation forward steps
preserve589824logit and760320KV coordinates plus layouts/cache fields/RNG/own
sampled sequences. Separate fullprefill A preserves196608logits andall60KV roots.
Original parameter tensorhash96c04987c93051ba2608b2e0905d67e36d299873bcf606ea2787cc01a77a93db
is unchanged. Explicit all-one mask plus original HF auto-pad value0 reproduces
the prior automatic-mask reference. Initial tracing error/source retained.

A/C matrixMAC ratios=1.0 at every captured signature; graph calls fall only by30
dead auxiliary nodes, not CSE of dominant projections. A captures5originalforwards,
C4;9programs total11775553B plus the original269060552B file. B executes12full
forwards plus5114880B minimum codec RW. Other state/memory/dispatch costs extra.
14tests and3510sciencefiles regenerate; timing fields excluded, original values
retained. No qualifying10x route and no 405B/8GiB/native4BQ4/TTFT evidence.

## Prior packet screen (unchanged; not the current HF execution claim)

Constructed native row-envelope query: min/max per coordinate, sign-aware RNE
endpoints, only nonzero finite BF16 singletons broadcast; otherwise original rows
read once. Full output, no input bank/future output/previous-input similarity.

Pinned SmolLM2-135M93efa2f097d58c2a74874c7e644dbc9b0cee75a2,12unaltered matrices,
96syntheticvectors,101376BF16coordinates:0mismatch;36Fractiondots. 0of432packets
have a common actual word. Cold reads100%, interval+direct FPwork/coefficient
payload100.78%-101.04%, other costs extra. Bound tightening cannot repair these
fixed broadcast packets; no all-algorithm or reachable-activation lower bound.

Real weight provenance established, but HF activations/forward/ABI and fullKV/RNG
not established. Originalpayload17,915,904B,summary158,976B; originals retained.
Planted256x64control is separate.12tests/38numerical replays with frozen hashes.
Need a cheap exact heterogeneous effect source. Other-session correlation_source
and local-only mantissa_source are not silently imported. Correlation_source is now
committed at PR147/378fcd584330d17f9d144f64ccbc01721903ea6a on another branch; see
[frontier reconciliation](experiments/output_envelope_20260908/FRONTIER_SYNC.md).
[Prior state](docs/research/history/pre_output_envelope_20260908/RESEARCH_STATE.md).
[Pre-native-global state](docs/research/history/pre_native_global_transition_20260908/RESEARCH_STATE.md).
