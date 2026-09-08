# Research state — 2026-09-08

Fixed mission and CTC unchanged. [Current frontier](experiments/causal_global_bridge_20260908/REPORT.md).
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false
README_CURRENT=true

HARDWARE_STATUS concerns the target; no latency benchmark in this round. Actual
pinned BF16 HF CPU generation now ran under a guarded supported API.

## Latest constructive frontier — causal/global producer bridge

The frozen round compared three materially different principles: (A) legal
causal dense-information exposure, (B) globally nonlinear checkpoint advice,
and (C) a paid dynamic exact summary. A was constructed first; B/C were then
continued rather than left as named primitives.

Established scoped E1 facts:

```text
HF DynamicCache binary v_proj exposure   square + GQA, exact native BF16 words
GQA source shape                         32x224, 7,168 coordinates, 0 mismatch
restricted basis-column producer         exact logits/K/V/RNG, 0 dense v_proj calls
producer source payload                   128 vs 229,376 bits/token on 2-layer control
small causal independent-right trace      32 queries, 8-bit right factor
exhaustive binary output rows             8,192 checked, 0 parity mismatch
area-5,400 causal source                  25x216, 4 native queries, 0 decode mismatch
area-5,400 32-query native trace          NOT EXECUTED
global nonlinear 8 GiB advice/direct-sum OPEN
arbitrary finite-word native producer     OPEN
```

The restricted producer is a genuine finite encoder/address/decoder, but its
binary source alphabet, basis-token states and zero q/k/MLP/lm-head paths prevent
promotion to O1. The sign/delta causal compiler also closes a hidden assumption
behind dynamic response-column maintenance: legal successive right factors can
differ densely, so literal `W(s'-s)` updates regain the complete dense effect.
Any successful dynamic summary now needs the same missing subdense arbitrary
matrix-vector aggregate as the global static producer.

The previous nonlinear-router theorem remains authoritative E0 evidence:
`25x108`/two probes and all target-feasible side<=128 cases are closed; area
5,400 is the first single-query local survivor; the independently selectable
32-query interface forces 2,048 bits=`64/675`, eight times target. The new
causal compiler narrows its reachability gap but does not yet prove the exact
area-5,400 32-query native/global-advice statement.

O1-O5 remain OPEN. O6 is partial reproducible E0/E1 evidence only.
405B/CUDA/<=8GiB/native4BQ4 p50/p95/TTFT remain NOT TESTED.

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
