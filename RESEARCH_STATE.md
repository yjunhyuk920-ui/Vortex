# Research state — 2026-09-09

## Restricted BF16 producer extension — 2026-09-09

Current executed extension: [experiments/codex_native_sparse_extension_20260909/REPORT.md](experiments/codex_native_sparse_extension_20260909/REPORT.md).
The finite BF16 / <=2-support scalar producer has a zero-sign proof, a paid
compiler/address/read/decode implementation, 1,566,724 frozen scalar controls
and a separate primary-run 124,423-case integer oracle. The final actual CPU HF
run in hf_results_v5 preserves the frozen 40-step trace and eight generation
samples, including logits, serialized cache fields/strides and CPU RNG states.
Canonical summary SHA-256:
7853685458d0b12de60cbdce613ab556e2444a0dc1d0e47fc094d2c781c2a5da.

Scalar v1 defects were found by static review; old source bytes/red execution
were not retained. HF v2 records the actual first-stride failure; v5 passes the
same fixed fixture after a native view/transpose repair. No general dense
producer, universal HF induction or closed physical 405B cost follows.
O1-O5 OPEN/O6 PARTIAL; CORE_ADMISSION=false; theory NOT_ESTABLISHED;
hardware NOT_TESTED. This is actual research progress, not goal completion.

## Native fiber obstruction — 2026-09-09

[Native proof and controls](experiments/codex_native_fiber_20260909/REPORT.md) show that algebraic full rank does
not imply native injectivity. A dense BF16 matrix with diagonal 2/off-diagonal 1
sends at least `256^(n-1)` distinct inputs to one FP32 output under the declared
balanced RNE ABI, for `2<=n<=16384`. Bijective input/output encodings preserve
fiber sizes, so this native map cannot become an injective full-coordinate copy.
A later proof-only binary full-rank corollary preserves the source alphabet.

This closes only a local O3 premise, not the general encoded-graph route.
Cross-wire/side-state encodings and nonbijective continuation representations
remain open. The `8(n-1)`-bit term is only for an optional reversible lift, not
a native KV-memory bound or a summable per-matrix resource lower bound.
No arbitrary dense-work reduction or full HF/native-state proof is supplied.
O1-O5 OPEN, O6 PARTIAL; THEORY_STATUS=NOT_ESTABLISHED,
HARDWARE_STATUS=NOT_TESTED, CORE_ADMISSION=false. Target hardware remains
unavailable, and the persistent goal remains active and unachieved.

## Independent native-gauge construction — 2026-09-09

[Scoped report](experiments/codex_gauge_transport_20260909/REPORT.md): an actual layout compiler now transports the
original product/reduction schedule through column permutations. A balanced
FP32 witness returns 0 originally and 1 under naive permutation; transported
schedule returns 0. An explicit opaque-word butterfly decoder also computes
the original native Hadamard gate in encoded state with `3*n*log2(n)/2` XORs
plus n native products. This limits an overbroad non-permutation-gauge no-go;
it **does not remove arbitrary dense projection work**. All original dense
coefficients/operations remain in the literal projection route.

O1-O5 remain OPEN, O6 PARTIAL; THEORY_STATUS=NOT_ESTABLISHED,
HARDWARE_STATUS=NOT_TESTED, CORE_ADMISSION=false. The user confirms that current
405B-capable hardware is unavailable. The actual Codex persistent goal remains
unachieved. This is a bounded independent audit of concurrent preregistration
`dab5774`, not a completed new three-principle core round. Its opaque-bit
butterfly also establishes that exact transformed native Hadamard need not be
quadratic; the arbitrary dense transformed projection remains missing.

Fixed mission and CTC unchanged. [Current frontier](experiments/implicit_program_carrier_20260909/REPORT.md).
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O5=OPEN
O6=PARTIAL
THREE_QUALIFYING_NEW_PRINCIPLES=false
README_CURRENT=true

HARDWARE_STATUS concerns the target; no target latency benchmark ran in this
round. The actual pinned BF16 HF CPU generation evidence belongs to the prior
causal/global round and remains preserved below.

## Latest constructive frontier — implicit checkpoint-program carriers

Three newly preregistered carriers were actually compiled and executed:

```text
P1 address-only alias router             EXACT GF2 / EXPLICIT CARRIER REJECTED
  square arithmetic fraction             0.053116608411073685
  registered alias count                 40,365,964,800
  registered 64-bit descriptors          300.74987411499023 GiB
  descriptor/source ratio                6.398601117664475x

P2 query-time Patricia synthesizer       EXACT GF2 / WORD-LABEL CARRIER REJECTED
  square favorable event fraction        0.023436546325683594
  frozen edge-label/source ratio         1.9998779296875x
  topology/membership in that ratio      not included

P3 rank-normal encoded state             EXACT GF2 / LITERAL NONLINEARITY REJECTED
  arbitrary rectangular A W B = J_r      constructed
  isolated full-rank linear fraction     1/n
  two-projection+AND micrograph fraction 1.0000305171124708x
```

Canonical result:

```text
results/e0_implicit_program_carrier_gate/summary.json
SHA-256 f6f911fce9383ab6b82a1ac4ad5ef79f354a8ca58d79d21ff71193000c8d5070
focused validation 11/11 PASS
```

The result says that three explicit places for the missing program fail their
own paid realization: routing metadata, query-time edge labels, or a conjugated
nonlinearity. It does **not** reject every nonlinear address encoding, every
succinct program synthesizer, or every globally co-designed encoded graph.

O1-O5 remain OPEN. O6 is partial E0 reproducibility only.
405B/CUDA/<=8GiB/native4BQ4 p50/p95/TTFT remain NOT TESTED.

## Prior constructive frontier — implicit nonlinear direct-query representation

The three preregistered implicit principles were executed from the clean
direct-global scientific head. Current canonical result:

```text
results/e0_implicit_direct_query_gate_v3/summary.json
SHA-256 78644fd5ef4b3e0131bf78867e8986718ed02faeaf37b4228fd1c37909a7af29
focused current-source validation 15/15 PASS
```

```text
P1 query-side complete image frame       CONSTRUCTED / TARGET CORE REJECTED
  arbitrary GF2 exact producer            yes
  checkpoint-dependent runtime program    none; query block bits are addresses
  registered query/source fraction        0.09997814246350742
  registered transformed storage          4,817.8189106583595 GiB
  registered query payload                4.699216783046722 GiB
  registered 8/675 multiple               8.435655770358439x
  one-sided linear frame target ratio     ~1e23x storage lower at registered widths

P2 exact value/state coalescing          CONSTRUCTED / REJECTED
  declared ordered finite-word ABI        exact
  static best operation fraction          16385/32768 = 50.0030518%
  high-distinctness fraction              1.0
  dynamic state/weight adversary          1.0 update fraction

P3 encoded-state safe gauge              REJECTED IN DECLARED SAFE CLASS
  arbitrary meet-preserving bijection     coordinate permutation only
  independent linear product isotopy      one aligned coordinate permutation only
  arbitrary dense support after gauge     unchanged
```

P1 is a meaningful constructive advance because it is a finite arbitrary-GF2
direct-query compiler/address/decoder with no checkpoint-dependent runtime
instruction stream. It nevertheless misses storage and target query traffic by
large factors and has no arbitrary-native lift. The one-sided frame lower bound
does not close nonlinear cells. P2 closes only duplicate value/transition
sharing. P3 closes only gauges that keep coordinatewise product cheap; a richer
transformed nonlinear operator remains open if explicitly constructed and paid.

O1-O5 remain OPEN. O6 is partial E0 reproducibility only.
405B/CUDA/<=8GiB/native4BQ4 p50/p95/TTFT remain NOT TESTED.

## Prior constructive frontier — direct global finite-word producer

Three new direct-producer principles were preregistered and executed from the
clean remote causal/Boolean frontier:

```text
P1 global exact reconstruction code     REJECTED
  binary max source-read removal          17.020392%
  Q4 max source-read removal               4.255098%
  BF16 max source-read removal             1.063775%
  native dense arithmetic retained       100%

P2 exact sum + rounding witness          REDUCES BACK TO DIRECT MATVEC
  8-leaf balanced FP32 exact sum           0 for b=0 and b=1
  rounded result                            -b
  aligned row/query gadget                 -popcount(row & query)

P3 global nonlinear adaptive router      GENERAL CLASS OPEN
  globally mixed nonlinear cells           allowed
  per-matrix advice split                   not used
  registered single-tuple floor            312,468 x 64-bit words
  registered route-cover floor             578,619 x 64-bit words
  floor / favorable target                 0.193471%
  target impossibility                     NOT PROVED

explicit arbitrary-GF2 producer          CONSTRUCTED, THEN REJECTED AS CORE
  algorithm                               Gauss-Jordan + reverse row-XOR replay
  width-16,384 operation fraction         16385/32768 = 50.0030518%
  row-program metadata lower              447.97 MiB for one matrix
```

The new route theorem is stronger in scope than the prior local nonlinear Gate:
cells may mix all 883 matrices and advice is never divided per matrix. It is
weak in magnitude, about 516.87x below the favorable target word allowance, so
it cannot be converted into a mission impossibility claim.

The Gauss-Jordan result is an actual finite encoder/representation/runtime/
decoder for arbitrary binary matrices, but it fails the >=90% operation entry
Gate and has adverse metadata. No arbitrary-native Q4/BF16/FP32 producer was
constructed.

O1-O5 remain OPEN. O6 is partial E0/E1 reproducibility only.
405B/CUDA/<=8GiB/native4BQ4 p50/p95/TTFT remain NOT TESTED.

## Earlier constructive frontier — causal/global producer bridge

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
Boolean Mv black-box -> F2 exact lift      REJECTED: >= |supp(v)| calls
one-shot nonlinear Boolean feature lift   REJECTED: exact rank 2^d-1
direct GF2/native global data structure    OPEN
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

Principle B was then narrowed further without changing the mission. A
deterministic exact lift that treats a fast Boolean-semiring MatVec structure as
a black-box complete-product oracle needs at least one Boolean product per
target-support coordinate on an exact deletion adversary. Hiding parity in one
transformed Boolean product does not rescue the route: arbitrary nonlinear
row/query feature maps have exact Boolean rank `2^d-1` for the complete GF(2)
inner-product function. At `d=216` this is exponentially beyond the source.
These are scoped algebraic rejections only; direct GF(2), globally nonlinear
advice and native finite-word data structures remain the active construction
frontier.

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
