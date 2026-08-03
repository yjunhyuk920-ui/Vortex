# VORTEX Decision Log

Append-only decisions. Authoritative run identities are read from committed result JSON.

## D-001 — Final target fixed

Arbitrary public unmodified Hugging Face dense model; runtime only; real 405B; total GPU VRAM <=8 GiB; original contract preserved; 4B-class user experience; independent reproduction.

Status: ACTIVE.

## D-002 — Current environment is not Phase D

GitHub Actions CPU cannot be labeled target GPU, 405B, CUDA, PCIe, SSD, TTFT, tokens/second, power, physical block reuse, or peak-VRAM evidence.

Status: ACTIVE. Phase D NOT TESTED.

## D-003 — Phase A/B/C/D adopted

Status: ACTIVE.

## D-004 — E0–E7 adopted

Status: ACTIVE.

## D-005 — MEASURED/DERIVED/PROJECTED/UNVERIFIED separation adopted

Status: ACTIVE.

## D-006 — mmap/index/DAG components are auxiliary

Evidence: PR #50/#52/#54.

Status: ACTIVE.

## D-007 — Raw prefix enumeration rejected as core

Evidence: 64/64 unique nodes excluding duplicate; held-out start coverage 0%.

Status: REJECTED.

## D-008 — Exact future-DAG accepted only as body compression

Evidence: 64->38 nodes; causal held-out start coverage 0%.

Status: AUXILIARY.

## D-009 — Core research must skip or amortize original operations causally on unseen prompts

Status: ACTIVE.

## D-010 — EXP-047 correctness primitive accepted at E1

Authority `results/exp_047/summary.json`, workflow `30793232558`.

Status: AUXILIARY.

## D-011 — Global-range and range-rescue CPTC rejected as core

EXP-047 certified 4/525 with 99.238% fallback. EXP-047R exact realized range oracle median/p90 was 100%.

Decision:

```text
REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY
```

Status: REJECTED CORE.

## D-012 — Completed evidence is immutable

Completed workflows write isolated reproduction output. Frozen result directories and raw checksums are authoritative.

Status: ACTIVE.

## D-013 — Exact longest-prefix block verifier accepted at E1

EXP-048 verified proposal blocks left-to-right, committed matching prefix plus exact first-mismatch correction, and never committed later predictions.

Status: AUXILIARY.

## D-014 — Perfect proposal proves verifier arithmetic only

96 exact future tokens / one target pass =1.0416667%, but future information true.

Status: NON-DEPLOYABLE UPPER BOUND.

## D-015 — Hard Jacobi and same-checkpoint partial-layer self-draft rejected

EXP-048 hard Jacobi p50 181.25%; partial-layer draft p50 committed 1 and p90 2893.843%.

Decision:

```text
REJECT_PARTIAL_LAYER_SELF_DRAFT_CORE_RETAIN_EXACT_BLOCK_VERIFIER
```

Status: REJECTED CORE.

## D-016 — Target-only continuous fixed-point generation rejected

Authority `results/exp_049/summary.json`, workflow `30803672059`, artifact `8851957250`.

Favorable reference-selected result: p50 prefix 4.5, maximum 6, p90 fraction 168.778596%, Anderson/Jacobi 0.25x. Hidden triangular targets preserved transcript indistinguishability and the one-new-position-per-round barrier.

Decision:

```text
REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
```

Status: REJECTED CORE; solver/verifier references retained.

## D-017 — EXP-050 target-independent external draft Gate executed

Authority:

```text
results/exp_050/summary.json
workflow 30806015309
source head SHA 1388c780abea11067c66cd666ed0a313ec2f682c
workflow merge SHA 6bdd0a20334e394ec5252a6c0e676c1f62b608d0
artifact 8852817664
artifact ZIP SHA-256 a32ffe8dbfc201c6d70ca8dac660164d8400691ad4d8fe3593d688e7754f6159
```

MEASURED correctness/causality:

```text
9 EXP-050 tests passed
repository validation passed
18 target/prompt cases
36 target/draft/prompt pairs
108 K rows
exact committed-output mismatches 0
target-future uses 0
E3 future-oracle failures 0
```

Status: EXECUTED, E1.

## D-018 — Fixed target-independent draft universal guarantee rejected

A deterministic draft proposed first token 7. An arbitrary causal target chose token 8 for the same prompt. Exact verification reported matching proposal prefix zero and committed only exact correction token 8.

Therefore a fixed target-independent draft cannot guarantee even one matching proposal token for every arbitrary target.

Status: UNIVERSAL FIRST-TOKEN COUNTEREXAMPLE ACCEPTED WITHIN DECLARED INTERFACE.

## D-019 — Tested TinyStories external draft pool rejected as practical core

The exact target reference selected the best eligible external draft and K per target/prompt.

MEASURED:

```text
p50 exact proposal prefix 0.5
maximum prefix 3
p90 normalized fraction 163.20987654%
matching prefix zero 72/108 rows
all selected K=64
Korean useful acceptance false
structured JSON useful acceptance false
target median prefixes 1.0 / 0.0 / 0.5
```

Gate failures: prefix, traffic, family coverage, target-size trend, and universal counterexample. Exactness/causality passed.

Decision:

```text
REJECT_TARGET_INDEPENDENT_EXTERNAL_DRAFT_AS_UNIVERSAL_CORE
```

The tested pool is also rejected as a restricted practical core. Proposal-tree continuation is prohibited.

Status: REJECTED CORE.

## D-020 — Actual 4B draft requires 507 exact proposal tokens before overhead

PROJECTED:

```text
4B/405B draft ratio = 0.0098765432
required total fraction = 0.01185185185
4/405 + 1/K <= required
K >=507
```

The older 85-token requirement applies only to a zero-cost proposal.

Status: ACTIVE RESOURCE EQUATION.

## D-021 — Next Gate changes skip axis from future tokens to Transformer depth

EXP-051 will use exact greedy prefixes and audit every intermediate target block depth for the current next-token decision.

Primary oracle: earliest suffix-stable depth whose intermediate final-norm/LM-head token equals the exact final token at every later depth.

A late-decision adversarial residual chain tests the universal fixed-depth boundary. No selector/certificate is built unless the non-deployable suffix-stable oracle survives lenient 10%/25% traffic Gates.

Status: ACTIVE NEXT GATE — EXP-051.
