# Validation Matrix

Legend: PASS validated in scope; FAIL contradicted; PARTIAL limited; NOT TESTED unavailable; N/A not applicable.

| Claim | Phase A | Phase B | Phase C | Phase D | Evidence | Current verdict |
|---|---|---|---|---|---:|---|
| Final target remains 405B/8 GiB/4B-class | defined | N/A | N/A | NOT TESTED | E0 | objective only |
| Target 8 GiB GPU available now | N/A | N/A | N/A | NOT TESTED | E0 | unavailable |
| Real 405B execution | N/A | N/A | NOT TESTED | NOT TESTED | E0 | NOT TESTED |
| 405B peak VRAM <=8 GiB | formula incomplete | N/A | N/A | NOT TESTED | E0 | NOT TESTED |
| 405B TTFT/tokens per second | formula incomplete | N/A | N/A | NOT TESTED | E0 | NOT TESTED |
| 4B-class user-perceived performance | target defined | N/A | NOT TESTED | NOT TESTED | E0 | NOT TESTED |
| Original 405B quality preserved | contract incomplete | N/A | small models only | NOT TESTED | E0 | NOT TESTED |
| mmap decision VM exact replay | PASS | PASS | bounded TinyLlama PASS | NOT TESTED | E2 | auxiliary |
| Raw exact-prefix graph scales | FAIL | FAIL | FAIL: 64/64, held-out 0% | N/A | E2 negative | REJECTED |
| Exact future-suffix DAG compresses body | PASS | PASS | PARTIAL: 64->38 | N/A | E2 | auxiliary |
| Metadata size equals traffic | FAIL | N/A | N/A | N/A | E1 theorem | false |
| One host probe proves latency failure | FAIL | counterexample PASS | N/A | NOT TESTED | E1 | false |
| EXP-047 Serfling implementation consistent | PASS | PASS: 0/525 mismatches | N/A | N/A | E1 | PASS in scope |
| EXP-047 fallback/reference equality | PASS | PASS | N/A | N/A | E1 | PASS synthetic scope |
| EXP-047 wrong accepts zero | defined | PASS: 0/525 | N/A | NOT TESTED | E1 | corpus only |
| EXP-047 broad savings useful | hypothesis | FAIL: 4/525; 98.294% at N=1024 | N/A | N/A | E1 negative | REJECTED FORM |
| EXP-047 Python overhead below full sum | hypothesis | FAIL: about 8.6–9.1x | N/A | N/A | E1 negative | FAIL implementation |
| EXP-047R exact margin reconstruction | PASS | PASS tests | PASS: 18 states | N/A | E1 | PASS small checkpoints |
| EXP-047R checkpoint bounds sound | PASS | PASS fault tests | PASS: 0 violations | N/A | E1 | PASS soundness only |
| EXP-047R C1 oracle median <=10% | target | N/A | FAIL: 100% | N/A | E1 negative | RANGE CORE REJECTED |
| EXP-047R C1 oracle p90 <=25% | target | N/A | FAIL: 100% | N/A | E1 negative | RANGE CORE REJECTED |
| EXP-047R C2 stratification useful | hypothesis | PASS reference | FAIL: median/p90 100%, best 99.21875% | N/A | E1 negative | FAIL core role |
| Range-based CPTC approaches 1.185% | formula | FAIL synthetic | FAIL oracle 84.375x target | NOT TESTED | E1 negative + PROJECTED | REJECTED CORE |
| CPTC certificate/fallback auxiliary correctness | PASS | PASS | PASS small audit | NOT TESTED | E1 | AUXILIARY |
| EXP-048 longest-prefix correction exact | PASS | PASS: 9 tests | PASS: B1/B2/B3 mismatch 0 | N/A | E1 | AUXILIARY PASS |
| EXP-048 predictions after first mismatch ignored | PASS | PASS | PASS committed cases | N/A | E1 | PASS verifier contract |
| EXP-048 B1 96-token one-pass verification | defined | PASS | PASS: 18/18, 1.041667% | N/A | E1 oracle | NON-DEPLOYABLE UPPER BOUND |
| EXP-048 B1 causal/deployable | FAIL by definition | N/A | future info true | N/A | E1 | NOT DEPLOYABLE |
| EXP-048 hard Jacobi cheaper than sequential | hypothesis | exact control PASS | FAIL: p50 181.25%, p90 193.75% | N/A | E1 negative | REJECTED CORE |
| EXP-048 B3 no future tokens | defined | instrumentation PASS | PASS: 0 uses | N/A | E1 | PASS causality |
| EXP-048 B3 exact committed output | defined | tests PASS | PASS: mismatch 0 | N/A | E1 | PASS verifier scope |
| EXP-048 B3 p50 committed >=16 | threshold | N/A | FAIL: 1 | N/A | E1 negative | SELF-DRAFT CORE REJECTED |
| EXP-048 B3 p90 fraction <=10% | threshold | N/A | FAIL: 2893.843% | N/A | E1 negative | FAIL |
| EXP-048 B3 useful proposal prefix | hypothesis | N/A | FAIL: max 1, 4/18 nonzero | N/A | E1 negative | FAIL |
| EXP-049 Picard/Anderson implementation fail-closed | PASS | PASS: 9 tests | PASS workflow | N/A | E1 | AUXILIARY PASS |
| EXP-049 selected exact verifier mismatch zero | defined | PASS tests | PASS: 0/18 | N/A | E1 | PASS selected scope |
| EXP-049 S1/S2 no target future info | defined | instrumentation PASS | PASS: 0 uses | N/A | E1 | PASS causality |
| EXP-049 unhandled numerical failure zero | defined | fault tests PASS | PASS: 0 | N/A | E1 | PASS |
| EXP-049 S3 exact future-state alignment | control | PASS | PASS: 0 failures | N/A | E1 oracle | NON-DEPLOYABLE CONTROL |
| EXP-049 favorable p50 prefix >=16 | threshold | N/A | FAIL: 4.5 | N/A | E1 negative | FIXED-POINT CORE REJECTED |
| EXP-049 favorable maximum prefix useful | hypothesis | N/A | FAIL: maximum 6 | N/A | E1 negative | FAIL |
| EXP-049 favorable p90 fraction <=10% | threshold | N/A | FAIL: 168.778596% | N/A | E1 negative | FAIL |
| EXP-049 Anderson improves >=4x over Jacobi | threshold | positive control only | FAIL: 0.25x | N/A | E1 negative | FAIL |
| EXP-049 target-size trend non-degrading | threshold | N/A | PASS: medians 4.5/5.0/4.0 | N/A | E1 | PASS but insufficient |
| Universal >1 exact position/round target-only guarantee | theorem target | adversarial PASS | FAIL guarantee: hidden chain barrier true | N/A | E1 negative | REJECTED IN DECLARED INTERFACE |
| Hidden suffix transcript indistinguishability | defined | PASS tests | PASS adversarial audit | N/A | E1 | LOWER-BOUND CONSTRUCTION PASS |
| Target-only continuous fixed-point reaches 1.185% | formula | positive solver controls | FAIL: favorable p90 ~142.4x target | NOT TESTED | E1 negative + PROJECTED | REJECTED CORE |
| EXP-050 external first-token universal guarantee | counterexample defined | NOT TESTED | NOT TESTED | N/A | E0 | next Gate E0 |
| EXP-050 fixed external pool p50 exact prefix >=16 | threshold | NOT TESTED | NOT TESTED | N/A | E0 | next Gate E2 |
| EXP-050 fixed pool p90 normalized fraction <=10% | threshold | NOT TESTED | NOT TESTED | N/A | E0 | next Gate E2 |
| 4B external draft final p50 exact prefix >=507 | derived requirement | N/A | NOT TESTED | NOT TESTED | E0 + PROJECTED | promotion requirement |
| External draft selector causal and target-independent | defined | NOT TESTED | NOT TESTED | N/A | E0 | unresolved |
| Savings persist with model size | formula pending | small models negative so far | NOT TESTED | NOT TESTED | E0 | A-006 unverified |
| Target RAM/SSD bandwidth sufficient | formula pending | small files only | small model only | NOT TESTED | E0 | NOT TESTED |

## Current overall classification

```text
Governance/provenance: implemented
EXP-047/047R certificate correctness: E1 PASS in scope
Range-based CPTC savings: FAIL; core rejected
EXP-048 exact block verifier: E1 auxiliary PASS
EXP-048 hard Jacobi and partial-layer self-draft: FAIL core
EXP-049 Picard/Anderson reference and fault handling: E1 auxiliary PASS
EXP-049 target-only continuous fixed-point proposal: FAIL core
EXP-049 hidden triangular target-round barrier: E1 adversarial PASS
EXP-050 external draft advice Gate: pre-registered, NOT TESTED
Real 405B/8 GiB/CUDA/PCIe/SSD/TTFT/tokens/sec: NOT TESTED
E6/E7: not achieved
```

Every experiment PR must update at least one row using commit-backed evidence. PROJECTED or UNVERIFIED fields may not be upgraded to MEASURED without the matching phase.
