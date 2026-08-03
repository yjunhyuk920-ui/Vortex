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
| Causal unseen-prompt start routing | open | unsolved | prior held-out 0% | NOT TESTED | E2 negative | unresolved |
| Metadata size equals traffic | FAIL | N/A | N/A | N/A | E1 theorem | false |
| One host probe proves latency failure | FAIL | counterexample PASS | N/A | NOT TESTED | E1 | false |
| EXP-047 Serfling implementation consistent | PASS | PASS: 0/525 mismatches | N/A | N/A | E1 | PASS in scope |
| EXP-047 fallback/reference equality | PASS | PASS | N/A | N/A | E1 | PASS synthetic scope |
| EXP-047 future generated tokens absent | PASS | PASS | N/A | NOT TESTED | E1 | PASS primitive |
| EXP-047 wrong accepts zero | defined | PASS: 0/525 | N/A | NOT TESTED | E1 | corpus only |
| EXP-047 adversarial fallback | defined | PASS: 15/15 | N/A | N/A | E1 | PASS |
| EXP-047 broad savings useful | hypothesis | FAIL: 4/525; 98.294% at N=1024 | N/A | N/A | E1 negative | REJECTED FORM |
| EXP-047 Python overhead below full sum | hypothesis | FAIL: about 8.6–9.1x | N/A | N/A | E1 negative | FAIL implementation |
| EXP-047R exact LM-head margin reconstruction | PASS | PASS tests | PASS: 18 states | N/A | E1 | PASS small checkpoints |
| EXP-047R checkpoint-derived bounds sound | PASS proof | PASS fault tests | PASS: 0 violations | N/A | E1 | PASS soundness only |
| EXP-047R wrong accepts zero | defined | PASS tests | PASS: 0/18 | N/A | E1 | corpus only |
| EXP-047R future generated tokens absent | PASS contract | PASS instrumentation | PASS: false all rows | N/A | E1 | PASS audit |
| EXP-047R C1 oracle median <=10% | target defined | N/A | FAIL: 100% | N/A | E1 negative | RANGE CORE REJECTED |
| EXP-047R C1 oracle p90 <=25% | target defined | N/A | FAIL: 100% | N/A | E1 negative | RANGE CORE REJECTED |
| EXP-047R C2 stratification useful | hypothesis | PASS reference | FAIL: median/p90 100%; best 99.21875% | N/A | E1 negative | FAIL core role |
| EXP-047R C2 CPU primitive cheaper than full sum | threshold 1x | PASS execution | FAIL: median 2165.057x after materialization | N/A | E1 negative | FAIL current implementation |
| Range-based CPTC can approach 1.185% target | formula target | FAIL synthetic trend | FAIL oracle 84.375x target fraction | NOT TESTED | E1 negative + PROJECTED | REJECTED AS CORE |
| CPTC certificate/fallback remains correct auxiliary | PASS | PASS | PASS small audit | NOT TESTED | E1 | AUXILIARY |
| EXP-048 verifier longest-prefix correction is exact | PASS | PASS: 9 tests | PASS: B1/B2/B3 mismatch 0 | N/A | E1 | AUXILIARY PASS |
| EXP-048 predictions after first mismatch are ignored | PASS | PASS tests | PASS committed cases | N/A | E1 | PASS verifier contract |
| EXP-048 B1 one-pass 96-token verification works | defined | PASS | PASS: 18/18, 1.041667% | N/A | E1 + oracle | NON-DEPLOYABLE UPPER BOUND |
| EXP-048 B1 deployable/causal | FAIL by definition | N/A | future information true | N/A | E1 | NOT DEPLOYABLE |
| EXP-048 B2 hard Jacobi cheaper than sequential | hypothesis | PASS exact control | FAIL: p50 181.25%, p90 193.75% | N/A | E1 negative | REJECTED AS CORE |
| EXP-048 B2 exact output | defined | PASS tests | PASS: mismatch 0 | N/A | E1 | CONTROL ONLY |
| EXP-048 B3 deployable path uses no future tokens | defined | PASS instrumentation | PASS: 0 uses | N/A | E1 | PASS causality |
| EXP-048 B3 exact committed output | defined | PASS tests | PASS: mismatch 0 | N/A | E1 | PASS verifier scope |
| EXP-048 B3 p50 committed tokens >=16 early Gate | threshold | N/A | FAIL: 1 | N/A | E1 negative | SELF-DRAFT CORE REJECTED |
| EXP-048 B3 p90 fraction <=10% early Gate | threshold | N/A | FAIL: 2893.843% | N/A | E1 negative | SELF-DRAFT CORE REJECTED |
| EXP-048 B3 cost below sequential | threshold | N/A | FAIL: minimum 1333.463% | N/A | E1 negative | FAIL |
| EXP-048 B3 useful exact proposal prefix | hypothesis | N/A | FAIL: max prefix 1; 4/18 nonzero | N/A | E1 negative | FAIL |
| EXP-048 p50 accepted block >=85 promotion | required by projection | N/A | FAIL: 1 | NOT TESTED | E1 negative + PROJECTED | NOT PROMOTED |
| EXP-048 p90 target-equivalent fraction <=1.185% | required | N/A | FAIL: 2893.843% | NOT TESTED | E1 negative + PROJECTED | NOT PROMOTED |
| EXP-048 complete real operation replacement | defined | N/A | NOT TESTED | NOT TESTED | E0 | not built |
| EXP-049 causal continuous block solver exactness | defined | NOT TESTED | NOT TESTED | N/A | E0 | next Gate |
| EXP-049 Anderson improves p50 prefix >=4x over Jacobi | threshold | NOT TESTED | NOT TESTED | N/A | E0 | next Gate |
| EXP-049 p50 prefix >=16 after <=4 solver passes | threshold | NOT TESTED | NOT TESTED | N/A | E0 | next Gate |
| EXP-049 p90 accounted fraction <=10% early Gate | threshold | NOT TESTED | NOT TESTED | N/A | E0 | next Gate |
| Universal >1 exact position/round target-only guarantee | theorem target | NOT TESTED | adversarial models pending | N/A | E0 | high-risk claim |
| Savings persist with model size | formula pending | small models negative so far | NOT TESTED | NOT TESTED | E0 | A-006 unverified |
| Target RAM/SSD bandwidth sufficient | formula pending | small files only | small model only | NOT TESTED | E0 | NOT TESTED |

## Current overall classification

```text
Governance/provenance: implemented
EXP-047/047R certificate correctness: E1 PASS in scope
Range-based CPTC savings: FAIL; core rejected
EXP-048 exact block verifier: E1 auxiliary PASS
EXP-048 perfect 96-token oracle: arithmetic upper bound only, non-deployable
EXP-048 hard Jacobi: FAIL core
EXP-048 partial-layer self-draft: FAIL core
EXP-049 continuous/Anderson fixed-point Gate: pre-registered, NOT TESTED
Real 405B/8 GiB/CUDA/PCIe/SSD/TTFT/tokens/sec: NOT TESTED
E6/E7: not achieved
```

Every experiment PR must update at least one row using commit-backed evidence. PROJECTED or UNVERIFIED fields may not be upgraded to MEASURED without the matching phase.
