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
| C3 variance adaptation should continue EXP-047R | N/A | N/A | FAIL by oracle Gate | N/A | E1 decision | DO NOT IMPLEMENT AS RESCUE |
| Real Transformer operation replacement by CPTC | defined | N/A | NOT TESTED | NOT TESTED | E0 | not built |
| EXP-048 exact block verification matches sequential greedy | defined | NOT TESTED | NOT TESTED | N/A | E0 | next Gate B1 |
| EXP-048 deployable path uses no future tokens | defined | NOT TESTED | NOT TESTED | N/A | E0 | next Gate |
| EXP-048 partial-layer self-draft lowers accounted streams | formula defined | NOT TESTED | NOT TESTED | NOT TESTED | E0 | next Gate B3 |
| EXP-048 p50 accepted block >=85 | required by projection | NOT TESTED | NOT TESTED | NOT TESTED | E0 | promotion requirement |
| EXP-048 p90 target-equivalent fraction <=1.185% | required | NOT TESTED | NOT TESTED | NOT TESTED | E0 | promotion requirement |
| Savings persist with model size | formula pending | small models pending | NOT TESTED | NOT TESTED | E0 | A-006 unverified |
| Target RAM/SSD bandwidth sufficient | formula pending | small files only | small model only | NOT TESTED | E0 | NOT TESTED |

## Current overall classification

```text
Governance/provenance: implemented
EXP-047/047R certificate correctness: E1 PASS in scope
Global/oracle-tight/stratified range CPTC savings: FAIL; core rejected
Certificate and exact fallback: auxiliary retained
EXP-048 causal block amortization: pre-registered, NOT TESTED
Real 405B/8 GiB/CUDA/PCIe/SSD/TTFT/tokens/sec: NOT TESTED
E6/E7: not achieved
```

Every experiment PR must update at least one row using commit-backed evidence. PROJECTED or UNVERIFIED fields may not be upgraded to MEASURED without the matching phase.
