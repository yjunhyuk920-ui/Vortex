# Validation Matrix

Legend: PASS validated in scope; FAIL contradicted; PARTIAL limited; NOT TESTED unavailable; N/A not applicable.

| Claim | Phase A | Phase B | Phase C | Phase D | Evidence | Current verdict |
|---|---|---|---|---|---:|---|
| Final target remains 405B/8 GiB/4B-class | defined | N/A | N/A | NOT TESTED | E0 | objective only |
| Target 8 GiB GPU available now | N/A | N/A | N/A | NOT TESTED | E0 | unavailable |
| Real 405B checkpoint execution | N/A | N/A | NOT TESTED | NOT TESTED | E0 | NOT TESTED |
| 405B peak VRAM <=8 GiB | formula incomplete | N/A | N/A | NOT TESTED | E0 | NOT TESTED |
| 405B TTFT/tokens per second | formula incomplete | N/A | N/A | NOT TESTED | E0 | NOT TESTED |
| 4B-class user-perceived performance | target defined | N/A | NOT TESTED | NOT TESTED | E0 | NOT TESTED |
| Original 405B quality preserved | contract incomplete | N/A | small models only | NOT TESTED | E0 | NOT TESTED |
| mmap decision VM exact replay | PASS | PASS | bounded TinyLlama PASS | NOT TESTED | E2 | auxiliary validated |
| mmap corruption/atomicity | PASS | PASS | N/A | NOT TESTED | E1 | PASS tested scope |
| Raw exact-prefix graph scales sublinearly | FAIL measured grammar | FAIL | FAIL 64/64, held-out 0% | N/A | E2 negative | REJECTED |
| Exact future-suffix DAG compresses body | PASS finite horizon | PASS | PARTIAL 64->38 | N/A | E2 | auxiliary PASS |
| Causal unseen-prompt start routing | open | not solved | prior held-out 0% | NOT TESTED | E2 negative | unresolved |
| Metadata size equals per-token traffic | FAIL | N/A | N/A | N/A | E1 theorem | false |
| One host probe/token proves latency failure | FAIL | counterexample PASS | N/A | NOT TESTED | E1 | false |
| CPTC alpha-spending Serfling formula implemented consistently | PASS | PASS: 0/525 bound mismatches | NOT TESTED | N/A | E1 | PASS declared assumptions |
| CPTC optimized/fallback matches reference | PASS contract | PASS: 0 fallback mismatches | NOT TESTED | N/A | E1 | PASS synthetic scope |
| CPTC uses no future generated tokens | PASS contract | PASS instrumentation | NOT TESTED | NOT TESTED | E1 | PASS current primitive |
| CPTC silent wrong accepts are zero | defined | PASS: 0/525 | NOT TESTED | NOT TESTED | E1 | PASS corpus only |
| CPTC adversarial cases exact-fallback | defined | PASS: 15/15 | N/A | N/A | E1 | PASS |
| CPTC positive control reads <=25% tiles | N/A | PASS: 10.449% at N=1024 | N/A | N/A | E1 | primitive PASS only |
| CPTC broad certified coverage is useful | hypothesis | FAIL: 4/525 certified | NOT TESTED | N/A | E1 negative | REVISE |
| CPTC broad fallback is low | hypothesis | FAIL: 99.238% fallback | NOT TESTED | N/A | E1 negative | REVISE |
| CPTC broad mean evaluated fraction approaches target | formula target 1.185% | FAIL: 98.294% at N=1024 | NOT TESTED | NOT TESTED | E1 negative | far from target |
| CPTC selector overhead is lower than reference | formula pending | FAIL in Python: ~8.8–9.1x slower | NOT TESTED | NOT TESTED | E1 negative | current implementation fails |
| Sound checkpoint-derived per-tile bounds exist | pending | synthetic range only | NOT TESTED | NOT TESTED | E0 | EXP-047R |
| Oracle-tight real-state range closes early | pending | N/A | NOT TESTED | N/A | E0 | decisive next test |
| Deployable stratified bounds improve C0 | pending | reference pending | NOT TESTED | NOT TESTED | E0 | decisive next test |
| Real Transformer operation replacement | defined | N/A | NOT TESTED | NOT TESTED | E0 | blocked pending audit |
| Held-out real-model certified coverage >0 | defined | N/A | NOT TESTED | NOT TESTED | E0 | blocked pending audit |
| Savings persist/grow with model size | formula pending | N/A | NOT TESTED | NOT TESTED | E0 | A-006 unverified |
| Target RAM/SSD bandwidth sufficient | formula pending | small files only | small model only | NOT TESTED | E0 | NOT TESTED |
| Full fallback cannot worsen declared output | contract PASS | PASS synthetic | NOT TESTED | NOT TESTED | E1 | real-model pending |

## Current overall classification

```text
Governance/provenance system: implemented
EXP-047 correctness primitive: E1 PASS
EXP-047 global-range architecture performance: REVISE / not promoted
Real-model operation skipping: NOT TESTED
Phase D: NOT TESTED
E6/E7: not achieved
```

Every experiment PR must update at least one row using commit-backed evidence. PROJECTED/UNVERIFIED fields may not be upgraded to MEASURED without the matching phase.
