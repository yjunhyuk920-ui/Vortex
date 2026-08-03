# Validation Matrix

Legend:

- PASS: validated in the declared phase and scope.
- FAIL: contradicted in the declared scope.
- PARTIAL: limited-scope evidence only.
- NOT TESTED: environment or implementation unavailable.
- N/A: phase is not applicable.

| Claim | Phase A | Phase B | Phase C | Phase D | Evidence | Current verdict |
|---|---|---|---|---|---:|---|
| Final target remains 405B/8 GiB/4B-class | defined | N/A | N/A | NOT TESTED | E0 | objective only |
| Current environment has target 8 GiB GPU | N/A | N/A | N/A | NOT TESTED | E0 | unavailable |
| Real 405B checkpoint execution | N/A | N/A | NOT TESTED | NOT TESTED | E0 | NOT TESTED |
| Actual 405B peak VRAM <=8 GiB | formula incomplete | N/A | N/A | NOT TESTED | E0 | NOT TESTED |
| Actual 405B TTFT/tokens per second | formula incomplete | N/A | N/A | NOT TESTED | E0 | NOT TESTED |
| Actual 4B-class user-perceived speed | formula incomplete | N/A | N/A | NOT TESTED | E0 | NOT TESTED |
| Original 405B quality preserved | contract incomplete | N/A | small models only | NOT TESTED | E0 | NOT TESTED |
| mmap decision VM exact replay | PASS | PASS | bounded TinyLlama integration PASS | NOT TESTED | E2 | auxiliary validated |
| mmap corruption/atomicity handling | PASS | PASS | N/A | NOT TESTED | E1 | PASS in tested files |
| Raw exact-prefix graph scales sublinearly | FAIL on measured grammar | FAIL | FAIL: 64/64 nodes, 0% held-out | N/A | E2 | REJECTED |
| Exact future-suffix DAG compresses body | PASS finite horizon | PASS | PARTIAL: 64->38 nodes | N/A | E2 | auxiliary PASS |
| Causal unseen-prompt start routing | open | synthetic not run | FAIL in prior held-out grammar: 0% | NOT TESTED | E2 negative | unresolved blocker |
| Metadata size equals per-token traffic | FAIL | N/A | N/A | N/A | E1 theorem | false |
| One host probe/token proves latency failure | FAIL | PASS counterexample prototype | N/A | NOT TESTED | E1 | false |
| CPTC mathematical confidence sequence is valid | pending | pending | NOT TESTED | NOT TESTED | E0 | active |
| CPTC optimized implementation matches reference/fallback | pending | pending | NOT TESTED | N/A | E0 | active |
| CPTC uses no future tokens | contract PASS | pending instrumentation | NOT TESTED | NOT TESTED | E0 | active |
| CPTC synthetic positive control reads <=25% tiles | N/A | NOT TESTED | N/A | N/A | E0 | promotion gate |
| CPTC adversarial cases fall back to 100% | defined | NOT TESTED | N/A | N/A | E0 | promotion gate |
| CPTC silent wrong accepts are zero | defined | NOT TESTED | NOT TESTED | NOT TESTED | E0 | mandatory |
| CPTC real Transformer operation replacement | defined | N/A | NOT TESTED | NOT TESTED | E0 | Phase C pending |
| CPTC held-out real-model certified coverage >0 | defined | N/A | NOT TESTED | NOT TESTED | E0 | Phase C pending |
| CPTC savings grow or persist with model size | formula pending | N/A | NOT TESTED | NOT TESTED | E0 | A-006 unverified |
| Target RAM/SSD bandwidth sufficient | formula pending | synthetic trends only | small file only | NOT TESTED | E0 | NOT TESTED |
| Full runtime fallback cannot worsen declared output | contract defined | pending tests | NOT TESTED | NOT TESTED | E0 | active |

## Mandatory update rule

Every experiment PR must update at least one row with a commit-backed result. No row may be upgraded from PROJECTED or UNVERIFIED to MEASURED without an actual run in the matching phase.
