# Validation Matrix

Legend: PASS validated in scope; FAIL contradicted; PARTIAL limited; NOT TESTED unavailable; N/A not applicable.

| Claim | Phase A | Phase B | Phase C | Phase D | Evidence | Current verdict |
|---|---|---|---|---|---:|---|
| Final target remains 405B/8 GiB/4B-class | defined | N/A | N/A | NOT TESTED | E0 | objective only |
| Real 405B execution | N/A | N/A | NOT TESTED | NOT TESTED | E0 | NOT TESTED |
| 405B peak VRAM <=8 GiB | formula incomplete | N/A | N/A | NOT TESTED | E0 | NOT TESTED |
| 405B TTFT/tokens per second | formula incomplete | N/A | N/A | NOT TESTED | E0 | NOT TESTED |
| Original 405B quality preserved | contract incomplete | N/A | tiny models only | NOT TESTED | E0 | NOT TESTED |
| mmap/index/DAG bounded functions | PASS | PASS | finite-domain PARTIAL | NOT TESTED | E1/E2 | auxiliary |
| Raw prefix graph scales | FAIL | FAIL | held-out start 0% | N/A | E2 negative | REJECTED |
| Metadata size equals traffic | FAIL | N/A | N/A | N/A | E1 theorem | false |
| EXP-047/047R certificate correctness | PASS | PASS | PASS audit | N/A | E1 | auxiliary PASS |
| Range CPTC useful savings | hypothesis | FAIL | exact oracle median/p90 100% | N/A | E1 negative | REJECTED CORE |
| EXP-048 exact block verifier | PASS | PASS 9 tests | mismatch 0 | N/A | E1 | auxiliary PASS |
| Perfect 96-token proposal arithmetic | defined | PASS | 1.041667%, future-aware | N/A | E1 oracle | upper bound only |
| Hard Jacobi cheaper than sequential | hypothesis | exact control | p50 181.25% | N/A | E1 negative | REJECTED |
| Partial-layer self-draft useful | hypothesis | reference | p50 committed 1, p90 2893.843% | N/A | E1 negative | REJECTED |
| EXP-049 solver/fault handling | PASS | 9 tests PASS | workflow PASS | N/A | E1 | auxiliary PASS |
| EXP-049 target-only fixed-point useful | hypothesis | positive controls | p50 prefix 4.5, p90 168.78% | N/A | E1 negative | REJECTED |
| Universal >1 exact position/target round | theorem target | adversarial PASS | hidden chain refutes | N/A | E1 negative | REJECTED IN SCOPE |
| EXP-050 accounting/counterexample implementation | PASS | 9 tests PASS | workflow PASS | N/A | E1 | PASS |
| EXP-050 cross-draft committed output exact | defined | verifier PASS | PASS: 0/108 mismatch | N/A | E1 | PASS scope |
| EXP-050 E1/E2 uses no target future tokens | defined | instrumentation PASS | PASS: 0 uses | N/A | E1 | PASS causality |
| EXP-050 E3 future oracle aligned | control | PASS | PASS: 0 failures | N/A | E1 oracle | non-deployable |
| Fixed target-independent draft guarantees first token | universal claim | adversarial PASS | FAIL: prefix 0 counterexample | N/A | E1 negative | REJECTED UNIVERSALLY |
| EXP-050 favorable pool p50 prefix >=16 | threshold | N/A | FAIL: 0.5 | N/A | E1 negative | FAIL |
| EXP-050 favorable pool maximum prefix useful | hypothesis | N/A | FAIL: maximum 3 | N/A | E1 negative | FAIL |
| EXP-050 p90 normalized fraction <=10% | threshold | N/A | FAIL: 163.20987654% | N/A | E1 negative | FAIL |
| EXP-050 all families have useful acceptance | threshold | N/A | FAIL: Korean/JSON false | N/A | E1 negative | FAIL |
| EXP-050 target-size trend non-degrading | threshold | N/A | FAIL: medians 1.0/0.0/0.5 | N/A | E1 negative | FAIL |
| 4B draft final exact prefix >=507 | derived requirement | N/A | current max 3 | NOT TESTED | E0+PROJECTED | far from target |
| Tested fixed external draft pool practical core | hypothesis | N/A | FAIL all usefulness Gates | N/A | E1 negative | REJECTED |
| EXP-051 intermediate final-depth reconstruction | defined | NOT TESTED | NOT TESTED | N/A | E0 | next Gate |
| EXP-051 suffix-stable oracle median bytes <=10% | threshold | NOT TESTED | NOT TESTED | N/A | E0 | next Gate |
| EXP-051 suffix-stable oracle p90 bytes <=25% | threshold | NOT TESTED | NOT TESTED | N/A | E0 | next Gate |
| EXP-051 median stable block depth <=10% | threshold | NOT TESTED | NOT TESTED | N/A | E0 | next Gate |
| EXP-051 fixed early depth exact across corpus | hypothesis | NOT TESTED | NOT TESTED | N/A | E0 | next Gate |
| Universal fixed early-exit depth exists | theorem target | late-flip pending | NOT TESTED | N/A | E0 | next Gate adversary |
| Sound nonlinear tail certificate exists | defined | NOT TESTED | NOT TESTED | NOT TESTED | E0 | blocked by oracle Gate |
| Real target layers causally skipped | defined | N/A | NOT TESTED | NOT TESTED | E0 | not built |
| Savings persist with model size | formula pending | tiny negative trends | NOT TESTED | NOT TESTED | E0 | unverified |
| Target RAM/SSD bandwidth sufficient | formula pending | small files only | small model only | NOT TESTED | E0 | NOT TESTED |

## Current overall classification

```text
Governance/provenance: implemented
CPTC certificate/fallback: E1 auxiliary
Exact block verifier: E1 auxiliary
Picard/Anderson reference: E1 auxiliary
Range CPTC core: rejected
Hard Jacobi core: rejected
Partial-layer recursive draft core: rejected
Target-only fixed-point core: rejected
Target-independent external draft universal/practical tested pool: rejected
EXP-051 layer-finalization Gate: pre-registered, NOT TESTED
Real operation replacement: NOT TESTED
405B/8 GiB/CUDA/PCIe/SSD/TTFT/tokens/sec: NOT TESTED
E6/E7: not achieved
```

Every experiment PR must update at least one row using commit-backed evidence. PROJECTED or UNVERIFIED fields may not become MEASURED without the matching phase.

<!-- EXP-052-AUTHORITATIVE-FINAL -->
## EXP-051/052 addendum

| Claim | Evidence | Verdict |
|---|---:|---|
| EXP-051 p90 favorable traffic <=25% | E1: 99.8011% | REJECTED |
| EXP-052 exact table integrity | E1: wrong hits 0 | AUXILIARY |
| EXP-052 held-out hit >=98.8148% | E1: 0% all P0/S0 families | REJECTED |
| EXP-052 natural reuse >=85 | E1: median/max 1/1 | REJECTED |
| EXP-052 p90 fraction <=1.185185% | E1: 600% | REJECTED |
| EXP-052 budget fallback <=1.185185% | E1: 99.9999364% | REJECTED |
| EXP-053 exact circuit compiler | E0 | NEXT GATE |

<!-- EXP-053-AUTHORITATIVE-FINAL -->
## EXP-053 addendum

| Claim | Evidence | Verdict |
|---|---:|---|
| EXP-053 exact circuit equality | E1: 0 mismatch / 4,506,624 inputs | PASS reference |
| EXP-053 no truth-table representation | E1: 0 cases | PASS |
| EXP-053 p50 query fraction <=10% | E1: 84.1683% | REJECTED |
| EXP-053 p90 query fraction <=25% | E1: 94.1072% | REJECTED |
| EXP-053 dense-random p50 <=25% | E1: 92.4521% | REJECTED |
| EXP-053 projected storage <=1 TiB | PROJECTED: max 255.5966 TiB | REJECTED |
| EXP-054 reduced decision diagram | E0 | NEXT GATE |

<!-- EXP-054-AUTHORITATIVE-FINAL -->
## EXP-054 addendum

| Claim | Evidence | Verdict |
|---|---:|---|
| EXP-054 exact equality | E1: 0 mismatch / 9,013,248 | PASS reference |
| EXP-054 no truth table | E1: 0 cases | PASS |
| EXP-054 p50 path <=10% | E1: 35% | REJECTED |
| EXP-054 p90 path <=25% | E1: 95% | REJECTED |
| EXP-054 no ceiling/fallback | E1: 0/48 | PASS |
| EXP-054 storage <=1 TiB | PROJECTED: 202.2479 TiB | REJECTED |
| EXP-054 adversarial growth <=1.5 | E1: 1.6873x/bit | REJECTED |
| EXP-055 word-level grouping | E0 | NEXT GATE |

<!-- EXP-058-AUTHORITATIVE-FINAL -->
## EXP-058 closure

Q4 checksum agreement 144/144; full integer/rational rank 144/144; certificate/control mismatches 0; p50/p90 exact factor operation lower bound 200%/200%; p50/p90 factor-storage lower bound 200%/200%. 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.

<!-- EXP-059-AUTHORITATIVE-FINAL -->
## EXP-059 closure

Q4 checksum agreement PASS; registration 144/144; operator certificates 612; control/certificate mismatches 0; p50/p90 selected displacement-rank fraction 100%/100%; favorable query lower bound 100%/100%; favorable generator storage 200%/200%. Hardware and 405B remain NOT TESTED.

<!-- EXP-060-AUTHORITATIVE-FINAL -->
## EXP-060 closure

Q4 checksum agreement PASS; dense registration 144/144; formats 1224; reconstruction/control mismatches 0; exact zero fraction p50/p90 17.76%/20.37%; operation fraction 82.22%/85.06%; query-byte fraction 150.93%/200.86%. Physical sparse kernels, 405B, 8 GiB, and target hardware remain NOT TESTED.

<!-- EXP-061-AUTHORITATIVE-FINAL -->
## EXP-061 closure

Reference/hooked output tokens 1,152/1,152 exact; projection registrations 147; calls 56,448; warm calls 54,684; hook/control mismatches 0; exact-zero count 0; warm p50/p90 operation fraction 100.002%/100.391%; query bytes 100.004%/101.566%. 405B and hardware remain NOT TESTED.

<!-- EXP-062-AUTHORITATIVE-FINAL -->
## EXP-062 closure

Cases 18; forwards 1,152; attention rows 9,216; token/registration/control mismatches 0; warm eligible probabilities 8,404,224; exact non-mask zeros 2,564; whole-model p50/p90 operations 100.048%/100.154%; bytes 100.093%/100.303%. 405B and hardware remain NOT TESTED.

<!-- EXP-063-AUTHORITATIVE-FINAL -->
## EXP-063 closure

18 cases; 1,152 forwards; 147,456 group rows; exact K duplicates 0; exact KV duplicates 0; mismatches 0; warm p50/p90 operations 100.021%/100.027%; bytes 106.263%/119.401%. 405B and hardware NOT TESTED.

<!-- EXP-064-AUTHORITATIVE-FINAL -->
## EXP-064 closure

153 tensors; 144 dense; 1,683 plans; zero checksum/reconstruction/control mismatch; exact identical/sign row matrices 0/0; p50/p90 operation 100%/100%; query bytes 100%/100%; projected static storage 211.31 GB. 405B execution and hardware NOT TESTED.
