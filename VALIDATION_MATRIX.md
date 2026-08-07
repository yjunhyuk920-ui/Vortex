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

<!-- EXP-065-AUTHORITATIVE-FINAL -->
## EXP-065 closure

153 tensors; 144 dense; 6,108 plans; 306 selected two-prime certificates; zero checksum/witness/control mismatch; all selected ranks 4/full; p50/p90 operation 203.891%/215.385%; storage 100.234%/101.042%; projected storage 202.66 GB. 405B execution and hardware NOT TESTED.

<!-- EXP-066-072A-AUTHORITATIVE-CATCHUP -->
## EXP-066 through EXP-072A closure

| Claim | Evidence | Verdict |
|---|---:|---|
| EXP-066 exact TT/MPO static p50 <=10% | E1: 11.0524% | REJECTED |
| EXP-067 joint exact Q/K/V work p50 <=10% | E1: 100% | REJECTED |
| EXP-068 favorable output-head-only demand p50 <=10% | E1: 13.7697% | REJECTED |
| EXP-069 causal exact temporal mandatory work p50 <=10% | E1: 100% | REJECTED |
| EXP-070 exact local-pattern operations p50 <=10% | E1: 88.4856% | REJECTED |
| EXP-071 registered theorems prove model-wide impossibility | E1 applicability audit: 0/9 CKL-covered families, no direct sum | NOT CERTIFIED |
| EXP-072A finite-domain basis-map injectivity | E1: 272/272 unique, 0 collision | PASS reference |
| Universal self-contained 405B-Q4 artifact fits 8 GiB | DERIVED: 188.9883 GiB required / 8 GiB allowed | REJECTED |
| Cold-backed exact online execution is impossible | outside EXP-072A; EXP-071 insufficient | NOT CERTIFIED |
| Private Ubuntu target inventory | MEASURED sanitized Stage 1: 8,192 MiB GPU, 23.4983 GiB RAM, 97.6183 GiB root free, max PCIe Gen2 x16 | PASS CALIBRATION INVENTORY ONLY — `results/exp_073/summary.json` |
| Same-machine storage/H2D/native 4B Q4 baseline | no authorized Stage 2 run | NOT TESTED — EXP-073 remains active |

Current overall classification: self-contained universal hot-artifact core rejected; restricted synthesis auxiliary; cold-backed online execution open but unsupported; Phase D/E6/E7 not achieved.

<!-- EXP-074-AUTHORITATIVE-FINAL -->
## EXP-074 closure

| Claim | Evidence | Verdict |
|---|---:|---|
| Reference accounting controls | E1: 7/7 pass | PASS reference |
| MTP-1 + fixed experts + free draft p50 <=1.2x 1B | DERIVED: 10.0x | REJECTED |
| Ideal fixed-route/free-draft block has path <=16 tokens | DERIVED: 9 | NECESSARY CEILING PASS ONLY |
| Independent-uniform expected routes reach p50 <=1.2x | DERIVED: 98 perfect tokens | UNFAVORABLE CONTROL |
| Fixed routes + 0.8B proposal reach p50 <=1.2x | DERIVED: 25 perfect tokens | UNVERIFIED ACCEPTANCE |
| 81 GB direct pull fits current free disk | DERIVED: 22.1812 GiB remains | NOMINAL FIT |
| 81 GB direct pull preserves 30 GiB workspace | DERIVED: false | FAIL |
| Real MTP accepted-prefix distribution | no checkpoint run | NOT TESTED |
| Real expert route locality | no checkpoint run | NOT TESTED |
| Physical 1B-class latency or VRAM | no runtime/server run | NOT TESTED |
| MoE surrogate validates dense 405B | outside claim | FALSE SUBSTITUTION PROHIBITED |

Current classification after EXP-074: MTP-1 plus paging rejected as a 1B-class
core; long-block candidate revised pending causal proposal and routing evidence;
no model download; Phase D/E6/E7 unchanged.

<!-- EXP-075-AUTHORITATIVE-FINAL -->
## EXP-075 closure

| Claim | Evidence | Verdict |
|---|---:|---|
| Pinned 0.8B config declares native MTP | E1 static: `mtp_num_hidden_layers=1` | PASS interface |
| Pinned 0.8B index contains registered MTP weights | E1 static: 15/15 exact keys | PASS interface |
| Pinned vLLM exposes Qwen3.5 MTP mapping/loader | E1 static: 3/3 source surfaces | PASS interface |
| Metadata audit controls | E1: 5/5 | PASS |
| Checkpoint weight was downloaded | no payload URL or file | FALSE / NOT PERFORMED |
| Native MTP proposals are causally correct | no execution | NOT TESTED |
| Accepted-prefix p05/p50 closes EXP-074 | no proposal trace | NOT TESTED |
| Quantized path retains and executes MTP | no quantized artifact | NOT TESTED |
| Current hardware supports the selected runtime | source audit only | NOT TESTED |
| Qwen surface validates arbitrary dense 405B | outside claim | FALSE SUBSTITUTION PROHIBITED |

Current classification after EXP-075: the metadata prerequisite passes and
authorizes a pinned 0.8B accepted-prefix falsification. Runtime feasibility and
the dense mission are unchanged; Phase D/E6/E7 remain not achieved.

<!-- EXP-076-AUTHORITATIVE-FINAL -->
## EXP-076 closure

| Claim | Evidence | Verdict |
|---|---:|---|
| Pinned unchanged BF16 payload loaded | E1: weight SHA-256 `04b1c301...fe4696` | PASS input |
| Prompt split and K selection preregistered | E1: 6 build, 18 held-out, selected K=4 | PASS |
| Native proposal never reads target future tokens | E1: 0 reads | PASS reference |
| Exact longest-prefix verifier never silently accepts wrong token | E1: 0 wrong accepts | PASS reference |
| Commit replay and rollback recompute preserve exact state | E1: 0/0 mismatches | PASS reference |
| Held-out accepted-prefix p50 reaches 11 | E1: 4 | REJECTED |
| Held-out accepted-prefix p05 reaches 9 | E1: 0 | REJECTED |
| Every required family reaches both acceptance minima | E1: false | REJECTED |
| Realized p50/p95 traffic reaches 1.2x/1.5x | fail-closed with 2/18 zero accepts | REJECTED |
| Physical vLLM/GPU/quantized performance | CPU reference only | NOT TESTED |
| 35B/122B router locality and scaling | no model/run | NOT TESTED |
| Arbitrary dense 405B runs in 8 GiB at 4B speed | outside claim | NOT VALIDATED |

Current classification after EXP-076: native-MTP long blocks are rejected as
the registered surrogate core; reference integrity passes; no larger-model or
hardware continuation is promoted; Phase D/E2-E7 remain not achieved.

<!-- EXP-077A-AUTHORITATIVE-FINAL -->
## EXP-077A closure

| Claim | Evidence | Verdict |
|---|---:|---|
| Pinned unchanged target trace replay | E1: 24 cases, 192 positions, 0 mismatch | PASS control |
| Realized selected MLP fraction <=10% | E1: 358/3,584 = 9.988839% | PASS budget |
| Held-out top-1 agreement >=99% | E1: 71.5278% | REJECTED |
| Every family top-1 agreement >=95% | E1: 54.1667%-83.3333% | REJECTED |
| Mean target-to-candidate KL <=0.02 | E1: 0.884161 | REJECTED |
| p95 target-to-candidate KL <=0.05 | E1: 3.080752 | REJECTED |
| 20% channel arm establishes near-lossless ceiling | E1: 83.3333% top-1 | REJECTED |
| Deployable selector/full gate-up cost | granted free oracle | NOT TESTED |
| Attention/DeltaNet/LM-head fracturing | outside experiment | NOT TESTED |
| Physical speed, VRAM, 35B/122B/405B | no server/GPU run | NOT TESTED |

Current classification after EXP-077A: activation-norm individual-channel MLP
fracturing is rejected under a favorable oracle; no core candidate survives;
Phase D/E2-E7 remain not achieved.

## EXP-078A preregistered Gate

| Claim | Evidence | Verdict |
|---|---:|---|
| Anchor macro equation equals factorized complete MLP at anchor | E1 bounded reference control | PREREGISTERED |
| 0.8B hot macro MLP work fraction | DERIVED: 9.52381% | NECESSARY HOT CEILING |
| 0.8B direct-build p50/p05 lifetime | DERIVED: 13,821/6,249 | PREREGISTERED |
| 122B nine-path hot fraction | DERIVED: 11.1111% | NECESSARY HOT CEILING |
| 122B direct-build p50/p05 lifetime | DERIVED: 115,299/26,354 | PREREGISTERED |
| Baseline frozen trace replay has zero mismatch | no EXP-078A run | NOT TESTED |
| Frozen anchor survives next seven positions at quality Gate | no EXP-078A run | NOT TESTED |
| Physical macro construction/application speed | reference only | NOT TESTED |
| Sentinel, exact repair, fallback | outside experiment | NOT TESTED |
| 35B/122B/405B execution | no payload/run | NOT TESTED |

<!-- EXP-078A-AUTHORITATIVE-FINAL -->
## EXP-078A closure

| Claim | Evidence | Verdict |
|---|---:|---|
| Pinned unchanged target replay | E1: 24 cases, 192 positions, 0 mismatch | PASS control |
| Exact anchor macro algebra/reference controls | E1: 6/6 contract tests | PASS reference |
| Held-out top-1 agreement >=99% | E1: 0/126 = 0% | REJECTED |
| Every family top-1 agreement >=95% | E1: all six at 0% | REJECTED |
| Mean target-to-candidate KL <=0.02 | E1: 14.898423 | REJECTED |
| p95 target-to-candidate KL <=0.05 | E1: 25.335417 | REJECTED |
| p50/p05 lifetime reaches 13,821/6,249 | E1: 0/0 | REJECTED |
| Physical macro construction/application | not executed | NOT TESTED |
| Delta update, sentinel, exact repair/fallback | outside frozen scope | NOT TESTED |
| 35B/122B/405B and E2-E7 | no model/server run | NOT TESTED |

Current classification after EXP-078A: unchanged prior-token tangent operators
are rejected; a causally delta-updated operator remains only an unsupported new
mechanism class; no core candidate survives.

## EXP-079A preregistered Gate

| Claim | Evidence | Verdict |
|---|---:|---|
| DCT pilot split and row-block L2 enclosure are exact-real sound | bounded reference/property tests | PREREGISTERED |
| Proof-state transition cannot commit an open or over-budget proof | pure state-machine tests | PREREGISTERED |
| Dense-405B shape plan p50 traffic <=1.185185% | DERIVED: 1.08883712% | FAVORABLE ROUTE ONLY |
| Dense-405B shape plan p50 favorable operations <=1.185185% | DERIVED: 0.976002844% | FAVORABLE ROUTE ONLY |
| Sidecar plus proof records fit inside 8 GiB | DERIVED: 487,843,776 bytes | PARTIAL; KV/work excluded |
| Pinned unchanged target replay has zero mismatch | no EXP-079A run | NOT TESTED |
| Held-out p50-arm top-1 agreement >=99% | no EXP-079A run | NOT TESTED |
| Every family top-1 agreement >=95% | no EXP-079A run | NOT TESTED |
| Mean/p95 KL <=0.02/0.05 | no EXP-079A run | NOT TESTED |
| Minimum local sound-radius p50/p95 <=1.0x signal | no EXP-079A run | NOT TESTED |
| Nonlinear token-margin proof and exact fallback | favorable grants only | NOT TESTED |
| Physical CUDA/VRAM/speed and 122B/405B | no payload/server run | NOT TESTED |

Current classification before EXP-079A execution: the exact DCT
block-zonotope mechanism has passed only E0 arithmetic admission. No core
candidate is promoted; E1 result and Phase D/E2-E7 remain absent.

<!-- EXP-079A-AUTHORITATIVE-FINAL -->
## EXP-079A closure

| Claim | Evidence | Verdict |
|---|---:|---|
| Pinned unchanged target replay | E1: 24 cases, 192 decisions, 0 mismatch | PASS control |
| p50 charged logical traffic <=1.185185% | E1 derived: 1.163034707% | PASS budget |
| p50 favorable logical operations <=1.185185% | E1 derived: 1.111385105% | PASS budget |
| Held-out top-1 agreement >=99% | E1: 6/144 = 4.1667% | REJECTED |
| Every family top-1 agreement >=95% | E1: 0%-8.3333% | REJECTED |
| Mean/p95 KL <=0.02/0.05 | E1: 7.544862/12.616226 | REJECTED |
| Minimum local sound-radius p50/p95 <=1.0x | E1: 48.663918x/57.778748x | REJECTED |
| p95 allowance establishes a near-lossless ceiling | E1: 14/144 top-1, median radius 46.993979x | REJECTED |
| Deployable selector/nonlinear proof/fallback | dual oracles and fallback free | NOT TESTED |
| Physical Q4/CUDA/VRAM/speed, 122B/405B | no model/server run | NOT TESTED |

Current classification after EXP-079A: the fixed DCT pilot plus correlated
row-block proof-ball path is rejected at E1. CPSM is not validated as a runtime,
no core candidate survives, and Phase D/E2-E7 remain not achieved.
