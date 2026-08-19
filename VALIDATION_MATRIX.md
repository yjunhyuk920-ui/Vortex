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

## EXP-080A preregistered Gate

| Claim | Evidence | Verdict |
|---|---:|---|
| Exact integer Strassen control equals classical product | no run | PREREGISTERED |
| Constructive Hyperblock traffic reaches p50 fraction | equation pending run | PREREGISTERED |
| Constructive Hyperblock arithmetic reaches p50 fraction | equation pending run | PREREGISTERED |
| Favorable workspace fits 8 GiB | shape equation pending run | PREREGISTERED |
| Unit-constant exponent oracle supplies theoretical headroom | diagnostic only | NOT A PROMOTION GATE |
| Future activation block is produced causally | granted perfect-future oracle | NOT TESTED / NON-DEPLOYABLE |
| Physical latency, CUDA, 122B/405B execution | no runtime | NOT TESTED |

<!-- EXP-080A-AUTHORITATIVE-FINAL -->
## EXP-080A closure

| Claim | Evidence | Verdict |
|---|---:|---|
| Exact signed-integer Strassen equals classical product | E1 reference: 80/80 comparisons | PASS control |
| Constructive traffic reaches p50 fraction | best `K=16,384`: 0.006103516% <=1.185185% | PASS upper bound |
| Constructive arithmetic reaches p50 fraction | best `K=16,384`: 36.111580% >1.185185% | REJECTED (`30.469145x` miss) |
| One registered block jointly passes traffic and arithmetic | 0 of 10 block lengths | REJECTED |
| Favorable incomplete workspace fits 8 GiB | best row: 7.539063 GiB | PASS upper bound only |
| Unit-constant exponent supplies constructive evidence | first theoretical pass `K=512` | NOT CONSTRUCTIVE |
| Streamed 4B draft plus exponent control passes | first theoretical pass `K=8,192` | DIAGNOSTIC; CAUSAL SOURCE ABSENT |
| Future activation block is produced causally | perfect-future oracle only | NOT TESTED / NON-DEPLOYABLE |
| Physical Q4/BF16 reduction fidelity or CUDA kernel | no implementation | NOT TESTED |
| Physical latency, peak VRAM, 122B/405B, Ubuntu host | no model/server run | NOT TESTED |

Current classification after EXP-080A: standard recursive Strassen is rejected
as the Hyperblock arithmetic core under the frozen interface. The cost model is
retained as an auxiliary. No core candidate survives; Phase D/E2-E7 remain not
achieved.

## EXP-081A preregistered Gate

| Claim | Evidence | Verdict |
|---|---:|---|
| In-code finite-field errors recover exactly | no run | PREREGISTERED |
| Out-of-code/faulted candidates are fingerprint-rejected | no run | PREREGISTERED |
| 405B favorable operations <=1.185185% | derived preregistration: 0.2148842% | NECESSARY PASS |
| 405B favorable traffic <=1.185185% | derived preregistration: 0.9265250% | NECESSARY PASS |
| Sidecar <=8 GiB | derived preregistration: 3.976903 GiB | NECESSARY PASS |
| Held-out exact residual-code coverage >=99.75% | no run | PREREGISTERED |
| Held-out corrected L2 p50/p95 <=0.01/0.05 | no run | PREREGISTERED |
| Complete model operation replacement or output quality | necessary projection Gate only | NOT TESTED |
| CUDA, peak VRAM, latency, 122B/405B | no hardware/model run | NOT TESTED |

<!-- EXP-081A-AUTHORITATIVE-FINAL -->
## EXP-081A closure

| Claim | Evidence | Verdict |
|---|---:|---|
| Exact/fail-closed finite-field reference | E1: 321/321 cases, zero wrong accept | PASS control |
| Metadata-complete favorable operations <=1.185185% | DERIVED: 0.2148841982% | PASS before fallback |
| Metadata-complete favorable traffic <=1.185185% | DERIVED: 0.9266579409% | PASS before fallback |
| Sidecar <=8 GiB | DERIVED: 3.978126 GiB | PASS shape equation |
| Held-out exact residual-code coverage >=99.75% | E1: 8.681672% | REJECTED |
| Every family exact coverage >=99% | E1: 6.7164%-10.7143% | REJECTED |
| Corrected relative-L2 p50/p95 <=0.01/0.05 | E1: 0.351269/1.213828 | REJECTED |
| Observed post-fallback traffic <=1.185185% | DERIVED: 92.244986% | REJECTED (`77.8317x`) |
| Independent deterministic reproduction | core plus 7/7 scientific files match | PASS provenance |
| Complete model operation replacement/output quality | projection observation only | NOT TESTED |
| CUDA, peak VRAM, latency, 122B/405B | no hardware/model run | NOT TESTED |

Current classification after EXP-081A: the exact syndrome algebra is retained
as an auxiliary fail-closed primitive, but the nonlinear lookup plus rank-eight
residual population assumption is rejected. No core candidate survives; Phase
D and E2-E7 remain not achieved.

## E0 decision/proof-trace triage

| Claim | Evidence | Verdict |
|---|---:|---|
| Entire 405B output head can be a core saving | DERIVED share `0.520459999%` | REJECTED AS CORE |
| Six-check trace verifier fits operations | DERIVED `0.058120260%` | PASS verifier-only ceiling |
| Six-check trace verifier fits traffic/sidecar | DERIVED `0.266838924%` / `0.427185 GiB` | PASS verifier-only ceiling |
| Same-machine proof-carrying execution fits target | DERIVED `100.058120260%` ops, `100.266838924%` traffic | REJECTED |
| Local sublinear trace source exists | no algorithm | NOT TESTED / REQUIRED SEPARATELY |

## EXP-082A preregistered Gate

| Claim | Evidence | Verdict |
|---|---:|---|
| Exact block-pattern distance never exceeds Hamming distance | no run | PREREGISTERED |
| Nearest-neighbor sum/2 lower-bounds exact MST | Phase A proof in contract | PREREGISTERED |
| Weighted best-orientation lower bound <=1.185185% | no model weight run | PREREGISTERED |
| Exact tree reconstructs source and direct integer MatVec | Stage 2 conditional | NOT AUTHORIZED YET |
| Fully charged operations/traffic/sidecar pass | Stage 2 conditional | NOT AUTHORIZED YET |
| BF16/Q4 output preservation, CUDA, 122B/405B | no execution | NOT TESTED |

Current classification: EXP-082A passed E0 novelty/upside triage only. No core
mechanism is promoted until the registered lower-bound Gate survives.

## EXP-082A authoritative validation

| Claim | Evidence | Verdict |
|---|---:|---|
| Block-pattern distance never exceeds coefficient Hamming distance | 72 exact controls, zero failures | PASS at E1 |
| Nearest-neighbor sum/2 lower-bounds exact terminal MST | proof plus exact Prim controls | PASS at E1 |
| Weighted favorable lower bound <= `1.185185185%` | `1.562367394%` over 21 matrices | FAIL, `1.318247489x` target |
| Result is isolated to one family or matrix | p50/p90 `1.562935965%/1.564025879%`; family range `1.559003194%`-`1.564025879%` | NO |
| Independent deterministic replay | decision/core and three tabular payloads byte-identical | PASS |
| Exact tree/runtime, BF16/Q4 outputs, CUDA, 122B/405B | stopped before Stage 2 | NOT TESTED |

Current classification: terminal-only exact row/column Hamming spanning trees
are rejected by a favorable certified coefficient-work lower bound. No core is
promoted; Phase D and E2-E7 remain not achieved.

## E0 synthetic-intermediate exact-circuit audit

| Claim | Evidence | Verdict |
|---|---:|---|
| Synthetic Hamming tree is outside static exact linear DAGs | exact recurrence/class inclusion proof | REJECTED |
| Static synthetic DAG is new relative to archived EXP-072B | EXP-072B permits shared synthetic coefficient forms | REJECTED |
| EXP-082A alone rejects arbitrary Steiner nodes | DERIVED `SMT >= MST/2`, only `0.781183697%` | NOT CERTIFIED |
| Random hypercube Steiner nodes approach the final fraction | published almost-all cost near `33.3%` of dense | ADVERSE E0 EVIDENCE |
| General static linear sharing universally fits target | published almost-all `Theta(n^2/log n)` binary complexity | NOT SUPPORTED; NOT A FINITE REAL-WEIGHT GATE |
| Cold placement proves operation/traffic closure | no selector or charged schedule | REJECTED |
| Query-Adaptive Cold Source exists and fits target | no algorithm/equation yet | OPEN E0 QUESTION |

Current classification: no static synthetic-intermediate Core Candidate is
promoted. No model or hardware was run. A query-adaptive cold-backed interface
remains logically open but has no surviving mechanism yet.

## E0 query-adaptive cold-backed equation

| Claim | Evidence | Verdict |
|---|---:|---|
| Fully charged branch equation includes common/hit/miss/build/fallback | symbolic derivation plus 9 calculator tests | PASS at E0 |
| Zero-cost minimum coverage at p50 fraction | `98.814814815%` | DERIVED |
| Minimum coverage after known verifier traffic | `99.081653739%` | DERIVED |
| EXP-081A coverage frontier is independently reproduced | `99.741472756%` | PASS cross-check |
| Zero-overhead useful information amplification | `84.375x` minimum | DERIVED |
| Amplification after known verifier | `108.891389x` minimum | DERIVED |
| Raw page omission alone determines exact arbitrary `W*x` | scoped indistinguishability argument | REJECTED |
| Coded causal information source exists | no algorithm | OPEN / NOT TESTED |
| Native-4B latency, SSD/H2D, peak VRAM, 405B execution | no hardware/model run | NOT TESTED |

Current classification: the accounting frontier is closed and reproducible,
but no Core Candidate survives. A Coded Causal Cold Source remains a search
class only; Phase D and E1-E7 remain unachieved for it.

## E0/E1-reference Causal Residual Atlas source

| Claim | Evidence | Verdict |
|---|---:|---|
| Prefix pairs causally construct `Q` and `Z=WQ` | exact algebra plus focused reference tests | PASS synthetic/reference |
| Unread page radius encloses exact residual | Frobenius/operator inequality; randomized no-false-bound tests | PASS synthetic/reference |
| Corrupt/non-finite capsule or bound fails closed | reference fault tests | PASS |
| All revealed pages recover dense result | exact reference comparison | PASS |
| Rank-16, requested-0.2%, 64-column logical traffic fits `8/675` | `1.085025716%` after minimum 64-token build amortization | PASS E0 arithmetic screen |
| Same screen fits logical operations | `0.557958575%` | PASS E0 arithmetic screen |
| Required certificate coverage exists on real causal traces | `99.899840530%` required; no real run | NOT TESTED / NEXT GATE |
| Complete 8 GiB state fits | capsule `0.980995178 GiB`; KV/work/buffers absent | NOT CLOSED |
| 1,009 cold pages meet native-4B latency | no physical I/O | NOT TESTED |
| Exact/native numerical enclosure across full Transformer | single-projection exact-real reference only | NOT TESTED |
| E2 operation replacement, 122B/405B, target hardware | no execution | NOT TESTED |

Current classification: a concrete Coded Causal Cold Source now exists, but it
is only an E0 information-source candidate with E1 synthetic controls. No
Surviving Candidate or positive milestone exists until the pinned real-weight
coverage Gate passes.

## Preregistered Causal Residual Atlas cheapest Gate

| Claim | Evidence | Verdict |
|---|---:|---|
| Gate population and exact page enumeration are frozen | 18 prompts, 36 branches, 1,296 candidates | PASS contract |
| Teacher position excludes the prompt position used to build the Atlas | fixed index 1, prompt-only basis | PASS contract |
| Target coverage permits any failure in the finite population | required 18/18; one failure is 94.444444% | NO |
| Exact-reference page selection is deterministic and fail-closed | 7 pure contract/oracle tests | PASS reference |
| Pinned checkpoint baseline and all-page identity | no Gate run | NOT TESTED |
| One-page oracle token coverage reaches 99.899840530% | no Gate run | NOT TESTED / NEXT |
| Every family succeeds 3/3 and all branches 36/36 | no Gate run | NOT TESTED / NEXT |
| Mean/p95 target-to-candidate KL <=0.02/0.05 | no Gate run | NOT TESTED / NEXT |
| Legal pair center and outward native bound certify the output | favorable center only | NOT TESTED |
| Actual operation replacement, physical speed, 122B/405B, E2-E7 | no execution | NOT TESTED |

Current classification: only the cheapest Gate is preregistered. There is no
real-weight result, Surviving Candidate, positive milestone, or experiment
number.

## EXP-083A authoritative validation

| Claim | Evidence | Verdict |
|---|---:|---|
| Registered payload, prompts, trace, Gate hash, rank/page/layer/position are unchanged | source commit plus input audit and checksums | PASS |
| Baseline follows the pinned EXP-076 trace | 0 mismatches | PASS |
| Dense/all-page and exact branch-point controls remain valid | 234/234 controls; 0 failures | PASS |
| Favorable one-page oracle preserves every token | 18/18 | PASS E1 favorable ceiling |
| Every family and projection branch succeeds | six families 3/3; 36/36 branches | PASS |
| Mean/p95 KL <=0.02/0.05 | 0.007225545/0.037660753 | PASS |
| Independent result verifier | checksum, selector, aggregation, core rebuild | PASS |
| Separate model replay reproduces scientific core | identical `ab96e611...f3954845` | PASS |
| A target-free minimum-radius page selector meets the Gate | exploratory 34/36 branches, 16/18 tokens, p95 0.075847 | FAIL post-hoc / non-authoritative |
| Legal pair center and outward native certificate | native-anchored oracle only | NOT TESTED / NEXT GATE |
| Simultaneous operation replacement and charged fallback | one projection patched; all else dense | NOT TESTED |
| Physical latency, 8 GiB GPU VRAM, 122B/405B, E2-E7 | no qualifying run | NOT TESTED |

Current classification: the registered favorable page-existence premise passes
and reproduces at E1 on the pinned 0.8B model. This is a material positive
necessary-condition result, not a Surviving Candidate, E2 replacement,
hardware result, or Fixed-Mission success.

## Preregistered legal pair/outward last-down Gate

| Claim | Evidence | Verdict |
|---|---:|---|
| Previously observed 18-prompt logits cannot tune the Gate | separate new prompt file, frozen SHA-256 | PASS contract |
| New scientific population is complete and balanced | 24 unique rows, six families at 4 each | PASS contract |
| Selector cannot read weights, native current output, or logits | function signature contains residual and page width only | PASS reference |
| Common spectral unread bound has no false underestimate | exact-dyadic proof plus randomized matrix/vector controls | PASS reference |
| Pair defect, unread, arithmetic, cast, and native terms are all charged | reference equation and fault tests | PASS reference |
| RMSNorm local ball and LM-head strict top-1 rule are sound | randomized ball and exhaustive-circle controls | PASS reference |
| Charged 405B component traffic/operations fit `8/675` | 1.093706272% / 0.928746620% at 20M static service tokens | PASS favorable E0 component equation |
| Component capsule plus proof metadata is below 1 GiB | 0.983090483 GiB | PASS component state only |
| Complete target peak including KV/work/fallback fits 8 GiB | 7.0169 GiB remainder excludes mandatory state | NOT CLOSED |
| Pair-only last-down certificate succeeds on real held-out rows | no model run | NOT TESTED / NEXT GATE |
| Required coverage | 99.908521087%; finite requirement 24/24 and 4/4/family | NOT TESTED / NEXT GATE |
| Backward layers, eight positions, simultaneous replacement, E2-E7 | deliberately outside first Gate | NOT TESTED |

Current classification: leakage-safe equations and ten focused controls are
preregistered. There is no result on the new prompt population, no Surviving
Candidate, and no E2 or hardware evidence.

## EXP-083B pre-execution implementation validation

| Claim | Evidence | Verdict |
|---|---:|---|
| Pair compiler has no weight parameter | signature test plus pair-defect randomized control | PASS source |
| Current page selector can receive only stored basis/current input/page width | signature tests | PASS source |
| Static operator proof is positive-definite rather than trusted SVD | outward Gram, verified Cholesky residual, verified inverse residual | PASS reference |
| Registered layer-23 matrix admits the proof | beta 1.3267520416; one attempt; positive margin 1.9485e-07 | PASS static preflight |
| Projection radius encloses BF16 candidate/native difference | randomized BF16 reference | PASS reference |
| Residual add and final RMSNorm radius encloses both paths | randomized BF16 reference | PASS reference |
| Gate rejects the first valid fallback and promotes only complete 24/24 | aggregation boundary tests | PASS reference |
| Focused/full repository regression | 17 / 453 tests | PASS |
| Independent raw-evidence replay | verifier implemented; no result bundle yet | NOT TESTED / EXECUTE NEXT |
| Untouched 24-prompt last-down certificate | no model prompt run | NOT TESTED / EXECUTE NEXT |
| E2 operation replacement, speed, 8 GiB, 122B/405B | outside this Gate | NOT TESTED |

Current classification: implementation and static metadata are ready, but the
scientific population remains untouched. There is still no EXP-083B decision,
Surviving Candidate, E2 result, or fixed-mission success.

## EXP-083B authoritative validation

| Claim | Evidence | Verdict |
|---|---:|---|
| Source/config/authority/prompt/checkpoint hashes are pinned | source `336d59b...`; config `2c8bdf...`; input audit | PASS |
| Static spectral claim is verified positive definite | beta 1.3267520416; positive margin 1.9485e-07 | PASS |
| Pair compiler reaches registered rank on first untouched row | rank 16; accepted prefix rows 0-15 | PASS |
| Selector is target-free and chooses registered page deterministically | maximum residual energy; page 0 | PASS |
| Every native/candidate arithmetic envelope holds | 19/19 controls; zero failures | PASS |
| Candidate certificate proves the native greedy winner | unresolved strict margin | FAIL SCIENTIFIC |
| Exact completion/fallback is correct | bitwise dense replay; one fallback | PASS CONTROL / FAIL GATE |
| Zero fallback required by 99.908521% coverage | observed 1/1 evaluated row fallback | FAIL |
| False accept/leakage/malformed state | 0 / 0 / 0 | PASS |
| Candidate/native winner and KL | 21461/21461; KL 0.05158216425 | MEASURED |
| Independent verifier | zero model forwards; ten checks; same deterministic core | PASS |
| Deterministic-core SHA-256 | `57e78fd4...3711ff4` | PASS |
| Same-certificate rescue by tighter RMS implementation term | actual hidden radius 38.9079 vs ideal row-margin limit 4.19783 | REJECTED NECESSARY CONDITION |
| Backward expansion and E2 Atlas integration | blocked by first-row failure | CLOSED |
| Physical latency, 8 GiB VRAM, 122B/405B, E2-E7 | no qualifying execution | NOT TESTED |

Authoritative decision:
`REJECT_CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_PATH`. The favorable page-
existence result remains true for EXP-083A, but its legal executable
generalization does not survive. Current classification is
`NO_SURVIVING_CANDIDATE`.

## E0 post-Atlas causal information-source audit

| Claim | Evidence | Verdict |
|---|---:|---|
| Primal/dual bilinear decomposition is exact | six deterministic reference/fault controls | PASS E0 |
| Cached `WQ` and `W^T P` determine every query | `Delta W=lambda r u^T` preserves both images but changes `r^T W u` | REJECTED |
| Forward-pair continuation is materially post-Atlas | no dual information; same omitted contribution | REJECTED AS ATLAS NORMAL FORM |
| One fresh full-model dual build fits at 64 service tokens | 1.5625% vs 1.185185185% final target, all other cost free | FAIL RESOURCE |
| Dynamic one-direction minimum lifetime | 85 tokens free-common; 109 with verifier; 999 atop Atlas | DERIVED |
| Static one-last-down vocabulary composite fits 8 GiB | favorable 12.720703125 GiB | FAIL STATE |
| Static one-last-down full scan fits target traffic | 6.765979992%; 5.708795618x target | FAIL TRAFFIC |
| Few exact directions plus row-norm screen resolves EXP-083B row | 248,319/248,319 competitors unresolved under actual and frozen radii | FAIL POST-HOC DIAGNOSTIC |
| Focused/full repository regression | 6 / 459 tests | PASS |
| Standard repository validation | `scripts/run_validation.py` completed successfully | PASS |
| All exact bilinear/MIPS structures are impossible | not established | OPEN / NO UNIVERSAL CLAIM |
| New model forward, experiment, hardware, or server action | none | NOT RUN / NOT AUTHORIZED |

Current classification: the two constructible Decision-Directional Source
variants are rejected at E0. The Bilinear Cross Residual interface remains a
research question without a construction, not a Surviving Candidate.

## E0 matrix-local separable cross-residual bound

| Claim | Evidence | Verdict |
|---|---:|---|
| Bilinear center/span/cross identity is exact | exhaustive GF(2) reference control | PASS E0 |
| Image-pair independent bit rank is `a*n+m*b-a*b` | formula plus independent small GF(2) ranks | PASS E0 |
| Sphere-cover witness is finite | `0.3+H_2(3/16)=0.9962122601<1` | PASS E0 |
| Complete 8 GiB binary side grant fits target cross work | 1.52104775688% vs 1.18518518519% | FAIL RESOURCE |
| Cross-probe lower bound | 6,141,198,336 cells; 1.28338404487x target | DERIVED |
| Hot point where this witness stops rejecting | 9.34710279225 GiB; not sufficient | ABOVE FIXED TOTAL |
| Focused reference/property tests | 8/8 | PASS |
| Focused/full repository regression | 8 / 467 tests | PASS |
| Standard repository validation | `scripts/run_validation.py` completed successfully | PASS |
| General nonlinear/cross-matrix/word-packed structure impossible | outside declared model | OPEN / NO CLAIM |
| Real Transformer reachability, E1/E2, model or hardware run | none | NOT TESTED |

Current classification: matrix-local separable linear covering-code upgrades
to Atlas are rejected at E0 under the arbitrary Cartesian coefficient-probe
contract. A nonseparable/global or causally restricted source remains an open
research interface, not a construction or Surviving Candidate.

## E0 cross-matrix global linear-advice localization

| Claim | Evidence | Verdict |
|---|---:|---|
| Outside global-advice content is free for a block-local query | exact identity `pi_-i(e)=pi_-i(a)` | REJECTED |
| Zero-outside useful spaces direct-sum | `sum dim(U intersect V_i)<=dim(U)` | PASS E0 |
| Fixed outside support can expose arbitrary local dimension | rank-nullity gives at most `dim(U_i)+abs(T)` | REJECTED |
| Projection dimensions may be summed as local state | exhaustive diagonal counterexample `U={(x,x)}` | REJECTED |
| Finite global covering-radius witness | `13/50`; hot rate + entropy `0.996950297239` | PASS E0 |
| Certified global covering radius | `104,974,453,310` coefficients | DERIVED |
| Registered rank-one tuple span | at most `16,384` aligned tuples | DERIVED |
| Total coefficient-use lower bound | `6,407,133`; `0.001586914271%` | DERIVED |
| Bound rejects p50 target | target is `746.8489x` the bound | NO / INSUFFICIENT |
| Ideal 64-bit/512-bit packing | `100,112` words / `12,514` lines; payload only | NOT PHYSICAL EVIDENCE |
| Focused exhaustive/property tests | 9/9 | PASS |
| Full repository regression | 476 tests | PASS |
| General nonlinear/adaptive structure impossible | outside declared model | OPEN / NO CLAIM |
| Real causal reachability, model, E1/E2, hardware | none | NOT TESTED |

Current classification: free cross-matrix projection reuse is closed only in
the systematic linear coefficient-use model. The finite general bound is too
weak for mission rejection, and no globally mixed construction survives E0.
The next Gate may address only a leakage-free causal query restriction.

## E0 causal bilinear query-restriction threshold

| Claim | Evidence | Verdict |
|---|---:|---|
| Rank `R` forces at least `R-B` misses for any best `B`-span | proof plus exhaustive GF(2) populations | PASS E0 |
| Aligned residual-pair coordinates over 883 matrices | `39,109,888` | DERIVED |
| Full factor scan with span 23 fits p50 traffic | `1.159413233%` including build/verifier/metadata | PASS favorable component |
| Span 24 fits p50 traffic | above `1.185185185%` | FAIL resource |
| Span-23 component state fits 8 GiB | `2.104879502 GiB` | PASS component only |
| Logical address counts are physical latency | 40,618 segments / 442,129 rounded pages | NOT MEASURED |
| 36-row last-down fallback allowance | 4 rows | DERIVED |
| Certified rank rejecting every `B<=23` ledger | 28 | DERIVED / PREREGISTERED |
| Modular nonincrease proves exact rational hit | explicitly forbidden | NO |
| Six-check field implementation union bound | `<1.43e-17` over 17.66B queries | PASS probabilistic equation |
| Real causal query rank/coverage | no model row | NOT TESTED / NEXT GATE |
| Paid general pair extractor and local result source | known raw VJP is dense; granted free | NOT CLOSED |
| Native BF16/Q4 operation replacement, hardware, E2-E7 | none | NOT TESTED |

Current classification: the cheapest causal-population statistic is now
preregistered, but no query restriction has been measured and no Core Candidate
survives. Only the frozen last-down rank Gate may execute next.

## EXP-084A authoritative causal bilinear rank validation

| Claim | Evidence | Verdict |
|---|---:|---|
| Frozen implementation/config/checkpoint/prompt/trace provenance | protected `6214700...`; config/checkpoint/input SHA match | PASS |
| Batched prompt capture is safe | attempt 01 position-0 activation/direction mismatch; zero query rows | FAIL CLOSED / QUARANTINED |
| Sequential prompt basis excludes current decode state | single-token committed-prefix path; 9 prompt records | PASS CONTROL |
| Prompt primal/dual side rank reaches 16 | every executed prompt | PASS |
| Registered build population | 6 prompts x 4 positions = 24 | PASS |
| Build outer-product rank | 24 under each of 65521/65519/65497 | MEASURED E1 |
| Dimension-23 exact ledger construction | first 23 rationally independent rows, frozen pivots | PASS |
| First five held-out rows hit build ledger | 0/5 exact rational hits | FAIL |
| First five held-out fixed-ledger modular ranks | 24 under all three primes per row | MISS CERTIFICATES |
| Held-out-only rank at registered stop | 5 under all three primes | MEASURED; NOT 28 |
| Four-fallback allowance | fifth exact miss at row 5 | FAIL / EARLY REJECTION |
| Control and leakage population | 114/114; target-future reads 0 | PASS |
| Independent verifier | 0 model forwards; 24 build + 5 eval rebuilt | PASS |
| Deterministic-core SHA-256 | `1e79550f...0a7c4` | PASS |
| Full 36-row rank or best post-hoc B=23 rejection | stopped after fifth miss | NOT ESTABLISHED |
| Paid pair extractor/native reconstruction | deliberately free/unsolved | NOT TESTED |
| Physical latency, 8 GiB VRAM, 122B/405B, E2-E7 | no qualifying execution | NOT TESTED |

Authoritative decision:
`REJECT_CAUSAL_BILINEAR_FACTOR_SPAN_LEDGER_AS_CORE`. This closes the frozen
automatic full-factor ledger, not every nonlinear or implicit exact query
code. Current classification remains `NO_SURVIVING_CANDIDATE`.

## E0 query-adaptive exact code-union bound

| Claim | Evidence | Verdict |
|---|---:|---|
| A perfect router creates new cached answer directions | union-information theorem | REJECTED |
| Independent hits across leaves | at most sum of leaf dimensions | PASS E0 |
| Maximum registered independent directions | 87,958 at leaf dimension 1 | DERIVED |
| Maximum independent coverage over 20M | `0.43979%` | INSUFFICIENT |
| Required full-fallback coverage at that ceiling | `99.999992826%` | DERIVED |
| Independent-stream traffic at `A_max` | `1.0074538801` dense; `85.0039x` target | FAIL E0 |
| Online one-pass independent stream | at least `1 + q`; `84.6328x` target | FAIL E0 |
| Full 24-row EXP-084A build ranks | 24 under 65521/65519/65497 | PASS REPLAY |
| Five eval rows inside full build span | every insertion raises rank to 25 | 0/5 / REJECTED |
| New model forwards | zero; frozen arrays only | NOT AN E1 RUN |
| Focused reference tests | 9/9 | PASS |
| Full repository regression | 508 tests | PASS |
| Standard validation runner | completed | PASS |
| Independent output-directory reproduction | identical summary SHA-256 | PASS |
| General implicit nonlinear data structure impossible | outside declared model | OPEN / NO CLAIM |
| Paid extractor, E2 replacement, physical VRAM/latency, 122B/405B | none | NOT TESTED |

Authoritative decision:
`REJECT_TRACE_BUILT_QUERY_ADAPTIVE_LINEAR_CODE_UNION_AS_CORE`. Restricted
repeated-query use remains auxiliary. Current classification remains
`NO_SURVIVING_CANDIDATE`.

## E0 exact-field nonlinear bilinear path-collapse audit

| Claim | Evidence | Verdict |
|---|---:|---|
| A fixed open-cell algebraic path computes the same rational function | rational identity argument | PASS E0 |
| `gradient_r(r^T W u)=W u` | exact algebra | PASS E0 |
| Static derivative-circuit overhead | Baur--Strassen all-unit-cost factor `<=4` | PASS E0 |
| Target scalar query implies static MatVec fraction `<=32/675` | exact fraction calculation | DERIVED |
| This proves finite target impossibility | reduction is containment only | NO / NOT CLAIMED |
| Twin-width general algorithm is new | Hamming predecessor traversal | REJECTED / F-049 |
| Rectangle/grammar evaluation is new | static exact schedule | REJECTED / F-050 |
| Exact rational nonlinear cancellation controls | 64/64 values; 64/64 gradients | PASS |
| Focused standard-library tests | 5/5 | PASS |
| Independent output-directory reproduction | identical summary SHA-256 | PASS |
| Full repository pytest regression | sandbox dependency unavailable in this session | NOT RUN |
| Standard validation runner | dependency ACL failure before validation work | NOT RUN |
| New model forwards or checkpoint mutation | zero | NOT RUN |
| Finite-word/cell-probe impossibility | outside theorem | OPEN / NO CLAIM |
| Core Candidate, E1/E2, hardware, 122B/405B | none | NOT TESTED |

Authoritative decision:
`REJECT_EXACT_FIELD_NONLINEAR_BILINEAR_ARITHMETIC_AS_DISTINCT_CORE_CLASS`.
Current classification remains `NO_SURVIVING_CANDIDATE`.

## E0 finite-word discontinuous rank-one source audit

| Claim | Evidence | Verdict |
|---|---:|---|
| Block truth-table equation is exact over GF(2) | 296/296 exhaustive queries | PASS |
| Basis queries recover stored matrix bits | 57/57 | PASS |
| User 2.5% query-traffic layout | 64-bit 25-by-26; 2.461538% | QUERY AXIS PASS |
| 2.5% layout table storage | 8.660768514174031e11 Q4 checkpoints | FAIL E0 |
| 2.5% layout address width | 78 bits on a 64-bit model | FAIL E0 |
| 2.5% layout build amortization | 173,215 dense/token over 20M | FAIL E0 |
| Registered 8/675 query-traffic layout | 64-bit 37-by-37; 1.168736% | QUERY AXIS PASS |
| Registered layout table storage | 3.449500717947148e18 Q4 checkpoints | FAIL E0 |
| Mailman favorable operation floor | 1/9 | FAIL E0 |
| Broadword coefficient payload | 100% read | FAIL AS SOURCE |
| Boolean cell-probe output | rectangle nonemptiness only | NOT NUMERICAL SOURCE |
| General nonlinear rank-one cell probes impossible | no covering theorem | OPEN / NO CLAIM |
| Focused standard-library tests | 9/9 | PASS |
| Independent output-directory reproduction | identical summary SHA-256 | PASS |
| Canonical summary SHA-256 | 2f785b0f...5b9007 | PASS |
| Pytest excluding 11 unavailable Torch files | 464 pass; 1 existing subprocess import failure | PARTIAL / NOT FULL PASS |
| Standard validation runner | Torch unavailable at initial import | NOT RUN |
| New model forwards or hardware actions | zero | NOT RUN |
| E1/E2, 8 GiB physical runtime, 122B/405B | no surviving core | NOT TESTED |

Authoritative decision:
NO_FINITE_WORD_CONSTRUCTOR_SURVIVES_E0_KEEP_GENERAL_RANK_ONE_CELL_PROBE_OPEN.
Current classification remains NO_SURVIVING_CANDIDATE.

## E0 global nonlinear rank-one frontier audit

| Claim | Evidence | Verdict |
|---|---:|---|
| Larsen--Williams is globally nonlocal/nonlinear | checkpoint-derived all-zero rectangle list | PASS CLASSIFICATION |
| Published output is exact numerical rank-one | Boolean-semiring nonemptiness bit | FAIL SEMANTICS |
| Leading `n=16,384,w=64` probe bytes | 262,144 words = 2 MiB = 6.25% | NOT 2.5%; BIG-O ONLY |
| Leading redundancy | 16,777,216 bits = 2 MiB = 6.25% | BIG-O ONLY |
| Published 32-query amortization | none | NOT ESTABLISHED |
| Exact summary of every subrectangle compresses arbitrary block | singleton injection; raw information required | REJECTED DIRECT LIFT |
| Full rank-one query answers are 3-wise independent | three-query XOR identity | FALSE |
| KPI25 64-bit minimum-even-time premise | max `k=2`; required `k>=130` | INAPPLICABLE |
| Ideal `n^2` whole-MatVec bound reaches scalar `n^2/40` | scalarizes to `n`; 409.6x short | FAIL TARGET BOUND |
| 2026 dynamic Multiphase result is static target theorem | update model; polylog bound | INAPPLICABLE |
| Full-sweep batching denominator | 40 outputs are 32x over `1/1280`; block `rho=1` | REJECTED AS TARGET |
| Focused deterministic tests | 7/7 | PASS |
| Related frontier regression | 22/22 | PASS |
| Full repository pytest regression | 535/535 | PASS |
| Standard validation runner | completed | PASS |
| General nonlinear numerical constructor | none | OPEN |
| General nonlinear numerical impossibility | no covering theorem | OPEN / NO CLAIM |
| Model forwards and hardware actions | 0 / 0 | NOT RUN |

Authoritative decision:
`REJECT_BOOLEAN_ZERO_RECTANGLE_AND_LIMITED_INDEPENDENCE_LIFTS_KEEP_GENERAL_NONLINEAR_NUMERICAL_RANK_ONE_GAP_OPEN`.
Current classification remains `NO_SURVIVING_CANDIDATE`.

## E0 finite-semiring preprocessing frontier audit

| Claim | Evidence | Verdict |
|---|---:|---|
| Williams graph is a valid finite-semiring preprocessed MatVec | primary theorem and construction | PASS IN PUBLISHED MODEL |
| Minimum selected neighbor value | `b*log2(K)` bits | DERIVED FAVORABLE |
| Ideal direct query/semantic-raw ratio | `1/b` | DERIVED |
| Ideal persistent sidecar/semantic-raw ratio | `K^b/b` | DERIVED |
| Registered theorem-parameter block | `n=16,384`, `b<=14` | DERIVED |
| One explicit Boolean square edge payload | `36.616085 GiB` | FAIL 8 GiB |
| Model-wide one-bit sidecar | `55,292.57 GiB`; `6,911.57x` 8 GiB | FAIL E0 |
| Model-wide Q4 one-query payload | `2.629552%` DFloat | FAIL 2.5% BLOCK |
| Model-wide BF16 one-query payload | `10.518207%` DFloat | FAIL 2.5% BLOCK |
| Native BF16 addition associative | exact three-term witness | FALSE |
| Native FP32 addition associative | exact three-term witness | FALSE |
| Published output is scalar `r^T W u` | full MatVec | NO / OVERCOMPUTES |
| Published 32-query shared-probe theorem | none | NOT ESTABLISHED |
| Direct no-reuse one-bit 32-query payload | `21.036415%` DFloat; `8.414566x` budget | FAIL CONSTRUCTOR PATH |
| Focused deterministic tests | 9/9 | PASS |
| Related finite-word/nonlinear regression | 36/36 | PASS |
| Full repository regression with repository import path | 544/544 | PASS |
| Standard validation runner | completed | PASS |
| Independent output-directory reproduction | identical summary SHA-256 | PASS |
| Canonical summary SHA-256 | `8188c0fd...3e09f1` | PASS |
| General nonlinear numerical constructor | none | OPEN |
| General nonlinear numerical impossibility | no covering theorem | OPEN / NO CLAIM |
| Model forwards and hardware actions | 0 / 0 | NOT RUN |

Authoritative decision:
`REJECT_WILLIAMS_FINITE_SEMIRING_GRAPH_AS_REFERENCE_EXACT_2_5_PERCENT_CORE`.
Current classification remains `NO_SURVIVING_CANDIDATE`.

## E0 global-advice synergy frontier audit

| Claim | Evidence | Verdict |
|---|---:|---|
| Global advice may be divided by matrix count | exact XOR conditional-synergy witness | FALSE |
| Exhaustive three-word XOR recovery | 512 cases; 1,536 target checks | PASS |
| Advice entropy versus conditional-information sum | 3 bits versus 9 bits | 3x SYNERGY |
| Registered complete hidden squares | 11/layer; 126 layers; 1,386 total | DERIVED |
| Invalid average advice per square | `128/693 = 18.470418%` | DIAGNOSTIC ONLY |
| Average lies in displayed `n^2/64` proof regime | above `4,194,304` bits | FALSE |
| Hidden-constant-one illegal tile sum reaches target | 8,781,696 probes; 12,553.84x short | FALSE / INVALID SUM |
| Separate-tile finite injectivity floor | one independently chosen hard probe | PROVED SCOPED |
| Invalid 6-by-6 floor sum reaches target | 10.173088%; 9.829856x short | FALSE / INVALID SUM |
| Published model-wide direct-sum theorem | none | NOT ESTABLISHED |
| Focused deterministic tests | 8/8 | PASS |
| Related frontier regression | 44/44 | PASS |
| Full repository regression | 552/552 | PASS |
| Standard validation runner | completed | PASS |
| Independent output-directory reproduction | identical summary SHA-256 | PASS |
| Canonical summary SHA-256 | `0b926e70...97feda` | PASS |
| General nonlinear constructor or impossibility | neither delivered | OPEN / NO CLAIM |
| Model forwards and hardware actions | 0 / 0 | NOT RUN |

Authoritative decision:
`REJECT_NAIVE_GLOBAL_ADVICE_DIVISION_AND_CKL_TILE_SUM_AS_TARGET_BOUND`.
Current classification remains `NO_SURVIVING_CANDIDATE`.

## E0 Fourier-fiber direct-sum frontier audit

| Claim | Evidence | Verdict |
|---|---:|---|
| Arbitrary nonlinear global advice is covered | fixed advice-fiber proof | PASS |
| Adaptive cross-block probes are jointly charged | one depth-`t` decision tree | PASS |
| Full Walsh characters span every advice fiber | restricted row orthogonality | PASS |
| Finite all-linear inequality | `2^(D-r)<=sum(j<=t,C(D,j))` | PROVED |
| Registered strict entropy witness | `delta=131/500`; margin `0.000927904...` | PASS |
| All-linear raw-bit floor | 106,332,501,836; 26.2% | DERIVED |
| Floor exceeds registered p50 coefficient budget | 22.10625x | PASS SCOPED |
| Floor exceeds user DFloat `1/40` bits | 0.96452x | NO |
| One registered rank-one tuple query log | 39,254,528 bits | DERIVED FAVORABLE |
| 32 rank-one tuples query log | 1,256,144,896 bits | DERIVED FAVORABLE |
| 32-tuple character-dimension method ceiling | 114,194,991 probes; 0.028137293% | FAR BELOW TARGET |
| Method ceiling is an algorithm | explicitly false | NO |
| Exhaustive small-fiber controls | 255 fibers; 2,040 query checks; 0 violations | PASS |
| Walsh orthogonality controls | 120 row pairs; 0 failures | PASS |
| Focused deterministic tests | 8/8 | PASS |
| Related frontier regression | 52/52 | PASS |
| Full repository regression | 560/560 | PASS |
| Standard validation runner | completed | PASS |
| Independent output-directory reproduction | identical summary SHA-256 | PASS |
| Canonical summary SHA-256 | `004f37f8...a6ca3` | PASS |
| General nonlinear rank-one target bound | restricted-fiber rank absent | OPEN |
| Model forwards and hardware actions | 0 / 0 | NOT RUN |

Authoritative decision:
`KEEP_FOURIER_FIBER_BOUND_FOR_ALL_LINEAR_TUPLES_REJECT_AS_GENERAL_RANK_ONE_TARGET_RESOLUTION`.
Current classification remains `NO_SURVIVING_CANDIDATE`.

## E0 native-exact shortcut frontier

| Claim | Evidence | Verdict |
|---|---:|---|
| Old 2.5% premise used | explicit result flag | NO |
| Registered compute-only allowed fraction | `1.18270517%` | DERIVED |
| Favorable rounding absorption upper | max `1.25558036%` | REJECT |
| Consecutive coordinate reuse | max `0.33482143%` | REJECT |
| Unlimited same-coordinate history reuse | max `2.98549107%` | REJECT |
| Natural-run saving upper | max `0.16741071%` | REJECT |
| Exact duplicate-product reuse | max `2.98549107%` | REJECT |
| Perfect opposite-product cancellation | max `5.69196429%` | REJECT |
| BF16 unique values per column | min / p50 `653 / 694.5` | POLLARD T GATE FAIL |
| BF16 unique values per row | min / p50 `1276 / 1331` | TRANSPOSE T GATE FAIL |
| Ideal Pollard `d=1` best ratio | `35.60267857%` | REJECT |
| ΩROUNDLOCK down-projection oracle lock | `1/1,024 = 0.09765625%` | REJECT |
| ΩROUNDLOCK post-residual oracle lock | `7/1,024 = 0.68359375%` | REJECT |
| ΩROUNDLOCK post-RMSNorm oracle lock | `2/1,024 = 0.1953125%` | REJECT |
| ΩBACKCUT candidate/native winner | `21,461 / 21,461` | FIRST GATE PASS |
| ΩBACKCUT exact information source | existing decision-dual `r^T W u` | DUPLICATE / REJECT |
| Focused deterministic tests | 7/7 | PASS |
| Full repository regression | 567/567 | PASS |
| Standard validation runner | completed | PASS |
| Independent output-directory reproduction | identical summary SHA-256 | PASS |
| Canonical summary SHA-256 | `fa8a2d0d...5eeb09be` | PASS |
| New model forwards / hardware actions | 0 / 0 | NOT RUN |
| General exact constructor or impossibility | neither delivered | OPEN / NO CLAIM |

Authoritative decision:
`REJECT_SCREENED_SHORTCUTS_KEEP_SEARCHING_NEW_INFORMATION_SOURCE`.
Current classification remains `NO_SURVIVING_CANDIDATE`.

## E0 Spiky / entrywise-power rank-one frontier

| Check | Observed | Status |
|---|---:|---|
| Old 2.5% premise used | false | PASS |
| Registered non-embedding coefficients / matrices | 403,747,897,344 / 883 | PASS |
| Global row / column coordinate universes | 19,997,952 / 19,111,936 | PASS |
| Complete model-wide factor-incidence grant | 4,800,000,000 | PASS |
| Model-wide representable log2 upper | 184,958,474,578.0694 | PASS |
| All sign checkpoints log2 | 403,747,897,344 | PASS |
| Hard-checkpoint exponent bits | 218,789,422,765.9306 | REJECT DIRECT EVALUATOR |
| Cross-matrix components / invalid cells | granted / ignored | FAVORABLE OVERCOUNT |
| Single-square hard-instance exponent | 178,629,392.76025754 | DIAGNOSTIC |
| Paper dense-factor diagnostic | 1.1904761905% / 1.1827051732% | REJECT |
| PowerFold expanded-term cap | 96 | PASS |
| `p=2/3/4/5` maximum root ranks | 13 / 7 / 5 / 4 | REJECT AS LOW-RANK NORMAL FORM |
| Exact small spiky query equation | reference match | PASS |
| Focused deterministic tests | 7/7 | PASS |
| Related constructor-frontier tests | 14/14 | PASS |
| Full repository regression | 574/574 | PASS |
| Standard validation runner | completed | PASS |
| Independent output-directory reproduction | identical summary SHA-256 | PASS |
| Canonical summary SHA-256 | `89a49e42...4c6299c` | PASS |
| Model forwards / hardware actions | 0 / 0 | NOT RUN |
| General nonlinear probe gap | unresolved | OPEN / NO CLAIM |

Authoritative decision:
`REJECT_SPIKY_AND_POWER_FOLDS_KEEP_GENERAL_PROBE_GAP_OPEN`.
Full execution provenance is recorded in `REPRODUCIBILITY.md`.

## E0 adaptive codebook / trapdoor frontier

| Check | Observed | Status |
|---|---:|---|
| Old 2.5% premise used | false | PASS |
| Registered square / repair budget | 16,384 / 3,181,457 cells | PASS |
| Packing removal radius | 388 query-side bits | PASS |
| Minimum literal table log2 bits | 13,741.254846862746 | REJECT TABLE |
| Complete 8 GiB log2 bit capacity | 36 | PASS |
| Storage exponent deficit | 13,705.254846862746 | REJECT TABLE |
| Minimum literal address width | 13,742 bits | REJECT WORD MACHINE |
| TrapShift identity | exact finite-field reference | PASS IDENTITY |
| Shifted arbitrary product source | absent; dense fraction 1.0 | REJECT SOURCE |
| Focused deterministic tests | 8/8 | PASS |
| Related constructor-frontier tests | 15/15 | PASS |
| Full repository regression | 582/582 | PASS |
| Standard validation runner | completed | PASS |
| Independent output-directory reproduction | identical summary SHA-256 | PASS |
| Canonical summary SHA-256 | `5bd9f7bc...1472325b` | PASS |
| Model forwards / hardware actions | 0 / 0 | NOT RUN |
| General compressed adaptive decoder | unresolved | OPEN / NO CLAIM |

Authoritative decision:
`REJECT_NEARESTPAIR_LITERAL_TABLE_AND_SOURCE_FREE_TRAPSHIFT`.
Full execution provenance is recorded in `REPRODUCIBILITY.md`.

## E0 average-oracle amplifier frontier

| Check | Observed | Status |
|---|---:|---|
| Old 2.5% premise used | false | PASS |
| Binary coefficients / output rows | 403,747,897,344 / 19,997,952 | PASS |
| Complete hot state fraction | 17.020392474626% | FAVORABLE GRANT |
| Publication metric | average coordinate distance | CORRECTED |
| Common advantage required by paper | false | PASS |
| Exact-row concentration GF(2) accuracy | 58.510196237313% | COUNTEREXAMPLE |
| Concentration near-linear query time | false | DOES NOT MEET FULL PREMISE |
| Common-row advantage ceiling log10 | -2,521.899341736204 | SCOPED ONLY |
| Direct rank-covering payload | 47.00244140625 GiB | REJECT ROW LOTTERY |
| Direct payload / 8 GiB | 5.87530517578125x | REJECT ROW LOTTERY |
| Exhaustive Walsh/self-correction controls | passed | PASS |
| Focused deterministic tests | 11/11 | PASS |
| Related frontier regression | 71/71 | PASS |
| Full repository regression | 593/593 | PASS |
| Standard validation runner | completed | PASS |
| Independent output-directory reproduction | byte-identical summary | PASS |
| Canonical summary SHA-256 | `4f955eb8...9a14e514` | PASS |
| Model forwards / hardware actions | 0 / 0 | NOT RUN |
| Cold-backed adaptive oracle | unspecified | OPEN / NO CLAIM |

Authoritative decision:
`REJECT_ROW_LOTTERY_AND_UNCHARGED_AMPLIFIER_AS_CORE`.

## E0 functional-array-code novelty Gate

| Check | Observed | Status |
|---|---:|---|
| Enumerated toy ambient dimensions | 4 and 6 bits | PASS |
| `2x3`, one / two hot forms | 3 / 2 raw reads | PASS |
| `3x2`, one / three hot forms | 4 / 2 raw reads | PASS |
| Witness family | row parity / linear covering | DUPLICATE |
| `GF(16)` all-linear radius root | 79.09389479630005% | DIAGNOSTIC ONLY |
| Rank-one/native-float promotion | not made | PASS |
| Full repository regression | 593/593 | PASS |
| Standard validation runner | completed | PASS |
| Model forwards / hardware actions | 0 / 0 | NOT RUN |
| General nonlinear probe gap | unresolved | OPEN / NO CLAIM |

Authoritative decision: `REJECT_OMEGA_FUNCTIONALSPAN_AT_NOVELTY_GATE`.

## E0 nonlinear-fiber decision-depth screen

| Check | Observed | Status |
|---|---:|---|
| Complete nonempty `2x2` fibers | 65,535 | PASS |
| Distinct `2x2` rank-one masks | 10 | PASS |
| Size 1 / 2--4 / 5--8 / 9--16 minimax depth | 0 / 1 / 2 / 4 | PASS |
| All-linear comparison | identical minima | PASS |
| Distinct `2x3` affine fibers | 26,387 | PASS |
| `2x3` affine depth profile | 0,1,1,2,2,3,6 | PASS |
| Large nonlinear/global-advice promotion | not made | PASS |
| Full repository regression | 593/593 | PASS |
| Standard validation runner | completed | PASS |
| Model forwards / hardware actions | 0 / 0 | NOT RUN |
| General nonlinear probe gap | unresolved | OPEN / NO CLAIM |

Authoritative decision:
`REJECT_OMEGA_FIBERDT_AS_AN_IMMEDIATE_SMALL_NONLINEAR_CONSTRUCTOR`.

## E0 lossless, exact-cut, and gauge frontier

| Check | Observed | Status |
|---|---:|---|
| Exact three-cut rectangle identity | algebraically valid | PASS / RELABELING |
| Implemented exact cut-value source | none | REJECT |
| Favorable BF16 entropy rate | 10.6 bits/parameter | GRANTED |
| 32 GB/s lossless sweep | 16.8046952448 s | PASS |
| Zero-compute tokens needed for 20 ms | 841 | REJECT STANDALONE |
| I/O/token at 32-token block | 525.1467264 ms | REJECT |
| Free deletion of all attention weights | 17.709431388354932% | REJECT |
| Remaining dense parameters | 82.29056861164507% | REJECT |
| Fused lossless execution | useful | AUXILIARY |
| Focused frontier test | 1/1 | PASS |
| Full repository regression | 594/594 | PASS |
| Standard validation runner | completed | PASS |
| Model forwards / hardware actions | 0 / 0 | NOT RUN |
| General nonlinear probe gap | unresolved | OPEN / NO CLAIM |

Authoritative decisions:
`REJECT_OMEGA_CUTSUM_AS_EXACT_BILINEAR_RELABELING`,
`REJECT_OMEGA_ZIPWAVE_AS_STANDALONE_CORE`, and
`REJECT_ATTENTION_ONLY_GAUGE_FUSION_AT_FAVORABLE_CEILING`.

## E0 biorthogonal decomposition cancellation Gate

| Check | Observed | Status |
|---|---:|---|
| Rank-4 anti-flag graph | 120 vertices, degree 28, least eigenvalue -8 | PASS |
| `30x40`, rank 22 expected pair intersections | 3.5230839509486778... | DERIVED |
| `30x40` guaranteed intersections / cancellation | 4 / 4 | PASS |
| `30x40` forced radius / capacity ratio | 304 / 0.05068730227558649... | REJECT |
| `31x39` capacity ratio | 0.5723204167945867... | REJECT |
| `30x44` capacity ratio | 0.18141146098814276... | REJECT |
| Combined first unclosed shape | `31x43`, `S=1559`, `t=15` | OPEN ONLY |
| Focused tests | 7/7 | PASS |
| Full repository regression | 671/671 | PASS |
| Standard validation runner | completed | PASS |
| Model forwards / hardware actions | 0 / 0 | NOT RUN |
| Adaptive word decoder and native lift | uncovered | OPEN / NO CLAIM |

Authoritative decision:
`REJECT_30x40_31x39_AND_30x44_BY_BIORTHOGONAL_SUPPORT_CANCELLATION`.

## E0 recursive biorthogonal cancellation Gate

| Check | Observed | Status |
|---|---:|---|
| Positive internal-edge bound | overlapping adjacent pair exists | PASS |
| Pair peel recurrence | `Delta_r >= 2+Delta_(r-2)` | PASS |
| `31x43` cancellation at ranks 18 / 20 / 22 | 2 / 4 / 6 | PASS |
| `31x43` rank-22 radius / capacity ratio | 324 / 0.4293893830152846... | REJECT |
| `32x42` fixed-linear case | capacity contradiction | REJECT |
| Combined first unclosed shape | `31x42`, `S=1523`, `t=15` | OPEN ONLY |
| First rejected capacity-ordered shapes | 856 | PASS |
| Focused tests | 5/5 | PASS |
| Full repository pytest regression | 676/676 | PASS |
| Standard validation runner | completed | PASS |
| Model forwards / hardware actions | 0 / 0 | NOT RUN |
| Adaptive word decoder and native lift | uncovered | OPEN / NO CLAIM |

Authoritative decision:
`REJECT_31x43_AND_32x42_BY_RECURSIVE_BIORTHOGONAL_PAIR_PEELING`.

## E0 extension-field adaptive-support Gate

| Check | Observed | Status |
|---|---:|---|
| Zero-transcript adaptive path | common kernel forces sparse field span | PASS |
| Exhaustive `GF(4)^3` intersection maxima for 0 / 1 / 2 generators | 1 / 2 / 4 | PASS |
| `k=16,384` proportional cells | 19,172 | DERIVED |
| Logical / four-lane targets | 194 / 776 probes | REJECT |
| Exact first count-feasible radius | 3,421 probes | PASS |
| All DFloat11 bits + 8 GiB aggregate relaxed minimum | 10.988948% | REJECT |
| Favorable one-plane four-lane traffic allowance | 4.740741% | REJECT |
| Required storage at four-lane target | 19,515.208649x / 895.764 TiB | REJECT |
| Focused tests | 5/5 | PASS |
| Full unittest regression | 172/172 | PASS |
| Full repository pytest regression | 681/681 | PASS |
| Standard validation runner | completed | PASS |
| Model forwards / hardware actions | 0 / 0 | NOT RUN |
| Nonlinear cells / cross-block mixing / native lift | uncovered | OPEN / NO CLAIM |

Authoritative decision:
`REJECT_BLOCK_LOCAL_EXTENSION_FIELD_LINEAR_CELLS_EVEN_WITH_ADAPTIVE_ADDRESSES`.

## E0 adaptive packed-linear word Gate

| Check | Observed | Status |
|---|---:|---|
| Decoder address dependence | query and prior word values | GRANTED |
| Per-word contents | 64 independent binary linear forms | GRANTED |
| `31x42` minimum non-rejected reads | 8 words | REJECT TARGET |
| `31x42` favorable traffic | 64/651 = 9.831029% | REJECT |
| Rectangles checked through side 128 | 8,256 | PASS |
| Rectangles meeting `8/675` target | 0 | REJECT |
| Best scanned favorable traffic | 11/472 = 2.330508% | REJECT |
| Focused tests | 4/4 | PASS |
| Global cross-matrix/nonlinear words | uncovered | OPEN / NO CLAIM |

Authoritative decision:
`REJECT_BLOCK_LOCAL_ADAPTIVE_PACKED_LINEAR_WORDS`.

## E0 adaptive nonlinear probe-degree Gate

| Check | Observed | Status |
|---|---:|---|
| Stored cells | arbitrary nonlinear checkpoint functions | GRANTED |
| Decoder | deterministic adaptive addresses and arbitrary exact postprocessing | GRANTED |
| Necessary inequality | `RankLeq(a,b,r) <= V_A(S,min(S,r*t))` | DERIVED |
| `31x42` arbitrary nonlinear bit-cell minimum | 15 probes | PASS |
| `31x42` arbitrary nonlinear 64-bit-word minimum | 2 words | CAPACITY ONLY |
| Smallest target-feasible scan point | `25x108`, 50 words, 2 reads | UNCONSTRUCTED |
| Target-feasible rectangles through side 128 | 4,257 | CAPACITY ONLY |
| Systematic `2x3` plus one arbitrary advice bit | no 2-probe scheme | REJECT SUBCLASS |
| Fully nonsystematic `2x3`, 7-bit SMT | 600 s timeout / unknown | INCONCLUSIVE |
| Focused deterministic tests | 9/9 | PASS |
| Full unittest regression | 185/185 | PASS |
| Full repository pytest regression | 694/694 | PASS |
| Standard validation runner | completed | PASS |
| Model forwards / hardware actions | 0 / 0 | NOT RUN |
| Global cross-matrix encoding/native lift | uncovered | OPEN / NO CLAIM |

Authoritative decision:
`KEEP_NONLINEAR_CAPACITY_SURVIVORS_OPEN_BUT_UNCONSTRUCTED_AND_PIVOT_TO_GLOBAL_SHARED_READS`.

<!-- FIXED_PUBLIC_DYNAMIC_EXECUTOR_RESULT:3b9534800671e17eaef74869cf792186e683d6de -->
## Fixed-public dynamic executor validation row

- G1 ACTUAL RUNTIME: PASS on DEV-W; official loader + forward executed.
- G2 EXACT TRANSITION: True.
- G3 REAL BOUNDARY: True.
- G4 PHYSICAL SAVING: False.
- G5 FULL-MODEL PATH: formula recorded; TARGET-W 4B-class budget not established.
- G6 8GiB LEDGER: TARGET-W not established.
- G7 EXISTING ISA: PASS for implemented software primitives; no hypothetical opcode.
- G8 REPRODUCIBILITY: raw hosted result committed by workflow; workflow artifact retained.

## Fixed-public dynamic executor hosted result — commit `3b9534800671e17eaef74869cf792186e683d6de` / run `31982198413`

- Verdict: `REAL_CHECKPOINT_MECHANISM_CLASS_REJECTED_AT_G4`
- DEV-W: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; resolved SHA `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; official `LlamaForCausalLM.from_pretrained()` executed.
- TARGET-W: `meta-llama/Meta-Llama-3.1-405B-Instruct@f9801cba95a53242b3cc928a4a418d12571d1c5f`; metadata access=True; full TARGET-W weights/performance remain unevaluated unless the gated tensors were actually accessible.
- Official reference forward: executed=True.
- Frozen workload SHA-256: `f52dad873703170179b08bf2c48637da2256c2db53ffd245e724c6fa91bb40e6`; 128 decode steps per workload.
- Actual layer tensor audit: `9` parameter tensors, `7080192` reference bytes.
- `cold_packed_projection_sequential_materialization`: G2=True, G3=True, G4=False; artifact=4208296 B; reference-layer=7080192 B; compiled-layer-resident=1771776 B; baseline p50/p95=105.321 ms/202.285 ms; candidate p50/p95=137.679 ms/234.948 ms.
- `checkpoint_mlp_torchinductor_existing_isa`: G2=False, G3=True, G4=False; artifact=169700 B; reference-layer=7080192 B; compiled-layer-resident=7080192 B; baseline p50/p95=102.057 ms/102.057 ms; candidate p50/p95=2268.660 ms/2268.660 ms.
- Instruction/SASS counts are not synthesized: each mechanism reports its measured/not-measured status explicitly.
- G5/G6 remain gated by TARGET-W measurements; DEV-W results are not extrapolated into a 405B performance claim.

<!-- FIXED_PUBLIC_DYNAMIC_EXECUTOR_RESULT:7777eda66232c6a37c39f4e55f5b9ce56032fe92 -->
## Fixed-public dynamic executor validation row

- G1 ACTUAL RUNTIME: PASS on DEV-W; official loader + forward executed.
- G2 EXACT TRANSITION: True.
- G3 REAL BOUNDARY: True.
- G4 PHYSICAL SAVING: True.
- G5 FULL-MODEL PATH: formula recorded; TARGET-W 4B-class budget not established.
- G6 8GiB LEDGER: TARGET-W not established.
- G7 EXISTING ISA: PASS for implemented software primitives; no hypothetical opcode.
- G8 REPRODUCIBILITY: raw hosted result committed by workflow; workflow artifact retained.

## Fixed-public dynamic executor hosted result — commit `7777eda66232c6a37c39f4e55f5b9ce56032fe92` / run `32219219400`

- Verdict: `REAL_CHECKPOINT_DYNAMIC_EXECUTOR_SURVIVES_G4`
- DEV-W: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; resolved SHA `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; official `LlamaForCausalLM.from_pretrained()` executed.
- TARGET-W: `meta-llama/Meta-Llama-3.1-405B-Instruct@f9801cba95a53242b3cc928a4a418d12571d1c5f`; metadata access=True; full TARGET-W weights/performance remain unevaluated unless the gated tensors were actually accessible.
- Official reference forward: executed=True.
- Frozen workload SHA-256: `f52dad873703170179b08bf2c48637da2256c2db53ffd245e724c6fa91bb40e6`; 128 decode steps per workload.
- Actual layer tensor audit: `9` parameter tensors, `7080192` reference bytes.
- `checkpoint_mlp_output_row_streamed_lossless_existing_isa`: G2=True, G3=True, G4=True; output-row-tile=128; artifact=4223092 B; reference-layer=7080192 B; compiled-layer-resident=1771776 B; peak-hot=393216 B; baseline p50/p95=113.832 ms/203.051 ms; candidate p50/p95=151.764 ms/241.453 ms.
- Instruction/SASS counts are not synthesized: each mechanism reports its measured/not-measured status explicitly.
- G5/G6 remain gated by TARGET-W measurements; DEV-W results are not extrapolated into a 405B performance claim.

<!-- EXP085A_PREREGISTERED -->
## EXP-085A validation registration

| Gate | Evidence required | Status before run |
|---|---|---|
| official public checkpoint | pinned `LlamaForCausalLM.from_pretrained` | NOT RUN |
| real causal activations | 6 families x 4 positions | NOT RUN |
| exactness | zero row-split mismatch, bound violation, false accept, candidate mismatch | NOT RUN |
| favorable oracle | p50/p95 fractions and fallback | NOT RUN |
| sound selector | p50/p95 fractions, fallback, metadata | NOT RUN |
| complete layer / 128 steps / hardware | separate later gate | NOT AUTHORIZED |


<!-- EXP086A_RESULT -->
## EXP-086A — Global BF16 successive refinement

```text
decision                                    REJECT_GLOBAL_BF16_MANTISSA_PREFIX_AS_COLD_QUERY_CORE
official checkpoint                         HuggingFaceTB/SmolLM2-135M
actual causal activations                   24
full-precision control                      True
row-adaptive entropy fraction p50/p95       0.9507781338525199 / 0.951998430042707
minimum perfect acceptance p50/p95          66 / 53
deterministic core SHA-256                   17478366c320c4f46ffa04a238a54e6fe4d34d37360439f8434c48dfd98bd4e4
```

The Gate grants ideal zero-order coding, reference-aided per-output-row down
precision, free non-MLP traffic, and free selector/decompression/kernel costs.
It is not a physical speed result.

<!-- EXP087A_RESULT:bda5ccfd4d4cfc66d430b09d45f311f1e02ece18 -->
## EXP-087A result — `bda5ccfd4d4cfc66d430b09d45f311f1e02ece18`

```text
decision                         REJECT_ALIGNED_CROSS_WEIGHT_CONTEXT_DICTIONARY_AS_COLD_QUERY_CORE
layer count                      3
best information fraction p50   1.028159047473748
best information fraction p95   1.0318905207445657
best candidate names             ['coordinate_period_2', 'coordinate_period_2', 'coordinate_period_2']
all exact reconstructions        True
```

This is a favorable exact checkpoint-description Gate. Model forwards, dense-operation replacement, complete layer/state, 405B, 8 GiB, and physical latency remain `NOT TESTED`.

<!-- EXP088A_RESULT:b5a5045a311a0f3c1e21968fc3451eb2f90775a0 -->
## EXP-088A result — `b5a5045a311a0f3c1e21968fc3451eb2f90775a0`

```text
decision                          REJECT_CROSS_LAYER_TEMPLATE_AND_RECURRENCE_CODE_AS_COLD_QUERY_CORE
layer count                       30
role-wise best candidates         {'down_transposed': 'previous_layer_xor', 'gate': 'previous_layer_xor', 'up': 'previous_layer_xor'}
total information fraction        1.0538305572685522
effective layer fraction p50      1.0543995370001458
effective layer fraction p95      1.0567897596057383
all exact reconstructions         True
```

This is a favorable no-forward checkpoint-description Gate. Dense operation replacement, complete transition/state, 405B, 8 GiB, and physical latency remain `NOT TESTED`.
