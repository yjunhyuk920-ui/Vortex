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
