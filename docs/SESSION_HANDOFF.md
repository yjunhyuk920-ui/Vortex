# Session Handoff

Last updated: 2026-08-03 Asia/Seoul

## Mandatory startup

Read in order:

1. `AGENTS.md`
2. `RESEARCH_STATE.md`
3. `FAILED_APPROACHES.md`
4. `DECISION_LOG.md`
5. `ASSUMPTION_REGISTER.md`
6. `VALIDATION_MATRIX.md`
7. `NEXT_EXPERIMENT.md`
8. `ARCHITECTURE.md`
9. `HARDWARE_VALIDATION_PLAN.md`
10. `REPRODUCIBILITY.md`
11. `docs/PROOF_FIRST_CONTRACT.md`
12. `docs/WORK_SESSION_PROTOCOL.md`
13. EXP-049 document, `results/exp_049/summary.json`, PR #59.

Root files and machine-readable result JSON are authoritative; conversation memory is not.

## Fixed target

Real arbitrary unmodified Hugging Face dense target, runtime only, 405B flagship, total peak GPU VRAM <=8 GiB, original contract preserved, p50 warm time/token <=1.2x native 4B Q4 and p95 <=1.5x on the same machine.

## Environment truth

```text
8 GiB target GPU unavailable
405B storage/execution unavailable
CUDA/PCIe/target SSD profiling unavailable
real 405B TTFT/tokens/sec/VRAM NOT TESTED
physical block weight reuse NOT TESTED
Phase D NOT TESTED
E6/E7 not achieved
```

## Current branch and PR

```text
repository yjunhyuk920-ui/Vortex
branch research/exp-049-anderson-continuous-fixed-point
PR #59
```

## EXP-049 authoritative evidence

```text
results/exp_049/summary.json
workflow 30803672059
source head SHA 91d0caa86d784c663bc520d36d9b512f0cc526e9
workflow merge SHA 173dd3477e2a6f5ecb0d55b58375ec18dfe774dd
artifact 8851957250
artifact size 105493 bytes
artifact ZIP SHA-256 4cd6c8c4afb833562438a97f052d45d331f3691362472fb08e594bd0c5585b9e
```

Pinned external revisions:

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

## EXP-049 MEASURED result

```text
9 EXP-049 tests passed
repository validation passed
3 models × 6 families = 18 cases
1,458 fixed solver trajectories
excluded states 0
selected exact mismatch 0
selected future-information uses 0
unhandled numerical failures 0
S3 oracle alignment failures 0
peak RSS 684684 KiB
```

Favorable exact-reference-selected S1/S2:

```text
p50 matching prefix 4.5
maximum matching prefix 6
p90 target-equivalent fraction 168.778596%
model medians 4.5 / 5.0 / 4.0
all selected rows: 4 target passes, block 64
17/18 hard top-1 Picard
0/18 Anderson
```

Controls:

```text
hard Jacobi p50 prefix after four passes 4
Anderson p50 prefix after four passes 1
Anderson/Jacobi improvement 0.25x
```

Triangular audit:

```text
Picard prefixes 1,2,3,4
Anderson prefixes 1,2,3,3
hidden suffix transcript indistinguishability true
one-new-exact-position-per-round barrier true
```

## Scientific decision

```text
EXP-049 solver/numerical reference: ACCEPT E1 AUXILIARY
exact block verifier: RETAIN E1 AUXILIARY
target-only continuous fixed-point core: REJECT
405B/8 GiB/4B-class performance: NOT TESTED
```

Required phrase:

> Even with exact-reference selection of the best fixed Picard/Anderson trajectory, EXP-049 achieved only p50 4.5 exact proposal tokens and p90 1.6878 target-equivalent streams per committed token. Hidden triangular targets also preserved the one-new-position-per-round transcript barrier. Target-only continuous fixed-point proposal generation is rejected as core; 405B and target hardware remain untested.

## Frozen evidence layout

```text
results/exp_049/summary.json
results/exp_049/raw/artifact_provenance.json
results/exp_049/raw/workflow_summary.json
results/exp_049/raw/checkpoint_manifest.json
results/exp_049/raw/triangular_audit.json
results/exp_049/raw/cases.jsonl
results/exp_049/processed/aggregate.json
results/exp_049/logs/run.log
results/exp_049/artifacts/
results/exp_049/checksums.sha256
```

`.github/workflows/exp_049_gate.yml` is manual-only and writes isolated reproduction output. The one-shot evidence-freeze workflow installed byte-identical authoritative evidence and does not trigger on ordinary result/document changes.

## Next work — EXP-050

`Target-Independent External Draft Advice Gate` changes the information source.

Algorithm:

1. load exact prompt into target and another already-published unmodified draft checkpoint;
2. generate a causal draft continuation with draft KV cache;
3. verify the entire draft block with one exact target pass;
4. commit only target-matching prefix plus exact first-mismatch correction;
5. charge every draft token forward, target verification, selector, rejected position, KV state, and correction.

Fixed initial pool:

```text
Target 1M <- Drafts 3M,8M
Target 3M <- Drafts 1M,8M
Target 8M <- Drafts 1M,3M
```

Controls:

```text
E0 target-independent first-token counterexample
E1 every cross-checkpoint single draft
E2 exact-reference favorable pool selection, non-deployable
E3 exact future-target oracle, non-deployable
E4 tree forbidden unless E2 survives
```

Early rejection:

```text
any verifier mismatch
any target-future leakage
favorable pool p50 exact prefix <16
p90 4B/405B-normalized fraction >10%
any required family with zero useful proposal acceptance
worsening size trend
universal first-token counterexample succeeds
```

Universal boundary: a fixed target-independent draft can always be contradicted by an arbitrary target choosing a different first token. Practical pool evidence is separate from the arbitrary-model claim.

PROJECTED 4B draft requirement:

```text
4/405 = 0.0098765432 target streams/draft token
required total fraction 0.01185185185
perfect proposal minimum after draft cost 507 tokens
```

## Reproduction

```bash
git checkout research/exp-049-anderson-continuous-fixed-point
python -m pytest -q tests/exp_049
python scripts/run_validation.py
bash experiments/exp_049/reproduce.sh
cd results/exp_049 && sha256sum -c checksums.sha256
```

Do not overwrite frozen `results/exp_047*`, `results/exp_048`, or `results/exp_049`. EXP-050 uses a new branch/result directory.

<!-- EXP-052-AUTHORITATIVE-FINAL -->
## EXP-052 handoff

Enumerative exact advice is rejected. Read `results/exp_052/summary.json` and `NEXT_EXPERIMENT.md`; continue with EXP-053 or a materially new mechanism only.

<!-- EXP-053-AUTHORITATIVE-FINAL -->
## EXP-053 handoff

Bit-exact AIG structural hashing is rejected as core. Read `results/exp_053/summary.json` and continue with EXP-054 reduced decision diagrams or a materially new mechanism only.

<!-- EXP-054-AUTHORITATIVE-FINAL -->
## EXP-054 handoff

Reduced decision diagrams are rejected as core. Continue with EXP-055 word-level column-signature/popcount aggregation or a materially new mechanism only.

<!-- EXP-072A-AUTHORITATIVE-FINAL -->
## Current handoff after EXP-072A

Branch: `research/exp-072a-information-capacity`.

Authority: `results/exp_072a/summary.json`; source `468f297925e10bdc541fe48f19c2f72a1e3f5e14`; evidence commit `f9ac26befb01fd9a71c7c6e1efed4c4b4df31389`.

Decision: reject a self-contained exact Q4 arithmetic DAG as a universal 8 GiB hot core. Exact basis outputs make the artifact an injective matrix encoding; worst-case Q4 information is `188.98828125 GiB`, `23.62353515625x` the hot allowance. Restricted synthesis remains auxiliary. Cold-backed online execution is not ruled out.

Next: EXP-073 sanitized, read-only inventory of the privately identified Ubuntu target, followed only with separate authorization by storage/transfer/native-4B baselines. Never commit connection details. Do not download 405B, install packages, restart services, or disturb workloads during inventory.

<!-- EXP-074-AUTHORITATIVE-FINAL -->
## Current handoff after EXP-074

Branch: `research/exp-074-weight-stationary-block-gate`.

Authority: `results/exp_074/summary.json`; source
`8abc06e73c884b839927cf41d5f4fa6cbb8fc051`; evidence
`c1d778af011672ec7fadfa66935ba2548de8e115`.

Decision: MTP-1 plus expert paging is rejected as a 1B-class core after
retaining `10.0x` baseline traffic under free drafting and fixed routes. The
long-block path is revised: fixed/free requires nine perfect accepted tokens,
0.8B draft requires 25, and independent-uniform expected routing requires 98.

Next: metadata-only native-MTP surface audit, then the smallest unchanged
checkpoint accepted-prefix Gate if the surface exists. Do not download 35B or
122B and do not build a page scheduler. EXP-073 Stage 2 remains not run.

<!-- EXP-075-AUTHORITATIVE-FINAL -->
## Current handoff after EXP-075

Branch: `research/exp-075-native-mtp-surface-audit`.

Authority: `results/exp_075/summary.json`; source
`f85ac583a129070247992987d1b3c63634e6447f`; evidence
`2fcf7315cf9da491a5ca361536eb0f07e325e74c`.

Decision: the official Qwen3.5-0.8B config, exact 15-key MTP index surface, and
pinned vLLM loader/runtime source all pass the metadata Gate. The complete
checkpoint is declared as 1.626911 GiB but was not downloaded. No model or
server command ran.

Next: preregister EXP-076 around this exact model/revision, an isolated compatible
dependency lock, causal recursive proposals for K=2..64, exact verification and
hybrid-state rollback, held-out prompt families, and fully charged p05/p50
acceptance accounting. Do not start 35B/122B, quantization, expert paging, or
server work. EXP-073 Stage 2 remains not run.

<!-- EXP-076-AUTHORITATIVE-FINAL -->
## Current handoff after EXP-076

Branch: `research/exp-076-native-mtp-accepted-prefix-gate`.

Authority: `results/exp_076/summary.json`; source
`5e331137f8e03250cc74aa796abbf69f49ef87a5`; evidence
`55b79937c1f21887ae76b7e56ad61ba7dde8322a`.

Decision: reject native-MTP long blocks as the registered surrogate core. The
build split selected `K=4`; held-out accepted-prefix p05/p50/p95 was `0/4/4`
versus required p05/p50 minima `9/11`, and 2/18 cases accepted zero drafts.
All causal, exact-acceptance, commit-state, and rollback controls passed.

Next: do not run 35B/122B, router tracing, quantization rescue, or a page
scheduler. No EXP-077 core implementation is authorized until a materially
different query-time information source passes E0. EXP-073 Stage 2 remains the
only specified physical calibration and requires separate explicit authority.
The private target server was not contacted in EXP-076.

<!-- E0-SPARSE-FUNCTIONAL-DICTIONARY-HANDOFF -->
## Current handoff after the sparse functional dictionary audit

Branch: `research/exp-082a-differential-spanning-tree`.

Authority:
`docs/research/E0_SPARSE_FUNCTIONAL_DICTIONARY_FRONTIER.md` and
`results/e0_sparse_functional_dictionary_frontier/summary.json`.

Literal global bilinear answers and direct local rank-one truth tables are
closed by finite address/storage Gates. A globally non-systematic cold linear
dictionary remains an exact open interface, but query cardinality supplies
neither rank-one alignment nor a sparse decoder. No candidate, model run,
backend, or hardware stage was promoted.

Next: search only for an explicit near-linear aligned dictionary with a
succinct exact decomposer and native-order lift, or a covering lower bound that
actually applies to the non-systematic rank-one linear model. Keep
`NO_SURVIVING_CANDIDATE` and do not assign EXP-085 without one deliverable.

The first decomposer subcase is now closed. Any fixed linear right inverse
activates almost every useful dictionary row across an independently selectable
32-query tuple; even perfect bit packing leaves `41.874345776 GB` cold after
the complete hot grant. Continue only with a nonlinear minimum-weight syndrome
decoder or a proved causal restriction. Authority:
`docs/research/E0_FIXED_LINEAR_FUNCTIONAL_DECODER_ACTIVITY_BOUND.md`.

<!-- EXP-077A-AUTHORITATIVE-FINAL -->
## Current handoff after EXP-077A

Branch: `research/exp-077a-oracle-fractal-mlp-gate`.

Authority: `results/exp_077a/summary.json`; source
`33fed17ed6abed7c8c14eec543efc620e4fe537d`; evidence
`0970c6626ff848c5026b684b3e2d1bb479e96603`; core
`e25083693c6db21a0da16c9305816958b149865f2fcfde0bd9b7e95c43022411`.

Decision: reject activation-norm individual-channel Fractal MLP at the 10%
favorable ceiling. The corrected causal-cache control matched 192/192 target
positions. The realized `9.988839%` arm achieved held-out top-1 `71.5278%`,
mean KL `0.884161`, and p95 KL `3.080752`; every family failed. The invalid
pre-authority runs were not committed as evidence.

Next: do not sweep this score, train a router, build a sparse kernel, or download
35B/122B. A new candidate must introduce a different correction/amortization
dependency and pass E0. EXP-073 Stage 2 remains separately authorized but not
run. The private Ubuntu server was not contacted; Phase D/E2-E7 remain open.

<!-- EXP-078A-AUTHORITATIVE-FINAL -->
## Current handoff after EXP-078A

Branch: `research/exp-078a-tangent-macroblock-gate`.

Authority: `results/exp_078a/summary.json`; source
`cc368190031d87db92f7a476dd741c18224239c1`; evidence
`fe6081917c65b2392f8760d72a2b0e17a4461982`; core
`c743ae14748effaad3a034def7d8d92e67fa09abf65eeb931bef3e8a26368667`.

Decision: reject frozen complete anchor-conditioned MLP operator reuse. The
unchanged checkpoint matched all 192 baseline decisions. The candidate then
matched 0/126 held-out later-token top-1 decisions; all 18 cases failed on the
first reuse token; mean/p95 KL was `14.898423/25.335417`.

The direct-build cost also required `13,821/6,249` small-checkpoint and
`115,299/26,354` 122B-screen p50/p05 reuse positions. Do not extend the trace,
sweep ranks/layers/prompts, add a sentinel, or construct a physical macro matrix.
A causal delta-updated map is a new unsupported mechanism and must first pass E0
with update, detection, repair, fallback, and cold-traffic costs. No core
candidate survives; the Ubuntu server was not contacted; Phase D/E2-E7 remain
open.

<!-- E0-POST-ATLAS-CAUSAL-SOURCE-AUDIT -->
## Current handoff after the post-Atlas causal-source audit

Branch: `research/exp-082a-differential-spanning-tree`.

Authority:
`docs/research/E0_POST_ATLAS_CAUSAL_INFORMATION_SOURCE_AUDIT.md`.

Decision: forward-only continuation is Atlas normal form; dynamic exact
decision-dual construction fails at `1.5625%` over 64 tokens; a favorable
static one-last-down vocabulary table fails at `12.720703125 GiB` and
`6.765979992%` scan traffic. The EXP-083B zero-forward post-hoc screen leaves
all 248,319 competitors unresolved. No experiment number or model run occurred.

Next: address only the lossless Bilinear Cross Residual `r^T W u` with a
concrete finite E0 construction or scoped lower bound. Do not run hardware,
contact the Ubuntu server, download a larger model, or start E1/E2. Current
status is `NO_SURVIVING_CANDIDATE`; this negative closure is not the Certain
Research Milestone.

<!-- E0-EXTENSION-FIELD-RANK-SATURATING-HANDOFF -->
## Current handoff after the extension-field rank-saturation audit

Branch: `research/exp-082a-differential-spanning-tree`.

Authority:
`docs/research/E0_EXTENSION_FIELD_RANK_SATURATING_FRONTIER.md` and
`results/e0_extension_field_rank_saturating_frontier/summary.json`.

Extension-field packing is exact over GF(2), but published coefficient rank is
not probe support. Direct full-ambient summaries cost `748.5509965564124 TiB`;
the rank-one-specific identity system restores a raw one-bit scan and exceeds
the zero-compute latency Gate. The throwaway small-search source is preserved
at commit `2f68300` on `prototype/rank-one-dictionary-small-search`.

Next: work only on an explicit joint batch coset dictionary satisfying
`q_i in span(G_T)` for one small shared physical `T`, or change mechanism
class. A single-query sparse leader is insufficient. Keep
`NO_SURVIVING_CANDIDATE`; do not start EXP-085, model, backend, download,
server, or hardware work.
