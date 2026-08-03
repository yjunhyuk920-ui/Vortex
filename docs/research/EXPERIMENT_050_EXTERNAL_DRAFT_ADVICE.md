# EXP-050 — Target-Independent External Draft Advice Gate

## Final status

```text
Scientific decision: REJECT_TARGET_INDEPENDENT_EXTERNAL_DRAFT_AS_UNIVERSAL_CORE
Tested fixed pool practical continuation: REJECTED
Phase: A/B/C-observation
Evidence: E1
Complete real operation replacement: false
Phase D: NOT TESTED
```

Authoritative evidence:

```text
results/exp_050/summary.json
workflow 30806015309
source head SHA 1388c780abea11067c66cd666ed0a313ec2f682c
workflow merge SHA 6bdd0a20334e394ec5252a6c0e676c1f62b608d0
artifact 8852817664
artifact size 34225 bytes
artifact ZIP SHA-256 a32ffe8dbfc201c6d70ca8dac660164d8400691ad4d8fe3593d688e7754f6159
```

## Question tested

After EXP-049 rejected target-only fixed-point proposals, EXP-050 tested whether another already-published, unmodified small causal checkpoint could provide long exact target proposals without target future tokens or target-specific training.

```text
external draft cached greedy proposal
        -> one exact target block pass
        -> longest target-matching prefix
        -> exact first-mismatch correction
```

## Pinned pool

```text
TinyStories-1M target <- drafts 3M,8M
TinyStories-3M target <- drafts 1M,8M
TinyStories-8M target <- drafts 1M,3M
```

The common tokenizer and exact revisions are recorded in the raw manifest. Six held-out families were English narrative, Korean, code, mathematics, structured JSON, and identifier boundary.

## Frozen execution

For every model/prompt, one 256-token cached greedy continuation was generated using that model's own KV state. For every target/draft/prompt pair, the target executed one exact causal pass over the 256-token external proposal. K=64/128/256 rows were causal prefixes of that same pass.

```text
3 models
18 target/prompt cases
36 target/draft/prompt pairs
108 pair/K rows
excluded states 0
```

## Correctness and causality

MEASURED:

```text
9 EXP-050 tests passed
repository validation passed
all-pair exact committed-output mismatches 0
all-pair target-future information uses 0
E3 exact future-target oracle failures 0
peak RSS 871824 KiB
```

Every draft charged one sequential forward per proposed token. Exact target verification committed only matching proposal tokens plus the exact first-mismatch correction.

## Favorable fixed-pool upper bound

The exact target reference selected the best eligible draft and K separately for every target/prompt. This selector is explicitly non-deployable and favors the hypothesis.

MEASURED:

```text
p50 exact proposal prefix 0.5
maximum exact proposal prefix 3
p90 normalized 4B/405B fraction 1.6320987654 =163.20987654%
selected K=64 in 18/18 cases
selected drafts: 1M 12 / 3M 4 / 8M 2
```

All pair rows:

```text
prefix 0: 72/108
prefix 1: 24/108
prefix 2: 6/108
prefix 3: 6/108
```

Best observed rows were identifier-boundary states with prefix 3. No proposal exceeded three matching target tokens.

## Family and target trend

```text
English narrative useful acceptance true
Code true
Mathematics true
Identifier boundary true
Korean false
Structured JSON false
```

Target median prefixes:

```text
TinyStories-1M 1.0
TinyStories-3M 0.0
TinyStories-8M 0.5
```

Thus the fixed pool failed required family coverage and non-degrading target-size trend.

## Universal first-token counterexample

A deterministic external draft proposed token 7. An arbitrary causal target chose token 8 on the same prompt.

```text
matching proposal prefix 0
exact committed tokens 1
correction used true
exact target output preserved true
```

Therefore no fixed target-independent draft can guarantee a nonzero exact prefix for every arbitrary target.

## Gate outcome

Required:

```text
p50 prefix >=16
p90 normalized fraction <=10%
useful acceptance in every family
largest target median >=75% smallest target median
no universal first-token counterexample
zero exact mismatch/future leakage
```

Observed:

```text
prefix FAIL: 0.5 <16
traffic FAIL: 163.20987654% >10%
family coverage FAIL: Korean and JSON
size trend FAIL
universal Gate FAIL: prefix-zero counterexample
exactness PASS
causality PASS
```

Decision:

```text
REJECT_TARGET_INDEPENDENT_EXTERNAL_DRAFT_AS_UNIVERSAL_CORE
```

The tested fixed pool is also rejected as a restricted practical core. Proposal-tree expansion is not permitted from this failed pool.

## Resource interpretation

PROJECTED 4B draft /405B target:

```text
draft ratio 4/405 =0.0098765432
required total fraction 0.01185185185
perfect proposal condition 4/405 + 1/K <= required
minimum K 507
```

The current maximum empirical prefix was 3, not 507. The favorable p90 fraction was 137.7083 times the required target fraction.

## Next mechanism boundary

EXP-051 changes the skip axis from future token prediction to Transformer layer depth. It uses exact target prefixes, probes every intermediate block output through the original final norm/LM head, and measures earliest suffix-stable final-token depth.

This is not recursive partial-layer drafting. It is a current-token oracle upper bound for skipping the remaining target tail. A late-final-layer residual adversary tests the universal fixed-depth claim.

## Claim boundary

```text
405B execution NOT TESTED
8 GiB VRAM NOT TESTED
combined target/draft/KV fit NOT TESTED
CUDA/PCIe/SSD/TTFT/tokens per second NOT TESTED
complete real operation replacement false
Phase D NOT TESTED
E6/E7 not achieved
```

## Reproduce

```bash
python -m pytest -q tests/exp_050
python scripts/run_validation.py
bash experiments/exp_050/reproduce.sh
cd results/exp_050 && sha256sum -c checksums.sha256
```
