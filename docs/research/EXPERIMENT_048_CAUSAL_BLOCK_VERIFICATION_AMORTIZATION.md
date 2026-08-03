# EXP-048 — Causal Block Verification Amortization Gate

## Final status

```text
Scientific decision: REJECT_PARTIAL_LAYER_SELF_DRAFT_CORE_RETAIN_EXACT_BLOCK_VERIFIER
Phase: A/B/C-observation
Evidence: E1
Complete real operation replacement: false
Phase D: NOT TESTED
```

Authoritative evidence:

```text
results/exp_048/summary.json
workflow 30798936320
source head SHA 484a1f0f313d88733d2f7210f2a24d3904bf1373
workflow merge SHA d60e392d66d694fc020f2cfe2435e47e5f5a22ca
artifact 8850040445
artifact ZIP SHA-256 67c587da36b968f9c38e0a7774ea03cecd2ad2d7d274d3e83c833c56529c3443
```

## Question tested

After EXP-047R rejected range-based weight skipping, EXP-048 tested whether one exact target stream could be amortized across a long causal proposal block without training or modifying the checkpoint.

The verifier and proposal source were separated deliberately:

- B1 asked whether a sufficiently accurate proposal would make the traffic arithmetic viable;
- B2 measured hard Jacobi fixed-point iteration with every target pass charged;
- B3 measured a deployable training-free proposal from the same checkpoint's early layers.

## Exact verifier contract

For exact prefix `p`, proposal `q`, and one exact causal target pass over `p + q`, target predictions were aligned at `prefix_length - 1 + i`.

Only the longest matching proposal prefix was committed. At the first mismatch, the exact target token at that position was committed and every later proposal/target position was discarded. This rule passed all model-independent tests and all committed model cases.

## Frozen conditions

### B0 — exact sequential baseline

96 exact greedy tokens with target KV cache; one logical target stream per token.

### B1 — perfect future-token oracle

The exact 96-token continuation was proposed and verified with one target pass.

MEASURED:

```text
18/18 exact blocks matched
96 committed tokens per target pass
target-equivalent fraction 1/96 = 1.0416667%
future generated tokens used true
deployable false
```

B1 beats the PROJECTED 1.185185% traffic threshold, proving that block verification is arithmetically sufficient when proposal accuracy is effectively perfect. It is not runtime evidence because it consumes future target tokens.

### B2 — hard Jacobi control

A 32-token block, fill token zero, and at most four exact target iterations per cycle.

MEASURED:

```text
exact mismatches 0
p50 target passes per 32 exact tokens 58
p50 accepted tokens per target pass 0.551724
p50 target-equivalent fraction 181.25%
p90 target-equivalent fraction 193.75%
maximum matching prefix 3
future information false
```

B2 was slower in logical target streams than exact sequential decoding.

### B3 — causal partial-layer self-draft

For every model/prompt state, 32 proposal tokens were generated sequentially from the exact current prefix using the first 1, 2, or 4 target layers, the target final normalization, and the target LM head. One exact full-target block pass then verified the proposal.

Every draft layer stream, full output-head/norm stream, gathered embedding rows, target pass, rejected position, and correction was charged.

MEASURED:

```text
3 pinned trained dense models
6 held-out families
18 cases
54 fixed B3 variants
exact mismatches 0
future information uses 0
best-variant cases with any matching proposal token 4/18
maximum best-variant matching prefix 1
p50 exact committed tokens per target verification 1
model medians 1 / 1 / 1
minimum fully accounted fraction 1333.463%
p90 fully accounted fraction 2893.843%
```

The four 2-token commits consisted of one matching proposal token plus the exact correction. The remaining best cases committed only the correction token.

## Pinned external state

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Exact file hashes are in `results/exp_048/raw/checkpoint_manifest.json`.

## Pre-registered Gate outcome

Required:

```text
B3 p50 committed tokens >=16
B3 p90 target-equivalent fraction <=10%
B3 cost below sequential B0
zero mismatch
zero deployable future information
non-degrading model-size trend
```

Observed:

```text
acceptance FAIL: 1 <16
traffic FAIL: 28.9384258 >0.10
cost FAIL: 28.9384258 >1.0
exactness PASS
causality PASS
size-trend PASS only because all model medians were equally poor at 1
```

Decision:

```text
REJECT_PARTIAL_LAYER_SELF_DRAFT_CORE_RETAIN_EXACT_BLOCK_VERIFIER
```

## Scientific interpretation

- Exact block verification is retained as an auxiliary primitive.
- B1 proves the final traffic target is not blocked by verifier arithmetic alone; it is blocked by causal proposal quality and proposal cost.
- Hard Jacobi does not amortize target streams.
- Early-layer self-drafting does not predict even a short exact prefix and repeatedly executes an LM head that dominates these small checkpoints.
- B4 proposal-tree expansion is not continued from B3 because B3 failed its mandatory early Gate.
- A new core mechanism must avoid both per-token sequential draft passes and dozens of full-target fixed-point passes.

## Next mechanism boundary

The next candidate is EXP-049, a large-block continuous fixed-point solver with damped and Anderson-accelerated soft-token states. It will execute a small pre-registered number of full batched target passes, harden a proposal, and use the retained exact block verifier. Hard Jacobi remains the control.

This does not assume the solver will work. The causal triangular dependency may itself impose a one-position-per-round barrier; EXP-049 must include an explicit lower-bound/counterexample analysis and reject the family if acceleration cannot produce long exact prefixes.

## Projection boundary

```text
405B Q4 full stream: 188.592821 GiB
1.2x 4B Q4 allowance: 2.235174 GiB/token
required target-equivalent stream fraction: 1.185185%
zero-cost perfect-proposal minimum: 85 accepted tokens/full target stream
B1 oracle 96-token fraction / required fraction: 0.87890625
B3 p90 fraction / required fraction: 2441.6793
```

These are logical same-bit projections, not target-hardware measurements.

## Evidence layout and restoration

```text
results/exp_048/summary.json
results/exp_048/raw/artifact_provenance.json
results/exp_048/raw/checkpoint_manifest.json
results/exp_048/raw/cases.jsonl.gz.b64
results/exp_048/processed/aggregate.json
results/exp_048/logs/run.log
results/exp_048/artifacts/
results/exp_048/checksums.sha256
```

Restore exact raw cases:

```bash
base64 -d results/exp_048/raw/cases.jsonl.gz.b64 | gunzip > /tmp/exp_048_cases.jsonl
sha256sum /tmp/exp_048_cases.jsonl
# expected e3278b735217b8ffea737a60578513271f286a58f3180f8109e04005ea734deb
```

## Claim boundary

```text
405B execution: NOT TESTED
8 GiB VRAM: NOT TESTED
complete real operation replacement: false
physical weight reuse across block positions: NOT TESTED
CUDA/PCIe/SSD/TTFT/tokens per second: NOT TESTED
Phase D: NOT TESTED
E6/E7: not achieved
```
