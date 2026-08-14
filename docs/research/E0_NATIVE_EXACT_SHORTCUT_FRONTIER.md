# E0 Native-Exact Shortcut Frontier

Status: all screened candidates rejected by one favorable necessary condition;
the target remains unsolved.

## Purpose

The old `2.5% evidence` hypothesis is not used here.  This audit asks only
whether a new mechanism could plausibly meet all three fixed requirements:

1. unchanged reference result;
2. at most 8 GiB accelerator state;
3. at most 20 ms per committed token on the registered machine contract.

The registered compute envelope permits `9.6 GFLOP/token` against
`811.698487296 GFLOP/token` for the dense 405B reference.  A shortcut proposed
as the universal core must therefore eliminate at least `98.81729483%` of the
dense coefficient work before its own positive costs.  Each screen below is
more favorable than a real implementation: metadata, addressing, decoding,
state, verification, and cache traffic are free.

## Fast design loop

The following ideas were rejected without an implementation because an
existing result or a semantic invariant already decides them.

| Candidate | Cheapest necessary condition | Decision |
|---|---|---|
| deterministic trace fingerprint | must also produce the exact trace being checked | duplicate of F-048; reject |
| compiled checkpoint transducer | finite table/BDD must fit the 8 GiB state and serve unseen contexts | duplicate of F-019/F-020/F-022; reject |
| computational storage / near-data execution | must use the unchanged same-machine hardware contract | contract change; reject |
| regular bit-plane circuit plus token batching | must both preserve native signed numerical semantics and amortize a full description over enough causally accepted tokens | neither premise is supplied; reject |
| CKL succinct Boolean MatVec | must return a zero-error signed numerical answer in native order | theorem returns randomized Boolean-semiring output; reject direct lift |

The next six candidates received one measurement on already pinned evidence.
No model forward, new checkpoint, target-server action, kernel, or hardware run
was performed.

## Pinned observation

```text
checkpoint       Qwen/Qwen3.5-0.8B, unchanged pinned EXP-076 payload
tensor           layer-23 MLP down_proj, [1024, 3584], BF16
causal evidence  legal_holdout_english_01 from EXP-083B
prefix rows      29
```

### Reference-rounding absorption

Grant a term free removal whenever its magnitude is at most **twice the FP32
ULP of the entire row L1 sum**.  This is deliberately looser than a native
accumulator-order test and is only an upper bound on potential absorption.

```text
row removable fraction  min 0.3627%, p50 0.6417%, p95 0.8650%, max 1.2556%
required universal elimination                              98.8173%
```

Decision: reject before implementing native-order delta accumulation.

### Exact temporal coordinate reuse

```text
consecutive same-coordinate values  max 0.3348%
unlimited prior-token coordinate cache, last/max 2.9855%
natural contiguous run saving upper       0.1674%
```

Decision: reject coordinate delta, run-tree, and unlimited same-coordinate
memoization variants.  Even a free infinite cache leaves almost every
coefficient contribution dynamic.

### Exact product reuse and cancellation

For the current exact activation, each output row was screened by FP32 product
word identity.

```text
unique product fraction       min 97.0145%, p50 97.8237%, max 98.6607%
best duplicate reuse upper                                       2.9855%
perfect p/-p cancellation     min 2.6228%, p50 4.1295%, max 5.6920%
```

Decision: reject product dictionaries and perfect opposite-pair cancellation.
The latter grants every exact opposite pair a zero-cost cancellation and still
misses the necessary reduction by a wide margin.

### Low-VC / Pollard structured MatVec

Anand, van den Brand, and McCarty give exact numerical MatVec query time
`~O(T n m^(1-1/d) + m)` when `T` bounds the number of unique values per
column and `d` is the Pollard pseudodimension.  The row-difference mechanism
behind the theorem is already covered and rejected by EXP-082A.  The direct
non-Boolean extension also fails an even cheaper premise:

```text
unique BF16 values per column  min 653, p50 694.5, max 739
unique BF16 values per row     min 1276, p50 1331, max 1461
columns/rows with <=84 values  0 / 0
```

Even granting the mathematically best exponent `d=1`, zero logarithms, and
unit constants, the better orientation costs at least `35.6027%` of dense
work.  The complete allowance is `1.1827%`.  Decision: reject this constructor
without computing VC dimension or building another spanning tree.

Primary source:
<https://arxiv.org/html/2502.21240v3>.

### ΩROUNDLOCK -- layerwise rounding firewall

ΩROUNDLOCK is an original state-exactization proposal from this project. A
cheap source proposes a layer output, coordinates whose BF16 words equal the
native target are locked as exact singletons, and only unlocked coordinates
are repaired. A successful lock resets uncertainty at every layer instead of
letting an enclosure grow through the whole network.

The cheapest Gate grants the unavailable target words and the lock selector
for free. It applies the already computed Atlas-plus-one-page candidate to the
same frozen row:

```text
location                    locked / 1,024    oracle lock fraction
down projection                    1 / 1,024                0.0977%
post-residual pre-norm             7 / 1,024                0.6836%
post-RMSNorm hidden                2 / 1,024                0.1953%
whole vector exact                 0 / 3 locations             false
```

Even at the earliest repair point, `99.9023%` of output rows remain unlocked.
If one unlocked coordinate receives one exact dense-row repair and every
selector, certificate, state operation, and suffix is free, the repair work
still exceeds the complete `1.1827%` allowance. Decision:

```text
REJECT_OMEGA_ROUNDLOCK_WITH_ATLAS_PLUS_ONE_PAGE_PROPOSER
```

This does not reject every possible rounding firewall. It rejects this
concrete cheap proposer/row-repair construction. A replacement must provide a
materially different predictor or a non-rowwise exact repair source and must
pay for it.

### ΩBACKCUT -- decision-only backward cut

ΩBACKCUT was a second direct design: do not exactize the whole hidden/KV
state; pull back only the signed logit differences between the proposed winner
and its competitors, and certify those scalars. The cheapest proposer Gate
survives on the frozen row because both candidate and native BF16 logits choose
token `21,461`.

The information-source Gate then rejects the design as a duplicate rather than
spending an experiment. After known primal and dual components are removed,
each exact correction is the previously audited Bilinear Cross Residual
`r^T W u`. Building one prompt-dependent full-model dual costs `1.5625%` over
the favorable 64-token service life, already above the complete allowance;
the static full-vocabulary composite needs `12.720703125 GiB` for one last
down projection and a `6.765979992%` scan. The existing norm screen leaves all
`248,319` competitors unresolved.

```text
REJECT_OMEGA_BACKCUT_AS_EXISTING_DECISION_DUAL_SOURCE_NOT_A_NEW_CONSTRUCTOR
```

This is not evidence against every decision-only method. It says that a new
name and a backward certificate wrapper do not create the missing exact
answer to `r^T W u`. Authority for the charged source equation is
`E0_POST_ATLAS_CAUSAL_INFORMATION_SOURCE_AUDIT.md`.

## What this changes

These measurements do **not** prove that every exact runtime is impossible.
They do remove a broad local-shortcut family: exact rounding absorption,
coordinate history, natural runs, repeated products, opposite products, and
the low-VC/Pollard constructor cannot be the missing universal core on this
real tensor.

The next admissible proposal must introduce genuinely new exact answer
information.  Renaming a cache, dictionary, row-difference tree, Boolean
nonemptiness structure, probabilistic verifier, or full-sweep batch does not
do so.  The unresolved frontier remains a globally coupled bounded-word
constructor for the causal scalar query, or a goal-contract change.  Current
classification:

```text
NO_SURVIVING_CANDIDATE
TARGET NOT ACHIEVED
CONTINUE CONSTRUCTOR SEARCH
```

## Reproduction

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_native_exact_shortcut_frontier.py `
  --output-dir results\e0_native_exact_shortcut_frontier
```

The runner reads only the already pinned model tensor and EXP-083B evidence.
