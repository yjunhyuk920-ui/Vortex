# EXP-087A — Cross-Weight Quadratic BF16 Residual Generator Gate

## Status

```text
FROZEN BEFORE PUBLIC-CHECKPOINT RESULT
Phase: B/C cheapest favorable-oracle falsification
Evidence ceiling: E1
Complete operation replacement: false
405B / 8 GiB / target latency: NOT TESTED
```

## Question

Can one checkpoint-static, non-affine, finite-word program generated from the
**complete SwiGLU function** reproduce unseen causal BF16 MLP outputs without
reading any gate/up/down weight page at query time?

EXP-086A closed two different sources:

1. four affine whole-SwiGLU programs had zero exact unseen output vectors and
   left a median `573/576` BF16 words wrong; and
2. global sign/exponent/mantissa successive refinement retained nearly all
   original BF16 information.

EXP-087A changes both the stored information and the runtime language. It does
not select pages, rows, ranks, or precision prefixes. It compiles the complete
BF16 output-word residual into a small **quadratic Boolean program** conditioned
on a low-width input context.

## Mechanism fingerprint

For program `j`, let `b_j` be one BF16 baseline output-word vector. A fixed
context extractor produces `d=12` bits:

```text
c_j(x) = sign((x - mean_j) Q_j) in GF(2)^12.
```

The compiler constructs the quadratic feature vector

```text
phi(c) = [1, c_1, ..., c_d, c_1 c_2, ..., c_(d-1) c_d],
T = 1 + d + d(d-1)/2 = 79.
```

For each output coordinate and each of its 16 BF16 bits, a GF(2) polynomial is
stored. At query time:

```text
residual_word_j(x) = XOR(t : phi_t(c_j(x))=1, coefficient_word[j,t])
candidate_word_j(x) = baseline_word_j XOR residual_word_j(x).
```

The candidate BF16 tensor is reconstructed directly from those 16-bit words.
There is no floating approximation after the word generator: a correct word is
bit-identical to the official BF16 output and an incorrect word is a measured
failure.

The compiler uses four programs. Build states are deterministically partitioned
into four equal populations by a frozen principal-direction ordering. Each
program chooses 12 context hyperplanes from a fixed pool of 64 by greedily
maximizing the GF(2) rank of the complete quadratic feature matrix. The
coefficient system is solved exactly over GF(2); no optimizer, tolerance, or
floating loss is used.

## What is materially new

- **Not affine:** the stored output residual is a degree-two Boolean function of
  the runtime context.
- **Not numeric low rank:** no `WQ`, output basis, singular-value truncation, or
  row repair is used.
- **Not bit-prefix refinement:** coefficients generate all 16 BF16 residual bits
  jointly; no native weight mantissa prefix is retained at query time.
- **Not page separable:** the target labels come from the complete composed
  `down(SiLU(gate(x))*up(x))` output, after every signed cross-page
  contribution and reference rounding.
- **Not a state table:** a program receives only 12 predicate bits and must be
  reused across many build and unseen states. Complete prefixes, KV digests,
  state hashes, and per-state leaves are forbidden.

This Gate tests whether a compact nonlinear finite-word language supplies the
new exact answer information missing from the rejected affine and precision
sources. It does not claim that every nonlinear compiler is covered.

## Frozen public checkpoint

```text
model        HuggingFaceTB/SmolLM2-135M
revision     93efa2f097d58c2a74874c7e644dbc9b0cee75a2
loader       transformers.LlamaForCausalLM.from_pretrained
reference    official eager BF16 CPU path
layer        0 complete MLP
```

The model weights remain unchanged.

## Frozen causal population

Six prompt families are used:

```text
English
Korean
code
structured JSON
mathematics
repeated pattern
```

Each family has one build prompt and one disjoint evaluation prompt. Every
prompt contributes 16 consecutive greedy causal positions. Therefore:

```text
build states       6 * 16 = 96
evaluation states  6 * 16 = 96
```

The exact MLP input and output at the last sequence position are captured by
hooks on the official model. No evaluation output, token, activation, or
residual is visible to compilation.

## Oracle and deployable selectors

### Favorable oracle

For each held-out state, evaluate all four already-compiled programs and grant
the program with the most exact BF16 output words. The oracle sees the target
output only for selection after every candidate has been produced.

This is non-deployable and can only establish an upper bound. If it cannot
produce the entire output vector exactly, no causal router among the same four
programs can rescue the language.

### Causal router

Choose the nearest stored normalized input centroid. It sees only the current
MLP input and checkpoint-static sidecar.

## Exactness and integrity controls

- BF16 word conversion must round-trip every registered control word exactly.
- The independent synthetic GF(2) quadratic generator must be recovered and
  replayed with zero bit mismatch.
- Every emitted program must replay all of its assigned build outputs exactly.
- Hashes cover input means, context bases, baselines, packed coefficients,
  assignments, prompts, config, and result core.
- A malformed dtype, dimension, nonfinite input, inconsistent GF(2) system, or
  checkpoint identity fails closed.
- Candidate mismatch is a scientific failure, never approximate acceptance or
  dense fallback.

## Favorable target-scale resource equation

For context width `d=12`, feature count `T=79`, hidden width `H=16,384`,
programs `K=4`, and layers `L=126`, the checkpoint-static sidecar charges:

```text
input mean             4H bytes / program / layer
context basis          4Hd
router centroid        4H
BF16 baseline          2H
quadratic coefficients 2TH
```

Thus:

```text
M_sidecar = K L H (4 + 4d + 4 + 2 + 2T)
          = 1,783,627,776 bytes
          ≈ 1.6611328125 GiB.
```

The favorable online instruction ledger charges:

```text
nearest-centroid router      3 K H scalar operations
context projection           2 H d
quadratic feature creation   d(d-1)/2 bit operations
packed coefficient replay    T * (16H/64) 64-bit XORs
word assembly                2H operations
```

This is approximately `0.036%` of one target-shape dense SwiGLU MLP. It is a
favorable ISA-equivalent screen, not a wall-clock claim.

Even a perfect MLP compiler leaves all non-MLP operations. The result must
therefore report the explicit whole-model fraction when only MLPs are replaced;
it may not promote the component ratio into final completion.

The compiler consumes 96 exact build transitions. Its minimum service length to
amortize those target calls below the p50 target fraction is:

```text
ceil(96 / (1.2*4/405)) = 8,100 committed tokens.
```

This preprocessing cost is reported rather than hidden. A pass would require a
separate checkpoint-wide compiler and deployment-amortization Gate.

## Frozen Gates

### Integrity Gate

```text
synthetic GF(2) quadratic mismatch = 0
all hashes, dimensions and finite checks pass
```

Failure:

```text
INVALID_QUADRATIC_BF16_RESIDUAL_GENERATOR_CONTROL_FAILURE
```

### Build-language Gate

```text
all four assigned build populations vector-exact = true
build oracle vector-exact fraction = 100%
```

Failure:

```text
REJECT_QUADRATIC_BF16_RESIDUAL_GENERATOR_BUILD_GATE
```

### Resource Gate

```text
projected complete sidecar <= 4 GiB
favorable compiled-MLP operation fraction <= 1.185185185%
```

Failure:

```text
REJECT_QUADRATIC_BF16_RESIDUAL_GENERATOR_RESOURCE_GATE
```

### Unseen favorable-oracle Gate

```text
evaluation oracle full-vector BF16 exact fraction = 100%
every used program serves at least 4 held-out states
no fallback
```

Failure:

```text
REJECT_QUADRATIC_BF16_RESIDUAL_GENERATOR_AT_ORACLE_GATE
```

### Causal-router Gate

```text
evaluation causal-router full-vector exact fraction = 100%
```

Oracle pass and router failure:

```text
PROMOTE_QUADRATIC_BF16_RESIDUAL_LIBRARY_TO_CAUSAL_ROUTER_GATE
```

Complete component promotion:

```text
PROMOTE_QUADRATIC_BF16_RESIDUAL_GENERATOR_TO_COMPLETE_MLP_GATE
```

A promotion authorizes only an actual complete-MLP replacement with successor
state and physical accounting. It does not establish 405B, 8 GiB, or 4B-class
latency.

## Stop rule

On oracle rejection, do not rescue this fingerprint with:

```text
context width 13/16/32
polynomial degree 3 or 4
more programs
new clustering
prompt/layer selection
state hashes or exact prefix keys
per-coordinate fallback
mixed precision or tolerance
post-result bit ordering
```

Those are capacity sweeps around the same finite-context polynomial language.
Reopening requires a materially different exact information source, such as an
online cross-state recurrence with proved successor-state closure or a
checkpoint-derived non-polynomial transition program.

## Claim boundary

```text
actual public checkpoint and causal states  TO BE MEASURED
synthetic finite-word correctness           TO BE CONTROL-TESTED
held-out exact program coverage             TO BE MEASURED
complete real MLP replacement               NOT TESTED
complete Transformer layer                  NOT TESTED
successor KV/state                           NOT TESTED
CUDA / PCIe / SSD / latency                  NOT TESTED
405B under 8 GiB                             NOT TESTED
E6 / E7                                      NOT ACHIEVED
```
