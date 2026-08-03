# Experiment 040 — Exact Dense-Operator Information and Traffic Lower Bound

Last updated: 2026-08-03 (Asia/Seoul)

## Evidence level and purpose

This is an E1 formal/executable lower-bound gate. It is not E4 and does not itself run a 405B model.

The fixed objective remains:

- arbitrary unmodified Hugging Face dense checkpoint;
- one 8 GiB VRAM GPU;
- no user training, distillation, fine-tuning, or model-specific adapter work;
- original-model decisions and quality;
- p50 warm decode within 1.2x a native 4B Q4 baseline;
- real 405B-class flagship validation.

Experiments 036–039 exhausted prompt-derived static, dynamic, repaired, and nonlocal exact decision programs. Experiment 040 now tests the worst-case information that any universal exact dense operator must retain or read.

## Claim A — lossless exact-output information

Let a dense checkpoint contain `N` independently selectable `b`-bit parameter codes. The checkpoint set has cardinality:

```text
|C| = 2^(N b)
```

Any checkpoint-specific representation from which every exact operator output can be recovered for every input must be injective over `C`. Therefore its worst-case length is at least:

```text
I_exact_output >= N b bits
```

This includes resident weights, compressed checkpoint-specific metadata, generated code, tables, or any other exact representation. Lossless compression may redistribute these bits but cannot guarantee fewer than `N b` bits for every arbitrary checkpoint.

For the 405B target specification used by the repository:

```text
N = 405,849,243,648 parameters
b = 4 bits
I = 1,623,396,974,592 bits
  = 188.98828125 GiB
```

Even granting all 8 GiB VRAM to checkpoint information and charging no KV cache or workspace:

```text
minimum nonresident exact information
>= 188.98828125 - 8
 = 180.98828125 GiB
```

If the interaction needs arbitrary exact operator output and no cross-token program amortizes that information, this nonresident information must be made available through reads or equivalent computation for the interaction.

## Claim B — skipped-coordinate adversary

For a dense affine operator:

```text
y = W x
```

assume a runtime's transcript does not depend on one coordinate `W[i,j]`: the coordinate is neither inspected nor represented by checkpoint-specific metadata available to the runtime.

Choose `x = e_j` and another output row `c != i`. Construct two checkpoints that are identical everywhere except `W[i,j]`:

```text
W0[c,j] = 1
W0[i,j] = 0
W1[c,j] = 1
W1[i,j] = 2
```

Then:

```text
runtime_observation(W0) = runtime_observation(W1)
W0 x != W1 x
argmax(W0 x) = c
argmax(W1 x) = i
```

Thus any unrepresented coordinate can be decision-relevant for some arbitrary checkpoint and input. In a coordinate-query execution model, an exact universal algorithm cannot safely terminate while an unrepresented coordinate remains unread.

The executable gate constructs this adversary for every possible single skipped coordinate in small dense matrices, verifies identical observations, verifies different exact outputs, and verifies different exact top-1 winners.

## What Claim B does and does not prove

Claim B proves that no coordinate may be declared universally irrelevant merely because it was not useful on previous prompts. It also proves a worst-case all-coordinate query lower bound when checkpoint-specific metadata does not encode skipped coordinates.

It does **not yet** prove that exact top-1 behavior alone requires all `N b` checkpoint bits under every possible metadata scheme. Claim A's full `N b` bound is for exact operator output or full lossless checkpoint recovery. The workflow must report these conclusions separately and must not silently promote the exact-output theorem into a stronger top-1-only theorem.

## Compute lower bound proxy

A dense matrix-vector product using every parameter requires at least one multiply and one accumulation per parameter, giving the optimistic arithmetic proxy:

```text
C_dense >= 2 N FLOP
```

For `N = 405,849,243,648`:

```text
C_dense >= 811.698487296 GFLOP
```

For a 4B dense baseline:

```text
C_4B >= 8 GFLOP
ratio >= 101.4623x
```

This is an arithmetic lower-bound proxy, not a measured GPU latency result. Kernel fusion and hardware scheduling cannot reduce the exact scalar information dependency below the operator's required contributions, although they can change achieved wall clock.

## 405B Gate equations

For exact-output universality:

```text
I_resident + I_external_available >= N b
I_resident <= 8 GiB * 8 bits/byte
I_external_available >= 180.98828125 GiB at Q4
```

For the coordinate-query worst case:

```text
Q_read + Q_represented >= N decision-relevant coordinates
```

For exact dense arithmetic without a proven reusable exact program:

```text
C_interaction >= 2 N FLOP
```

The previously established strong amortization requirement remains about 247 future tokens per full target interaction. Experiments 036–039 measured far less reuse, including a future-aware global suffix oracle maximum of 75, 28, and 5 tokens.

## Executable falsification gate

The branch must implement:

1. `construct_skipped_coordinate_adversary` for arbitrary small matrix shapes and any skipped coordinate;
2. an observation function that returns only inspected coordinates;
3. exact output and exact top-1 comparison for `W0/W1`;
4. exhaustive enumeration showing every single skipped coordinate can flip a winner;
5. exact checkpoint information, resident fraction, external-information, and dense-compute budgets for Q4/Q8/FP16;
6. strict machine-readable separation between:
   - proven exact-output information bound;
   - proven coordinate-query top-1 adversary;
   - not-yet-proven metadata-aware top-1 bit bound.

## Promotion and interpretation

The lower-bound certificate passes when:

- every generated `W0/W1` pair has identical inspected observations;
- exact outputs differ;
- exact top-1 winners differ;
- exhaustive single-coordinate coverage is 100%;
- Q4 exact checkpoint information exceeds 8 GiB;
- reported arithmetic and byte equations reproduce the committed constants;
- no conclusion conflates exact output with top-1-only information complexity.

A passing certificate means:

> The four fixed requirements are contradicted in the arbitrary dense **exact-output** worst case unless full checkpoint information is resident, read, or equivalently represented and its cost amortized. For exact top-1 only, every skipped coordinate is potentially adversarially relevant, but a fully general metadata-aware bit lower bound remains a separate proof obligation.

This does not rule out:

- exploiting measured structure in a restricted checkpoint family;
- declared approximation with bounded quality loss;
- changing hardware/resident memory;
- accepting slower wall clock;
- automatic model-specific first-run compilation whose full exact information and construction costs are charged.

It does rule out claiming universal exact dense execution by simply omitting arbitrary unrepresented weights.

## Next step after this gate

If the certificate passes, the repository must explicitly classify the original target:

1. mathematically incompatible in the arbitrary dense exact-output worst case;
2. still open for exact top-1 under a checkpoint-specific compressed decision representation, but requiring a new metadata-aware information bound or a constructive representation;
3. empirically approachable only after narrowing universality to structured released checkpoints or relaxing exactness/latency/memory.
