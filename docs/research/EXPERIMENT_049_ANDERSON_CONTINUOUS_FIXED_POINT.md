# EXP-049 — Anderson-Accelerated Continuous Block Fixed-Point Gate

## Status

```text
Implementation branch: research/exp-049-anderson-continuous-fixed-point
Gate registration: COMMITTED BEFORE REAL-CHECKPOINT RUN
Scientific result: PENDING
Phase: A/B with small-real-checkpoint falsification observation
Evidence ceiling: E1 until a complete operation-replacement generator exists
Complete real operation replacement: false
Phase D: NOT TESTED
```

## Prior failure addressed

EXP-048 proved that one exact target block pass can verify 96 correct future tokens at a logical target-stream fraction of `1/96 = 1.0416667%`, but its causal proposal sources failed:

```text
hard Jacobi p50 fraction 181.25%
partial-layer self-draft p50 committed tokens 1
partial-layer self-draft p90 fraction 2893.843%
```

EXP-049 removes the per-token draft loop. It tests whether a small fixed number of full batched target passes over continuous future-token states can generate a long exact proposal.

## Continuous causal map

For exact prefix embeddings `E(p)` and a block of future soft states `Z`:

```text
L(Z) = target_logits(E(p) concat Z)
P_i = top-k-softmax(L_i / temperature)
F(Z)_i = sum_v P_i(v) E(v)
R(Z) = F(Z) - Z
```

Logits are aligned at `prefix_length - 1 + i`. Therefore `F(Z)_0` depends only on the exact prefix and `F(Z)_i` depends causally on states before position `i`.

The target checkpoint and embedding table remain unmodified. Deployable conditions use no training, adapter, external draft model, reference continuation, or future generated token.

## Conditions

### S0 — hard synchronous Jacobi control

Start every future position from token zero. Apply four exact hard target block passes. Record proposal prefix length after target pass 1, 2, and 4. Every pass is charged. This is a single-block propagation control, distinct from EXP-048's complete-generation Jacobi accounting.

### S1 — damped continuous Picard

```text
Z_next = (1 - lambda) Z + lambda F(Z)
```

Fixed variants:

```text
last-token initialization, top-k 8, lambda 1.0
last-token initialization, top-k 8, lambda 0.5
zero initialization, top-k 8, lambda 1.0
last-token initialization, top-k 1, lambda 1.0 hard-projection control
next-token-repeat initialization, top-k 8, lambda 1.0
```

The next-token-repeat condition obtains one token from a counted first target pass, repeats its original embedding, and uses only the remaining passes up to the same total pass budget.

### S2 — bounded Anderson acceleration

Use last-token initialization, top-k 8, temperature 1.0, damping 1.0, and history sizes:

```text
m in {2, 4, 8}
```

The residual least-squares system is solved in float64 with regularization `1e-8`, coefficient clipping `[-10, 10]`, normalization to sum one, and condition limit `1e12`. NaN, Inf, singularity, excessive condition, or invalid normalization fails closed to one damped Picard update and increments the numerical fallback count.

### S3 — exact future-state oracle

Initialize `Z` from exact future token embeddings and execute one target pass. This validates inputs-embeds alignment, projection, hardening, and the retained exact verifier. It uses future information and is excluded from deployable aggregates.

### S4 — hidden triangular adversarial family

For hidden exact tokens `t_0...t_{K-1}`:

```text
F(Z)_0 = E(t_0)
F(Z)_i = E(t_i) if hard(Z_{i-1}) == t_{i-1}
         E(decoy) otherwise
```

Two models may share a resolved prefix and contain different hidden suffixes. Before the predecessor is resolved, both return the same decoy suffix, so the later exact token is indistinguishable from the black-box transcript.

## Lower-bound claim and scope

Declared interface:

- one synchronous black-box causal block evaluation per target round;
- no external future information;
- the solver may use the exact prefix, fixed initialization, all prior states, all prior target outputs, continuous arithmetic, and bounded Anderson history;
- exact hard tokens are committed only through the retained verifier.

For the hidden triangular family, a deterministic solver cannot guarantee knowledge of position `i` before the exact predecessor needed to reveal it has been resolved. An adversary can choose an unobserved suffix token among alternatives consistent with the transcript. Thus no target-only solver under this interface can guarantee more than one new exact position per target round for every arbitrary causal model.

Continuous embeddings and Anderson mixing extrapolate observed states; they do not add the hidden suffix information. The theorem is a worst-case statement. It does not assert that average real checkpoints cannot occasionally propagate farther.

## Small-checkpoint corpus

Pinned unchanged Dense checkpoints:

```text
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Tokenizer:

```text
EleutherAI/gpt-neo-125M @ 21def0189f5705e2521767faed922f1f15e7d7db
```

Held-out families are the same six used by EXP-048: English narrative, Korean, code, mathematics, structured JSON, and brittle identifier continuation.

Block sizes and total target solver-pass checkpoints:

```text
K in {64, 128, 256}
passes in {1, 2, 4}
```

States exceeding the checkpoint context limit are recorded as excluded and may not disappear silently.

## Offline favorable selection boundary

For early falsification, the runner records every fixed variant and also computes an oracle-best row per case over the pre-registered variants, block sizes, and pass checkpoints using the exact reference prefix. This selection is non-deployable and is labeled `variant_selection_uses_reference=true`.

A negative result for this favorable upper-bound selector rejects the family. A positive result would not promote a runtime until a causal fixed selector is separately committed and tested.

## Exact verification and accounting

Every selected hard proposal is executed through `verify_exact_proposal` using one exact target pass over the hardened block. Only the longest matching prefix plus the exact first-mismatch correction is committed. Predictions after the first mismatch are discarded.

Record:

```text
solver target full streams
exact verification full streams
matching proposal prefix
exact committed tokens
rejected positions
future-information label
projection logical read bytes
projection operations
Anderson history peak bytes
coefficient maximum and condition
numerical fallbacks
CPU elapsed time and peak RSS
```

Traffic-normalized metric:

```text
target_equivalent_stream_fraction =
  (solver_target_streams
   + exact_verify_streams
   + projection_read_bytes / model_parameter_bytes)
  / exact_committed_tokens
```

Projection operations and Anderson arithmetic are reported separately and may not be hidden as zero cost. Physical target-weight reuse, accelerator kernels, KV traffic, and wall-clock target speed remain UNVERIFIED.

## Pre-registered early rejection Gate

Reject target-only continuous fixed-point proposal generation as the core path if any condition holds:

```text
exact verifier mismatch >0
future information in S1/S2 >0
unhandled numerical failure >0
oracle-best S1/S2 p50 matching prefix after <=4 total target passes <16
oracle-best S1/S2 p90 target-equivalent stream fraction >10%
S2 p50 matching prefix after 4 passes <4x max(1, S0 p50 prefix)
largest-model median prefix <75% of smallest-model median prefix
hidden triangular family disproves any universal >1-position/round guarantee
```

Because the fixed mission requires an arbitrary unmodified Dense model, a valid worst-case barrier is independently sufficient to reject a universal target-only fixed-point mechanism. Empirical rows remain useful negative/average-case evidence.

Failure decision:

```text
REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
```

## Promotion Gate

A later complete Phase-C runtime would still require:

```text
zero exact mismatch
zero deployable future information
zero unhandled numerical failure
p90 target-equivalent stream fraction <=0.011851851851851851
p50 committed tokens >= ceil(total target-equivalent streams / 0.011851851851851851)
nonzero success in every held-out family
non-degrading model-size trend
a causal deployable variant selector
claim scope consistent with the lower bound
```

## Projection boundary

```text
405B Q4 full stream: 188.592821 GiB
1.2x 4B Q4 allowance: 2.235174 GiB/token
required target-equivalent stream fraction: 1.185185%
```

These values are PROJECTED from parameter counts. EXP-049 does not measure 405B, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, tokens/second, or physical block weight reuse.

## Commands

```bash
python -m pytest -q tests/exp_049
bash experiments/exp_049/run_current_env.sh
bash experiments/exp_049/reproduce.sh
```
