# EXP-091A — Exact Jacobi Fixed-Point Gate

## Status

```text
FROZEN BEFORE PUBLIC-CHECKPOINT RESULT
Phase: C cheapest decisive joint-token Gate
Evidence ceiling: E3 causal held-out exact-prefix signal
TARGET-W execution / target hardware / 4B-class latency: NOT TESTED
```

## 1. Why this mechanism is different

EXP-090A generated tokens autoregressively from one shallow layer. It failed on the first candidate in every build and holdout case, and the exact token was outside the top 16 in most positions. Increasing shallow depth, history length, residual order, or candidate rank would tune the same missing-suffix source.

EXP-091A removes the shallow surrogate entirely. It applies the unchanged target checkpoint to a complete guessed token block and solves the causal token equations as a parallel finite fixed-point problem.

No training, distillation, auxiliary model, checkpoint modification, target-output table, or hidden remote compute is allowed.

## 2. Exact block equation

Let `S` be the exact committed prefix state and let `y0` be the exact boundary token already emitted by the preceding exact block. The unknown future block is

```text
g = (g1, ..., gK),  K = 128.
```

Construct the target input block

```text
x(g) = (y0, g1, ..., g(K-1)).
```

One unchanged target-model block sweep produces logits at all `K` positions. With the frozen deterministic token rule, define

```text
J_W,S(g) = argmax_token F_W,S(x(g))
         = (p1, ..., pK).
```

A full fixed point satisfies

```text
g = J_W,S(g).
```

Because the target graph is causal and the token tie rule is deterministic, a full fixed point is the unique greedy autoregressive continuation for that block.

## 3. Exact prefix release theorem

Define

```text
A(g) = max L such that g[1:L] = J_W,S(g)[1:L].
```

Then the first `A(g)` guessed tokens equal the original autoregressive tokens exactly.

Proof by induction:

1. `p1` depends only on committed state `S` and boundary token `y0`; if `g1=p1`, then `g1` is the exact first continuation token.
2. Assume `g1...g(i-1)` are exact. Causality makes `pi` the official next-token decision conditioned on the exact committed prefix plus those exact tokens. If `gi=pi`, then `gi` is exact.
3. Repeat to `L=A(g)`.

Therefore an online executor may commit the self-consistent prefix and the corresponding official cache from the same block sweep. It does not need a second dense verification pass for those positions.

The public-checkpoint runner additionally generates the ordinary autoregressive target only after every Jacobi trajectory is complete. This is an integrity control, not an online dependency.

## 4. Jacobi trajectory

For seed `g^(0)`:

```text
g^(r+1) = J_W,S(g^(r)),  r = 0,...,7.
```

For each sweep the runner records:

```text
self_consistent_prefix_r = A(g^(r))
changed_positions_r
full_fixed_point_r
input/proposal hashes
input/proposal agreement with the delayed AR target
processed block positions
```

The block sweep uses the official public checkpoint, official causal mask, official BF16/eager ABI, and a fresh clone of the exact prompt cache. No approximate layer or custom Transformer forward is the reference.

## 5. Frozen causal seeds

The compiler evaluates exactly three deterministic checkpoint-independent seeds.

### `boundary_repeat`

```text
g_i = y0
```

### `prompt_suffix_cycle_16`

Cycle the last at most 16 committed prompt token IDs until 128 guesses are filled. The exact boundary token is not inserted into the source cycle.

### `prompt_longest_ngram_8`

Build a read-only successor database from the committed prompt plus `y0`. Generate the seed left-to-right. At each position, find the longest current suffix of order at most 8 that occurred in the committed database with a successor. Use the most recent matching occurrence; on a tie use the later occurrence. If no suffix has a successor, emit `y0`.

Generated guesses may form the lookup query but are never added to the successor database. Thus the seed cannot learn from target results or from its own future.

No seed, order, cycle width, block length, iteration cap, or prompt is changed after results.

## 6. Build selection and untouched holdout

All three seed trajectories are completed and hashed for the three build cases before the ordinary AR target is generated. The selected seed maximizes lexicographically:

```text
minimum first-sweep exact-release length
sum first-sweep exact-release lengths
minimum best exact-release length over eight sweeps
sum best exact-release lengths
minimum fixed-point sweep count
frozen seed-order preference
```

After selection, one seed is frozen. Only that seed is run on the three untouched holdout cases, and every holdout trajectory is complete before its AR target control begins.

## 7. Traffic equation

Let a target checkpoint sweep commit `A` exact tokens. With lossless checkpoint compression ratio `C>=1`, the most favorable logical checkpoint weight fraction is

```text
rho_W = 1 / (A*C).
```

The registered p50 whole-model fraction is

```text
f_target = 0.011851851851851851.
```

Therefore one sweep requires:

```text
A >= ceil(1/f_target) = 85                         # no compression
A >= ceil(1/(1.261972*f_target)) = 67             # favorable DEV-W ratio
```

At `K=128`, two or more complete sweeps cannot meet either threshold even if the second sweep fixes the entire block:

```text
2/128 = 1.5625% > 1.185185185%
2/(128*1.261972) ~= 1.2381% > 1.185185185%.
```

Consequently only the first sweep can promote this mechanism toward the fixed target. Later iterations are measured to characterize the fixed-point dynamics and to prevent an incorrect broad claim, not to hide extra checkpoint reads.

The sweep still performs the target arithmetic for 128 positions. Block GEMM throughput, KV traffic, workspace, and physical latency remain separate required costs.

## 8. Integrity controls

The run is invalid unless all controls pass:

- every seed is deterministic and length 128;
- n-gram lookup reads only the committed database;
- stable argmax uses the smallest token ID on ties;
- changing a future guess cannot change an earlier toy-map proposal;
- on an exhaustive triangular toy map, every self-consistent prefix equals the autoregressive target;
- a one-token-per-sweep positive control reaches its fixed point in the expected number of iterations;
- required-token equations evaluate to 85 and 67;
- every public-model self-consistent prefix equals the delayed AR target prefix;
- every public-model full fixed point, if observed, equals the complete delayed AR target;
- every build/holdout trajectory is complete before its AR target control starts;
- holdout seed selection precedes every holdout target control;
- build and holdout prompts are disjoint;
- checkpoint and runtime pins match exactly.

## 9. Decisions

### No-compression promotion

```text
PROMOTE_ONE_SWEEP_JACOBI_PREFIX_TO_EXACT_BLOCK_EXECUTOR_GATE
```

requires:

- zero integrity failures;
- the selected seed releases at least 85 exact tokens on the first sweep in every untouched holdout case.

### Favorable lossless-compression promotion

```text
PROMOTE_COMPRESSED_ONE_SWEEP_JACOBI_PREFIX_TO_BLOCK_EXECUTOR_GATE
```

requires the no-compression Gate to fail but:

- zero integrity failures;
- the selected seed releases at least 67 exact tokens on the first sweep in every untouched holdout case.

This promotion remains conditional on reproducing a lossless target-checkpoint compression ratio at least `1.261972` and charging decompression.

### Rejection

Otherwise:

```text
REJECT_NATIVE_JACOBI_FIXED_POINT_AS_405B_CORE
```

## 10. Stop rule

After rejection, do not sweep:

- seed token or seed family;
- n-gram order;
- cycle width;
- block length;
- iteration count;
- prompt subset;
- argmax tie rule;
- acceptance threshold;
- shallow-draft initialization from EXP-090A;
- learned consistency refinement, because retraining violates the fixed mission.

Reopening requires a materially new source that changes first-sweep exact-release information, such as a checkpoint-static exact certificate, a globally shared nonseparable token solver, or a proof that a different unchanged-model block operator can commit at least 85 exact tokens per checkpoint sweep.

## 11. Claim boundary

A pass authorizes only an existing-ISA streamed block executor with complete resource traces. It does not establish TARGET-W acceptance, 8-GiB residency, maximum-context KV handling, target SSD/PCIe/HBM traffic, arithmetic throughput, or same-machine 4B-class p50/p95.
