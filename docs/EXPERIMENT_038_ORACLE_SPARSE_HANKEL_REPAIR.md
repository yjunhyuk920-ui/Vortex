# Experiment 038 — Oracle Sparse Repair for Hankel Decision Programs

Last updated: 2026-08-03 (Asia/Seoul)

## Fixed objective

The project target remains an arbitrary unmodified 405B-class dense Hugging Face model on one 8 GiB VRAM GPU, original-model quality, no user-authored training/adapters, and warm decode comparable to a native 4B Q4 model.

This experiment is an optimistic E2 repair-rate lower bound. It does not provide a deployable detector.

## Failure addressed

Experiment 037 compiled prompt-only controlled Hankel programs with tiny 405B memory and compute projections, but the longest autonomous exact continuation was only two tokens. The remaining possibility is sparse exact repair: use the cheap recurrence most of the time and invoke the original model only when the program would choose a wrong token.

The critical question is not whether repair can restore correctness; exact computation obviously can. The question is whether even a perfect repair oracle can keep exact target calls rare enough to amortize their physical bytes and arithmetic.

## Optimistic oracle contract

For each future step `t`, the program predicts a reduced state and token using the exact previous output token as control:

```text
z_hat_t = F(z_history, exact_token_(t-1))
pred_t  = argmax(W_program z_hat_t + b_program)
```

The evaluation oracle compares `pred_t` with the exact target token.

- If equal and all values are finite, accept `z_hat_t` without a target repair.
- If unequal or non-finite, charge one exact target interaction, output the exact token, and replace `z_hat_t` with the projection of the exact final hidden state.

The repaired state enters the recurrence history. Thus every emitted token remains exact by construction.

This oracle is strictly stronger than any deployable causal detector because it sees the exact target token before deciding whether to repair. Its measured repair rate is therefore a lower bound on real exact-target traffic and compute.

## Repair amortization

One optimistic 405B exact interaction at 4-bit-equivalent source traffic costs:

```text
B_repair = 405,849,243,648 * 4 / 8
         = 188.9883 GiB

C_repair = 2 * 405,849,243,648
         = 811.6985 GFLOP
```

For `R` repairs over `T` warm-decode tokens:

```text
A_repair = T / max(R, 1)
B_repair/token = R * B_repair / T
C_repair/token = R * C_repair / T
```

The project retains the strong requirement:

```text
A_repair >= 247 tokens
```

For a 256-token trace this means at most one exact repair. A real sound detector, metadata, synchronization, and repair latency would make the requirement stricter, not easier.

The report also records direct envelope checks:

```text
program hot compute + program build / 256 + repair compute <= 9.6 GFLOP/token
repair traffic <= 2.4 GiB/token
```

## Real-model protocol

Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

Prompts and recurrence configurations are identical to Experiment 037 so the only changed mechanism is exact repair.

For every prompt/configuration:

1. collect exact prompt hidden states and 256 exact future hidden states/tokens;
2. compile the recurrence only from prompt states/tokens;
3. start from exact prompt history and the exact first generated token;
4. predict every future state with exact previous-token control;
5. repair exactly on token mismatch or numerical non-finiteness;
6. record repair positions, accepted run lengths, mean/minimum/maximum interval, repair traffic, repair compute, and hidden drift before every decision.

## Configurations

```text
rank=8,  control=8,  order=1, lift=linear
rank=16, control=8,  order=2, lift=linear
rank=16, control=16, order=2, lift=full
rank=32, control=16, order=2, lift=full
rank=32, control=16, order=4, lift=full
rank=64, control=16, order=2, lift=full
```

## Promotion thresholds

The same configuration must satisfy every prompt:

```text
repairs <= 1 over 256 tokens
mean repair interval >= 247 tokens
all 256 emitted tokens exact after repair
repair traffic <= 2.4 GiB/token
program hot + build/256 + repair compute <= 9.6 GFLOP/token
no uncharged non-finite state
```

A pass would only justify building a sound pre-repair certificate. It would not itself solve the target.

## Rejection rule

Reject recurrence-plus-sparse-repair if every configuration needs two or more repairs on any prompt, or if direct 405B traffic/compute exceeds the envelope.

Do not respond by making the detector less conservative: this experiment already uses the impossible perfect token oracle and therefore gives the minimum repair count for the tested recurrence family.

## Durable evidence

```text
vortex_runtime/oracle_hankel_repair.py
scripts/run_oracle_hankel_repair.py
tests/test_oracle_hankel_repair.py
results/tinyllama_1_1b_oracle_repair_<prompt>.json
results/tinyllama_1_1b_oracle_repair_frontier.json
.github/workflows/oracle-hankel-repair.yml
```
