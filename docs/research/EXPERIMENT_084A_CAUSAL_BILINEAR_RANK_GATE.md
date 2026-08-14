# EXP-084A Causal Bilinear Rank Gate

## Status and evidence boundary

- Date: 2026-08-09 Asia/Seoul
- Status: SCIENTIFICALLY REJECTED AT E1; INDEPENDENTLY REPRODUCED
- Decision: `REJECT_CAUSAL_BILINEAR_FACTOR_SPAN_LEDGER_AS_CORE`
- Evidence: one unchanged pinned Qwen3.5-0.8B last-`down_proj` population,
  exact float32-dyadic membership, three finite fields, and zero-forward replay
- Evidence ceiling: one small-checkpoint structural Gate; not an executor,
  operation replacement, physical benchmark, 122B/405B result, or E2-E7

## Frozen provenance

```text
authority commit             2fbcb3f
protected implementation     6214700d6a5b83097c043b660e4c84e7f83feae0
execution commit             9fea5522a6b3dec6ac378a12195565645e19a2fe
config SHA-256               92d619c12716966d8d3b1fd3cdd09417427a1b38ddafc5d68e2b3d20bc7cf5b3
checkpoint revision          2fc06364715b967f1860aea9cf38778875588b17
weight SHA-256               04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
prompt SHA-256               46c9779c24e59f356247145f7ef870a1b82ba4907eb9dffcd48652e519f7612a
trace SHA-256                1e921698ce8ee522c0d3cb9b8b9004139beb54cc2dee82e1a2aa0a08fa245e4f
```

Attempt 01 used batched prompt-position capture. Its first-position activation
and decision direction did not bit-match a causal-prefix replay, so it failed
closed before producing any build or evaluation query. That independently
verified control artifact is preserved at
`results/exp_084a_attempt_01_control_failure`. No scientific value from that
attempt influenced a threshold or query result.

The protected second runner uses only sequential single-token committed-prefix
prompt states. It retained the frozen side rank 16, build/evaluation prompts,
decode positions, primes, exact-hit definition, and stop thresholds. A shell
wrapper reached its one-hour limit, but the same child process remained alive
and completed without restart; the recorded runner wall time is
`6,395,350,600,200 ns`.

## Frozen query and decision

For every registered decode state, the unchanged dense target supplies the
last-layer `down_proj` input `x`, output `y`, residual branch, final hidden, and
native logits. Winner `w` and strongest different competitor `c` use the
lowest token ID on ties. Autograd replays only
`residual add -> final RMSNorm -> tied head rows {w,c}` to obtain `v`.

Prompt-only sequential states build float32 two-pass MGS bases `Q,P` of rank
16. The current decode state is excluded. The exact IEEE float32 residuals are

```text
u = x - Q(Q^T x)
r = v - P(P^T v).
```

The build ledger accepts the first 23 exact-rationally independent outer
products `r u^T` in file/position order. A held-out hit requires exact rational
coefficients at the frozen pivot coordinates and a complete factorized
identity. Modular rank nonincrease is never a hit.

## Result

```text
controls                                      114 / 114 passed
build rows                                      24 / 24
build rank modulo 65521/65519/65497             24 / 24 / 24
ledger dimension                                23
held-out rows executed                           5 / 36
exact hits / misses                              0 / 5
held-out rank lower bound after stop              5
held-out rank modulo 65521/65519/65497            5 / 5 / 5
target-future token reads                         0
stop reason                         fifth_exact_miss
```

All five observed evaluation rows were outside the frozen build ledger by an
exact rational residual coordinate. Each also raised the fixed-ledger modular
rank from 23 to 24 under all three registered primes. The fifth miss exceeded
the four-row fallback allowance, so the preregistered scientific stop fired.

The run stopped before rank 28. Therefore this result does **not** prove that
every post-hoc 23-dimensional subspace misses five rows, nor does it establish
rank 28 for the full 36-row population. It rejects the frozen automatic
calibration-built full-factor ledger and forbids rescuing it with a
rank/prompt/prime/position sweep. A materially different nonlinear, implicit,
or non-factor-scanned query code remains logically open but has no current E0
construction.

## Independent verification

The verifier performed zero Transformer forwards. It reloaded all 29 saved
factor pairs, rebuilt the first-23 exact ledger, exact coefficients and
residual coordinates, all modular trajectories, the stop decision, and the
deterministic core.

```text
verification passed             true
verified build/evaluation       24 / 5
verified exact misses           5
deterministic-core SHA-256      1e79550fb66fe050338b2eedaf069728fd959583052dee2032fdd57f5cd0a7c4
summary SHA-256                 7b0e9e1db3783648b0e3db8fd4ade4f4be269ed078c5c1177cd4e14eed61afc1
verification SHA-256            37ebacdcfa345f23375d5fa943f8e34f71a0067127fb3b176febda78b55dbb78
```

Authority is `results/exp_084a`.

## Claim boundary and next admissible work

```text
REJECT_CAUSAL_BILINEAR_FACTOR_SPAN_LEDGER_AS_CORE
DO_NOT_SWEEP_LEDGER_RANK_SIDE_RANK_PRIMES_PROMPTS_OR_POSITIONS
DO_NOT_CLAIM_FULL_HELDOUT_RANK_28
KEEP_NONLINEAR_OR_IMPLICIT_EXACT_QUERY_CODES_LOGICALLY_OPEN
KEEP_NO_SURVIVING_CANDIDATE
KEEP_E2_HARDWARE_SCALE_AND_FIXED_MISSION_UNACHIEVED
```

No Ubuntu server action, model download, GPU run, physical latency claim, or
large-model promotion follows. The next research ticket must first present a
materially different exact query representation and complete finite E0
constructor/query/state/traffic/verification/fallback equation.

## Reproduction

Canonical execution from an empty result directory:

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  experiments\exp_084a\run_experiment.py `
  --model-dir .deps\exp076-model `
  --output-dir results\exp_084a
```

Zero-forward verification:

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  experiments\exp_084a\verify_results.py `
  --model-dir .deps\exp076-model `
  --output-dir results\exp_084a `
  --write-report
```

Pre-execution validation observed `13` focused tests, `499` repository tests,
and a successful standard `scripts/run_validation.py` run.
