# EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit

## Status

Pre-registered implementation branch. No result is authoritative until a workflow run completes, raw artifacts are inspected, checkpoint revisions are frozen, and durable state is updated.

```text
Phase: A/B with small-real-checkpoint current-state observation
Evidence ceiling: E1
Real operation replacement: false
Phase D: NOT TESTED
```

## Previous failure addressed

EXP-047 CPTC-v1 used one broad contribution range and evaluated almost the entire tile population. The unresolved question is whether this failure came from loose metadata or from an intrinsic limitation of range-only finite-population certification.

This experiment does not tune CPTC-v1. It gives the family its strongest favorable oracle test first.

## Exact hypothesis

For current-token LM-head top-1-versus-runner-up margins from unmodified trained small dense checkpoints, an exact per-state min/max contribution range can certify the sign after reading at most:

```text
median evaluated fraction <=10%
p90 evaluated fraction <=25%
wrong accepts = 0
```

If the non-deployable exact oracle misses either fraction threshold, range-based CPTC is rejected as a core execution path. C2 or C3 may not rescue a family whose exact range oracle already fails.

## Operation and decision audited

For final hidden state `h`, exact LM-head rows `w_top` and `w_runner`, bias margin `b`, and input-dimension tiles `T_i`:

```text
margin = logit_top - logit_runner
       = b + sum_i c_i
c_i    = sum_{j in T_i} (w_top,j - w_runner,j) h_j
```

The audit certifies only this pairwise sign. Top-1 and runner-up are identified from the fully evaluated reference logits. Therefore this is an offline oracle audit, not a deployable top-1 selector and not E2.

No future generated token is used.

## Compared conditions

### C0 — global checkpoint-derived range

For each hidden dimension, compile the checkpoint output-weight column span:

```text
s_j = max_o W[o,j] - min_o W[o,j]
```

For tile `T_i`:

```text
|c_i| <= B_i = sum_{j in T_i} |h_j| s_j
```

C0 uses one global range `[-max_i B_i, +max_i B_i]`.

### C1 — exact per-state oracle range

C1 uses the exact minimum and maximum of all materialized `c_i`. This consumes the full contribution population and is explicitly non-deployable. It is the favorable upper bound used for the primary rejection Gate.

### C2 — checkpoint-span stratified range

C2 groups tiles by the magnitude of `B_i`, samples without replacement inside each stratum, and sums per-stratum Serfling intervals. Error probability is union-accounted across both stratum and adaptive sample count:

```text
delta_s   = delta * 6 / (pi^2 (s+1)^2)
delta_s,n = delta_s * 6 / (pi^2 n^2)
```

The bound metadata uses only checkpoint column spans and the current hidden state, not the selected skipped output rows. The present implementation is still offline because exact contributions and reference logits are materialized for validation.

### C3 — variance-adaptive bound

`NOT IMPLEMENTED` in this branch. C3 is forbidden until a separate independent finite-population proof, reference calculator, property tests, and fault injection are committed. An unverified empirical-Bernstein formula may not influence the Gate.

## Checkpoints and prompts

Candidate trained Dense checkpoints:

```text
roneneldan/TinyStories-1M
roneneldan/TinyStories-3M
roneneldan/TinyStories-8M
```

The workflow resolves each Hugging Face revision SHA before downloading or executing, loads only that SHA, and saves a file SHA-256 manifest. The first successful run is discovery evidence; exact revisions must then be frozen into the authoritative reproduction state.

Held-out prompts are fixed in `experiments/exp_047r/config.json`. No prompt-derived calibration is used.

## Correctness contract

- reconstruct exact current-token logits from the returned final hidden state and output embedding;
- reconstruct exact top-1/runner-up margin from tile contributions;
- validate every C0/C2 bound against materialized contributions in the offline audit;
- causal random sampling without replacement;
- union-accounted probabilistic intervals;
- zero wrong certified accepts in the committed corpus;
- invalid numbers, malformed strata, uncovered contributions, or reconstruction mismatch abort the run;
- failed certification evaluates the complete contribution population before returning the exact sign.

Probabilistic certification is not deterministic exactness.

## Pre-registered rejection Gate

Reject range-based CPTC from the core path if any condition holds:

```text
C1 oracle median evaluated fraction >10%
C1 oracle p90 evaluated fraction >25%
wrong certified accept >0
sound checkpoint-derived bound violation >0
C2 materialized-contribution CPU selector/fallback median cost > full materialized sum
```

The timing comparison is only a CPU primitive measurement after contributions have already been materialized. It is not LM-head, GPU, PCIe, SSD, or 405B timing.

## 405B projection boundary

From the already committed same-bit parameter-count projection:

```text
405B Q4 full stream: 188.593 GiB/token
1.2x 4B Q4 allowance: 2.235 GiB/token
required average evaluated weight fraction before overhead: about 1.185%
```

These values remain PROJECTED. This experiment cannot measure target traffic or speed.

## Strongest counterexample

The exact per-state C1 range is the strongest range-only favorable control because no sound global or static range can be tighter than the realized exact minimum and maximum for that state. If C1 still requires high coverage, further range metadata tuning cannot close the core gap.

## Commands

```bash
python -m pytest -q tests/exp_047r
bash experiments/exp_047r/run_current_env.sh
```

## Result decision vocabulary

```text
CONTINUE_TO_INDEPENDENT_C3_AND_REAL_OPERATION_REPLACEMENT
REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY
INFRASTRUCTURE FAILURE — NO SCIENTIFIC DECISION
```
