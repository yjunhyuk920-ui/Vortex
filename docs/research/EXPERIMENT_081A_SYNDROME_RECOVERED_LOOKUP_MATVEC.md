# EXP-081A -- Syndrome-Recovered Lookup MatVec Gate

## Status

Preregistered Phase A/B/C E1 prototype, with the pre-measurement topology
repair recorded below. No EXP-081A scientific result existed before either
registration commit.

## Pre-measurement topology repair

The first canonical command stopped during module resolution, before any
prompt forward pass or evidence write. Registered layer `0` is a
`linear_attention` layer in the pinned checkpoint and therefore has
`in_proj_qkv`, not the registered `q_proj`. The layer tuple is amended from
`0, 11, 23` to `3, 11, 23`, where layer `3` is the earliest full-attention
layer and all three layers expose the same `q_proj` operation. No activation,
residual, or metric was observed before this data-independent repair; the
failed command is an infrastructure event, not a scientific result or a layer
sweep.

## Question

Can a nonlinear lookup generator, a small learned residual code, and an
independent algebraic fingerprint jointly replace an online dense `W @ x`
while retaining exact fail-closed semantics and the final 405B/4B-equivalent
resource ceiling?

This is throwaway prototype research, not a production runtime or a claim that
an established complexity conjecture has been broken.

## New data structure

Work first over a prime field `F_p`. For fixed `W[m,n]`, let `G(x)` be a cheap
nonlinear lookup candidate and let `D[m,r]` be a residual dictionary. Choose a
recovery sketch `R[r,m]` such that `R D` is invertible, and choose an independent
uniform verification sketch `V[q,m]`. Precompute:

```text
A = R W
B = (R D)^-1
C = V W
```

For a causal query `x`:

```text
g = G(x)
s = A x - R g
a = B s
y = g + D a
accept y iff C x == V y
otherwise compute the original W x
```

If `e = W x - G(x)` lies in `col(D)`, the syndrome recovers it exactly. If it
does not, the corrected error is nonzero and `q` independent fingerprint rows
accept it with probability at most `p^-q`, provided `V` is independent of the
generator, dictionary, and recovery sketch. The fallback is the unchanged
operation. This gives a conditional query cost

```text
T_G + O(rn + 2rm + r^2 + q(n+m))
```

instead of `O(mn)`. Worst-case cost remains dense, so this does not contradict
OMv or succinct MatVec lower bounds. The only possible breakthrough is that
real causal Transformer residuals may occupy a tiny, repeatedly recoverable
code family with population coverage above the fallback threshold.

## Nonlinear candidate generator

The registered generator is an eight-stage vector regression forest. Each
stage is a depth-four tree and contributes one of sixteen stored output pages.
Stages are fitted sequentially to the preceding output residual on build-only
activations. A query performs 32 scalar comparisons and sums eight selected
pages. Pages are target-accounted as signed int8 plus per-page scale metadata.

This generator is related to MADDNESS-style learned lookup AMM, but the proposed
claim is only the composition with syndrome recovery, independent fingerprint
verification, and exact fallback. Regression trees, residual dictionaries,
random sketches, and Freivalds-style fingerprints are not claimed as new in
isolation.

## Difference from closed Vortex families

- EXP-069 required a new input to lie exactly in a prior linear span. EXP-081A
  first applies a nonlinear code-cell generator and asks whether its **output
  error** lies in a small residual code.
- EXP-078A reused one frozen prior-token operator. The lookup route evaluates
  the current activation and has no frozen causal coefficient.
- EXP-079A bounded unread residuals but could not reconstruct them. EXP-081A
  obtains query-time syndromes of the complete residual and attempts algebraic
  recovery before independent verification.
- EXP-080A required a future token block. EXP-081A is single-token causal.

No future generated token, evaluation output, or evaluation residual may train
the generator or dictionary.

## Registered field and soundness

Synthetic exact controls use `p=2,147,483,647`, recovery rank `r=8`, and
`q=6` independent verification rows. For at most `10^12` verified projection
queries, the registered union upper bound is:

```text
10^12 / p^6 < 1.3e-44
```

This is probabilistic certification, not deterministic exactness. A malformed,
singular, nonfinite, or unverifiable state must run the original multiplication
or abort; it may never silently accept.

## Frozen 405B resource equation

The shape population is the eight non-embedding families in
`results/exp_071/raw/tensor_rows.jsonl`, totaling `403,747,897,344` coefficient
uses per logical token. For each matrix instance:

```text
recovery operations = r*n + 2*r*m + r^2 + q*(n+m)
generator operations = C*m + 4*C
recovery bytes = 4 * (r*(n+m) + q*n + r^2)
selected-page bytes = C*m
lookup storage = C*K*m bytes
```

with `r=8`, `q=6`, `C=8`, and `K=16`. Procedural `R,V`, small tree metadata,
scales, and the original cold matrix are still required; metadata/scales are
charged by the runner rather than silently omitted.

Frozen favorable aggregate before fallback:

```text
operation fraction             0.2148842%
logical traffic fraction       0.9265250%
sidecar storage                3.976903 GiB plus small metadata
p50 final allowance            1.1851852%
remaining traffic for fallback 0.2586602%
required weighted fast coverage >=99.7413398%
```

The sidecar is not a measured VRAM result. A pass only authorizes a minimal
integer operation-replacement Gate.

## Synthetic controls

The embedded reference must demonstrate:

- exact recovery when the constructed residual lies in `col(D)`;
- rejection and exact fallback for an out-of-code residual;
- rejection after candidate, syndrome, dictionary, or fingerprint fault
  injection;
- deterministic replay and agreement with direct finite-field `W @ x`;
- zero wrong accepts in registered randomized adversarial trials;
- singular recovery sketches fail closed.

## Small-real-checkpoint necessary Gate

Use the unchanged pinned `Qwen/Qwen3.5-0.8B` payload and frozen EXP-076 prompt
split. Build prompts alone fit the eight-stage generator and rank-eight residual
dictionary. Evaluation prompts are held out. Capture all causal prompt
positions from `q_proj` and `down_proj` at layers `3, 11, 23`; no generation or
future target token is used.

The cheap Gate forms a deterministic symmetric W4/A8 integer reference for
these six projections. It reports nonlinear-generator error, the most favorable
rank-eight residual correction, modular residual-span membership, family
coverage, and scaling. This is not yet model-wide operation replacement.

Promotion requires all of:

- every synthetic control passes with zero wrong accept;
- weighted exact residual-code coverage at least `99.75%`;
- every required family exact coverage at least `99%`;
- held-out corrected relative-L2 p50 at most `0.01` and p95 at most `0.05`;
- the frozen 405B operation/traffic/storage equations pass after metadata;
- zero build/evaluation leakage and zero checkpoint/input hash mismatch.

Failure of coverage or quality stops decoder integration, model-wide output
replacement, kernels, larger checkpoints, and target-server work. Do not rescue
the result with post-result tree depth, leaf count, rank, layer, projection,
prompt, field, or tolerance sweeps.

## Strongest counterexample

If held-out lookup errors keep adding independent directions, a rank-eight
dictionary recovers essentially none of them. The independent fingerprint then
correctly forces almost every query to the dense fallback, making traffic at
least `100.9265%` rather than `1.1852%`. Approximate low numerical rank alone
cannot satisfy the exact finite-field path.

## Decisions

```text
PROMOTE_SYNDROME_RECOVERED_LOOKUP_TO_INTEGER_OPERATION_REPLACEMENT_GATE
REJECT_SYNDROME_RECOVERED_LOOKUP_RESIDUAL_CODE_PATH
INVALID_SYNDROME_LOOKUP_CONTROL_FAILURE
INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION
```

## Evidence ceiling

Phase A/B synthetic exact finite-field validation and Phase C small-checkpoint
necessary residual-code observation, at most E1. BF16/Q4 output equivalence,
complete generation, physical CUDA, peak VRAM, SSD/PCIe/H2D, latency, 122B,
405B, and E2-E7 remain **NOT TESTED**.
