# VORTEX Research Handoff

## Latest completed local Gate — EXP-106A

```text
REJECT_DEPTH_COMPLETE_WIDTH_THIN_LINEAR_AND_BIT_SLICED_SURROGATES_AS_UNIVERSAL_CAUSAL_CORE
```

Evidence:

```text
results/exp_106a/local/result.json
results/exp_106a/local/checksums.sha256
docs/research/EXP_106A_LATEST_RESULT.md
```

Deterministic core:

```text
9399769c602659bc2c52c0bb7fe2435aef679063bde85fb34523d01d58ef081b
```

Validation:

```text
8 focused tests passed
resource frontier widths 128..16,384
exact dense linear-encoder collision: PASS
full-depth 512-step retained-channel positive control: 512/512
omitted-channel negative control: first mismatch 1, matches 0/512
flat-spectrum dense Hadamard control: PASS
byte-identical deterministic rerun
SHA-256 verification: PASS
GitHub Actions: not run
```

Interpretation: a fixed width-thin linear state can be cheap, but every cheap
width is noninjective and an arbitrary legal checkpoint can distinguish its
collisions. Full width and even one all-parameter bitplane restore resource
failure. Do not reopen by changing `m`, SVD rank, coordinate ordering, bridge
rotation, bitplane count, or Q4 group size. Reopening requires a nonlinear
cold-backed injective code with an explicit sub-dense decoder or shared
value-changing computation across distinct states.

## Next action

Run EXP-107A locally: exact distinct-state shared-weight sweep. The first Gate
must prove that branch coding removes real operations/bytes rather than merely
batching `N` branches, and must fully charge `N/A`, branch matrices, nonlinear
separation, KV/state, and workspace.

## Operating rule

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

No duplicate GitHub Actions run unless explicitly requested.
