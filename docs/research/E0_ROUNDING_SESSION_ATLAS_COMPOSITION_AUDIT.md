# E0 Rounding-Session-Atlas Composition Audit

Status: complete deterministic composition audit; the concrete raw-repair
composition is rejected.  This document does not assign a new experiment
number because the checkpoint observation is already authoritative under
F-066/ΩROUNDLOCK and the raw-page source is already bounded under F-051.

## Question

Can a prompt-derived session basis and cached image

```text
Q, Z = WQ
```

propose a BF16 layer output, repair only coordinates whose words do not match
the reference, and then use DFlash-style block acceptance to amortize the
repair sufficiently for the 405B/native-4B target?

The proposed execution equation was:

```text
x = Qa + r
y = Za + Wr
```

with BF16-equal coordinates locked after every linear operation and ordinary
raw checkpoint rows/pages used to repair the rest.

## Existing measured checkpoint evidence

The pinned source is
`results/e0_native_exact_shortcut_frontier/summary.json`, SHA-256
`fa8a2d0d8c434400db63634730b52d891be4359eaab2509cf4e1f3415eeb09be`.
It uses the unchanged Qwen/Qwen3.5-0.8B checkpoint, layer-23 MLP
`down_proj`, and the frozen causal row from EXP-083B.

ΩROUNDLOCK already granted more information than the proposed executor has:

- the true target BF16 words were visible to the lock selector for free;
- the Atlas-plus-one-page candidate was already computed;
- selector, certificate, state work, and suffix work were free.

MEASURED:

```text
coordinates                            1,024
BF16-equal locked coordinates              1
oracle lock fraction                 0.09765625%
oracle unlocked fraction            99.90234375%
whole vector locked                       false
```

Therefore an ordinary exact row repair still executes at least
`1023/1024` of the dense row-dot population at that repair point.

## The accounting correction

Block speculation can amortize one weight load across several positions, but
it does not make the corresponding row-dot evaluations disappear.

Let:

```text
rho = repaired dense-row fraction
A   = perfectly accepted positions in the block
B_W = one dense weight-stream byte cost
C_W = one dense MatVec arithmetic cost
```

Granting perfect page reuse across the entire block:

```text
weight traffic per committed token = rho * B_W / A
```

But ordinary raw repair performs the repaired row dots for all `A` positions:

```text
block repair arithmetic             = A * rho * C_W
arithmetic per committed token       = (A * rho * C_W) / A
                                    = rho * C_W
```

The `1/A` factor therefore cancels from arithmetic.  The earlier proposed term
`C_repair/A` is valid only if `C_repair` denotes a block-total algorithm whose
work is independent of `A`.  Raw row/page repair does not have that property.

## Derived Gate

The registered compute allowance from the native-shortcut audit is:

```text
allowed compute fraction              1.1827051732%
measured raw repair fraction          99.9023437500%
miss factor                            84.46935552x
```

Even under perfect acceptance:

```text
A     optimistic traffic/token     repair arithmetic/token
8          12.48779297%                  99.90234375%
64          1.56097412%                  99.90234375%
85          1.17532169%  PASS            99.90234375%  FAIL
96          1.04064941%  PASS            99.90234375%  FAIL
128         0.78048706%  PASS            99.90234375%  FAIL
```

A perfect accepted length of 85 is enough only for the optimistic p50 weight
traffic equation.  It leaves the arithmetic 84.47 times above the complete
compute allowance before basis projection, proposal generation, addressing,
metadata, attention, KV state, verification, or synchronization.

## Independent cross-position arithmetic check

The second pinned source is `results/exp_080a/summary.json`, SHA-256
`0debbb96f0a31b1ef2ffc1e662109687aed3b2cedaaf89ac1c013bf9a94c1d83`.
EXP-080A granted exact future activations, a free perfect proposal, one target
sweep, and favorable workspace.  Standard constructive Strassen at the largest
registered block still required:

```text
block length                         16,384
constructive arithmetic fraction    36.1115796714%
p50 miss factor                     30.46914535x
```

Thus merely changing GEMV repair into a standard exact block GEMM does not
close the arithmetic gap either.

## Decision

```text
REJECT_ROUNDING_SESSION_ATLAS_RAW_REPAIR_COMPOSITION
```

Rejected scope:

```text
prompt/Atlas basis proposal
+ BF16 coordinate locking
+ ordinary raw row or page repair
+ DFlash-style perfect block acceptance and optimistic weight reuse
```

The session basis can remain an approximate drafter, prioritizer, or diagnostic
auxiliary.  It is not the exact core under ordinary raw repair.

## What remains open

This is not a rejection of DFlash, every rounding firewall, or every exact
cross-position algorithm.  A continuation must add a materially different
source that reduces **repair arithmetic itself**, for example a fully specified
lossless non-rowwise/coded cross-position repair.  That source must include its
artifact, selector, decoder, state, verification, miss, fallback, RAM, SSD,
PCIe, VRAM, and existing-ISA equations before implementation.

Changing basis rank, page size, prompt, selector, certificate tolerance, or
accepted block length does not invalidate this verdict.

## Evidence classification

```text
ΩROUNDLOCK lock counts                   MEASURED, existing actual checkpoint
Hyperblock operation ratios              DERIVED, existing exact cost model
DFlash/raw-repair composition             DERIVED in this audit
new model forward                         NOT RUN
new hardware measurement                  NOT RUN
405B execution                            NOT TESTED
8 GiB and native-4B latency acceptance    NOT TESTED
```

## Reproduction

```bash
python -m pytest -q tests/test_rounding_session_atlas_composition.py
python scripts/derive_rounding_session_atlas_composition_audit.py \
  --output-dir /tmp/rounding-session-atlas-audit
diff -ru results/e0_rounding_session_atlas_composition_audit \
  /tmp/rounding-session-atlas-audit
```
