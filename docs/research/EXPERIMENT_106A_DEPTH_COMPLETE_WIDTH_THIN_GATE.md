# EXP-106A — Depth-Complete Width-Thin Checkpoint Surrogate Reality Gate

## Question

Can an untrained checkpoint-derived draft preserve causal information from all
126 target layers with a narrow state `m << 16,384`, while the complete resident
representation, per-token scan, vocabulary decision path, KV, workspace, and
exact target verification remain inside the 8-GiB and p50 ledgers?

## Three materially different principles

1. **Depth-complete width-thin checkpoint surrogate — implemented.** A fixed
   checkpoint-static linear bridge derives a width-`m` state. Every target layer
   is represented by narrow Q/K/V/O and SwiGLU operators; a separate narrow
   embedding table and full narrow vocabulary head are charged. No training,
   future target state, free bridge/head, or free HBM scan is used.
2. **Bit-sliced all-layer state transducer — statically rejected.** Even one
   resident checkpoint bit per target parameter is larger than both the 8-GiB
   budget and the p50 per-token byte budget.
3. **Cold-probe exact vocabulary decision index — auxiliary only.** EXP-105A
   already charged a query-dependent exact head tournament. A stronger head
   index still supplies no all-layer causal hidden state or `A>=339` source.

## Fully charged representation

For each registered width, the ledger includes:

```text
Q4 payload + group-128 FP16 scale/zero bytes
all 126 narrow layer operators and RMSNorm vectors
separate narrow embedding and 128,256-row narrow LM head
draft KV for A=339
target KV reserve, target staging, activation workspace
allocator/fragmentation reserve
one actual embedding-row read per proposed token
all-layer and full narrow-head scan per proposed token
raw BF16 exact target sweep amortized over actual A
```

Narrow shapes are frozen before results:

```text
intermediate = align128(ceil(53,248 * m / 16,384))
KV width     = max(128, align128(ceil(1,024 * m / 16,384)))
```

## Resource frontier

| m | peak resident | draft scan/token | required A p50 | kernel dim | flat-spectrum best-rank Frobenius error |
|---:|---:|---:|---:|---:|---:|
| 128 | 1.465912 GiB | 0.026286 GB | 342 | 16256 | 99.608609% |
| 512 | 1.698543 GiB | 0.249908 GB | 378 | 15872 | 98.425098% |
| 1,024 | 2.347819 GiB | 0.912178 GB | 546 | 15360 | 96.824584% |
| 1,536 | 3.381136 GiB | 1.986807 GB | 1965 | 14848 | 95.197164% |
| 1,664 | 3.729347 GiB | 2.351975 GB | 16902 | 14720 | 94.785943% |
| 1,792 | 4.063258 GiB | 2.701788 GB | impossible | 14592 | 94.372930% |
| 2,048 | 4.798493 GiB | 3.473798 GB | impossible | 14336 | 93.541435% |
| 16,384 | 202.377974 GiB | 214.493273 GB | impossible | 0 | 0% |

Widths through 1,664 pass the resident and scan-alone Gates, but every one is a
noninjective state map. Width 1,792 and above already fail the p50 scan floor;
width 3,072 and above also exceed 8 GiB. Restoring `m=16,384` removes the kernel
but restores a 202.377974 GiB representation and a
214.493273 GB per-token scan.

## Exact finite-word controls

### Full-depth coordinate control

A depth-4 exact integer residual stack with tied coordinate norms is executed for
512 steps. When the decision coordinate is retained, the narrow surrogate
matches all 512 decisions. Moving the equally weighted decision coordinate to
an omitted channel causes a mismatch at step 1 and every subsequent step.

```text
positive matches       512 / 512
negative matches       0 / 512
negative first mismatch 1
```

### Dense exact encoder collision

A full-row-rank dense integer encoder uses a Hadamard left block. Two distinct
states have byte-identical width-16 codes, while a legal target readout gives
scores 0 and 2.

```text
encoder rank               16
nonzero kernel vector      True
encoded states byte-equal  True
target distinguishes them  True
```

### Flat-spectrum dense operator

An exact dense `+/-1` Hadamard operator has all singular values equal to
`sqrt(128)` within `7.105e-15`. Therefore
for every `m<d`, even the best rank-`m` approximation has relative operator-norm
error 1. This closes the rescue claim that a better checkpoint-static SVD would
provide a uniform arbitrary-checkpoint guarantee.

## Bit-sliced floor

```text
one bit per target parameter  50,731,155,456 bytes
                               47.247070312 GiB
ratio to 8 GiB                5.905883789x
ratio to p50 bytes/token      21.137981440x
max resident bits/parameter   0.169322667990
```

One bitplane cannot be resident or scanned per proposed autoregressive token.
Dropping state/weight bits has the same distinguishable-state counterexample;
a lossless sub-bit-per-parameter program would require a new checkpoint
structure not available for arbitrary checkpoints.

## Decision

```text
REJECT_DEPTH_COMPLETE_WIDTH_THIN_LINEAR_AND_BIT_SLICED_SURROGATES_AS_UNIVERSAL_CAUSAL_CORE
```

This rejection is a resource/finite-word dichotomy:

```text
m < d  -> resource-feasible widths exist, but the bridge has a kernel and an
          arbitrary legal target can distinguish collided states immediately.
m = d  -> the state collision disappears, but full-width bytes and scan fail
          both 8-GiB and p50 constraints by large margins.
```

It does not reject empirical narrow drafts for selected checkpoints, trained or
distilled models (which are forbidden by the mission), or a future nonlinear
cold-backed injective code with a genuinely sub-dense value-changing decoder.
