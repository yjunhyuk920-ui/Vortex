# E0 Cartesian Bilinear Causal-Utility Gate

## Verdict

```text
PROMOTE R^T W U AS AN EXACT BATCHED-SCALAR AUXILIARY
REJECT K^2 SCALARS AS K^2 CAUSAL TOKENS
REJECT A FULL LOSSLESS SWEEP AT K=32
REQUIRE A PARTIAL EXACT INFORMATION SOURCE
NO SURVIVING CANDIDATE
```

No checkpoint, model, download, backend, server, or hardware action occurred.

## Elementary explanation

Suppose 32 children each take one test with 32 questions. One grading pass can
fill a table with `32 x 32 = 1,024` marks. It is still work for 32 children,
not 1,024 children.

The same distinction applies here. If `U` contains 32 forward states and `R`
contains 32 questions about each state, then

```text
C = R^T W U
```

contains 1,024 exact scalar answers. Only the 32 columns of `U` are distinct
model states that can advance a causal token path. A different column of `R`
changes the question asked about a state; it does not create another state.

## 1. Exact algebra

For left directions `r_j` and forward states `u_i`, one streamed matrix use
can compute every

```text
C[j,i] = r_j^T W u_i.
```

This is genuine and useful reuse. For example, many winner-versus-competitor
tests for the same state can share one weight pass. It reduces duplicated
arithmetic for scalar certification.

The causal denominator is nevertheless the number of distinct forward
states. Every next Transformer state must have its own exact or certified
forward activation and KV consequence. The left direction supplies neither.
Consequently, a batch with `K` state columns can contribute at most `K` nodes
to one accepted causal path, even when it produces `K^2` scalar comparisons.

## 2. Cheapest full-sweep Gate

Give the mechanism the favorable 10.6-bit entropy figure for every one of the
405,849,243,648 registered parameters. Also make decompression, GEMM, KV,
metadata, scheduling, and every other cost free. At 32 GB/s, one sweep is

```text
405,849,243,648 x 10.6 / 8 = 537,750,247,833.6 bytes
537,750,247,833.6 / 32 GB/s = 16.8046952448 seconds.
```

With 32 causally distinct states, the valid floor is

```text
16.8046952448 / 32 = 525.1467264 ms/token,
```

or `26.25733632x` the 20 ms target. A full sweep needs at least 841
causally usable tokens merely to cross the I/O line.

Dividing instead by the 1,024 scalar table entries produces
`16.4108352 ms/scalar` and appears to pass. That denominator is invalid:
competitor inequalities and other measurements of the same state are not new
generated tokens. The registered DFloat byte count independently gives
`538.30078125 ms/token` at 32 states.

## 3. New lossless-compression evidence

Tan et al. report a tile-addressable ANS codec that reaches within 0.01--0.1
bits of empirical Shannon entropy and decodes consumed tiles directly into a
GEMM pipeline. This is a useful implementation source for any future partial
tile scheme. Their BF16 measurements remain about 10--12 bits per weight; the
Llama-405B BF16 compression factor shown is about 1.5x, not the 7.6x number
reported for an already-INT4 representation. The method still decodes every
tile consumed by GEMM and supplies no partial-state or token-decision
certificate. Primary source: [Approaching Shannon Bound with Lossless LLM
Weight Compression](https://arxiv.org/abs/2606.15789).

Thus the new codec improves the exact tile transport layer but does not turn
a full sweep into a 20 ms causal decode.

## 4. Claim boundary

This Gate rejects only one accounting promotion:

```text
K^2 exact scalar measurements != K^2 causally usable token states.
```

It does not reject Cartesian batching as an auxiliary, selective ANS tile
reads, a sound partial-evidence executor, or a globally nonlinear rank-one
data structure. Those routes must still explain how the unread information
determines the accepted path and exact state.

## 5. Reproduction

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_cartesian_bilinear_causal_utility_gate.py `
  --output-dir results\e0_cartesian_bilinear_causal_utility_gate

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_cartesian_bilinear_causal_utility_gate -v
```
