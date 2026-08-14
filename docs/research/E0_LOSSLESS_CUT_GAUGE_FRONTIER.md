# E0 Lossless, Exact-Cut, and Gauge Frontier

## Question

Do three materially different exact mechanisms remove the arbitrary dense
checkpoint traffic without assuming the old 2.5% number?

1. `OMEGA-CUTSUM`: represent a matrix as a weighted bipartite graph and answer
   the Bilinear Cross Residual through exact cut queries.
2. `OMEGA-ZIPWAVE`: stream an entropy-coded BF16 checkpoint and decompress
   directly into matrix-multiply registers.
3. `OMEGA-GAUGEFOLD`: use function-preserving Q/K and V/O gauge freedom to
   remove or fuse attention projections.

The only first Gate is the final unchanged-output, 8 GiB, and 20 ms/token
contract. No model, hardware, or 2.5% premise is used.

## Exact-cut relabeling

For a matrix `W`, form a bipartite weighted graph with row vertices on the
left, column vertices on the right, and edge weight `W[i,j]`. For row subset
`R` and column subset `U`, let `delta(S)` be the exact weight of the cut of
vertex set `S`. Then

```text
sum(W[i,j] for i in R, j in U)
    = (delta(R) + delta(U) - delta(R union U)) / 2.
```

The reduction is exact for indicator queries, but it does not implement
`delta`. Expanding one cut gives

```text
delta(S) = sum(i in S, degree(i))
           - 2 * sum(i<j, i in S, j in S, weight(i,j)),
```

so the arbitrary quadratic subset aggregate remains inside the alleged
oracle. The cut-query literature treats this value as an input oracle; it
does not provide a target-cheap exact evaluator for an arbitrary dense
weighted graph. Approximate cut sketches and sparsifiers cannot preserve
native exact inference. This candidate is therefore a relabeling of the open
bilinear source, not a constructor.

Exact cut values also recover every edge through singleton and pair cuts,

```text
2 w(i,j) = delta({i}) + delta({j}) - delta({i,j}),
```

but this is only an injectivity fact; it is not overstated as a query-time
lower bound. A primary description of the VMV/cut-query relationship is
Rashtchian, Woodruff, and Zhu, *Vector-Matrix-Vector Queries for Solving
Linear Algebra, Statistics, and Graph Problems*,
<https://arxiv.org/abs/2006.14015>.

## Fused lossless streaming ceiling

ZipServ is a real bit-exact engineering improvement: it decodes fixed-length
compressed tiles directly into Tensor Core registers. It reports 11.3 bits
per BF16 weight and a 10.6-bit entropy lower bound for the measured format,
with up to 30% model-size reduction and 1.22x average end-to-end speedup.
Primary source: Fan et al., *ZipServ*,
<https://arxiv.org/abs/2603.17435>.

Grant the lower 10.6-bit figure to every one of the registered
`405,849,243,648` parameters, charge decompression and all other work as zero,
and grant a 32 GB/s link. The exact favorable calculation is:

```text
lossless bytes                  537,750,247,833.6
one sweep at 32 GB/s            16.8046952448 s
tokens/sweep needed at 20 ms    841
I/O per token at 32 tokens      525.1467264 ms
```

The target-link favorable 8 GB/s ceiling needs 3,361 zero-compute tokens per
sweep. Thus compressed-domain execution removes a decompression buffer and
roughly a third of traffic; it does not remove the checkpoint sweep. It is a
useful auxiliary for any future surviving reuse mechanism, but not the core.

## Attention-gauge ceiling

The exact gauge group of generic multi-head attention factorizes into per-head
Q/K and V/O transformations; RoPE restricts the Q/K group further. See Wang
and Wang, *Complete Characterization of Gauge Symmetries in Transformer
Architectures*, <https://nips.cc/virtual/2025/136893>.

The stronger query-elimination result is explicitly proved under simplifying
assumptions and validated by training reduced models; it is not a post-hoc,
bit-identical conversion theorem for an arbitrary RoPE/GQA checkpoint. See
Karbevski and Mijoski, *Key and Value Weights Are Probably All You Need*,
<https://arxiv.org/abs/2510.23912>.

The Gate does not need those caveats. Grant the candidate the impossible
advantage of deleting **all Q, K, V, and O parameters in every layer for
free**. On the registered Llama 3.1 405B shape this removes only
`71,873,593,344` parameters, or about 17.71% of the model. More than 82% of
the dense coefficients remain, while the compute-only 20 ms envelope allows
1.18270517%. Hence any attention-only gauge or fusion route fails before
native-rounding and representation costs.

Per-head dense QK/OV products do not help: each product has rank at most the
head dimension, and the existing two factors are already its compact exact
representation. Materializing one dense product per head expands storage.

## Decision

```text
OMEGA-CUTSUM: REJECTED AS EXACT BILINEAR RELABELING
OMEGA-ZIPWAVE STANDALONE CORE: REJECTED BY SHANNON SWEEP CEILING
ATTENTION-ONLY GAUGE/FUSION CORE: REJECTED BY FREE-DELETION CEILING
FUSED LOSSLESS EXECUTION: RETAIN AS AUXILIARY
GENERAL NONLINEAR ADAPTIVE RANK-ONE GAP: OPEN
NO SURVIVING CORE CANDIDATE
```

## Reproduction

```powershell
$env:PYTHONPATH = (Resolve-Path '.').Path
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_lossless_gauge_frontier.py `
  --output-dir results\e0_lossless_gauge_frontier
```

No checkpoint, model forward, backend, download, server, or hardware action is
part of this E0 calculation.

Canonical `summary.json` SHA-256:

```text
af1cff41f0f0add560427ef3e2ac0d115dc70b68d26545a346da215c234846d1
```
