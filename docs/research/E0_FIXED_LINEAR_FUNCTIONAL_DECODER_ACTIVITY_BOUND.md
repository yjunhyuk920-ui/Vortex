# E0 Fixed Linear Functional Decoder Activity Bound

## Verdict

```text
REJECT_FIXED_LINEAR_FUNCTIONAL_DECODER
KEEP_NONLINEAR_MINIMUM-WEIGHT_SYNDROME_DECODER OPEN
NO SURVIVING CANDIDATE
```

This is a scoped exact theorem for the sparse functional dictionary opened in
the preceding audit.  It closes invertible basis changes, fixed FFT/tensor
inverse transforms, and fixed linear syndrome representatives.  It does not
close a query-dependent nonlinear minimum-weight decoder.

No model, checkpoint, server, backend, or hardware action occurred.

## Elementary explanation

Suppose a dictionary contains mixed pieces of the checkpoint.  A fixed linear
decoder is like a transparent stencil: every question always lights the
dictionary pieces dictated by one fixed wiring diagram.

A nonzero wire tested by a random rank-one question lights at least one time in
four.  Ask 32 independent questions and the chance that the wire never lights
is at most `(3/4)^32`, about `0.010045%`.  In other words, almost every useful
wire lights at least once.

An exact reversible wiring diagram needs at least as many useful wires as
there are checkpoint bits.  Even after putting every one-bit cell that fits in
8 GiB into free hot memory, the remaining cold batch contains at least
`41.874345776 GB` of lit cells.  This exceeds the complete registered block
byte allowance by `3.0387x`.

The way out cannot be another fixed transform.  It must choose a different,
nonlinear short representation for each question.

## 1. Declared interface

Let a binary dictionary have atoms in the columns of

```text
G in F_2^(D x S)
```

and stored checkpoint cells

```text
c = G^T W.
```

This audit considers a fixed linear query decomposer

```text
a(q) = H q,
G H = I_D.                                                (1)
```

Equation (1) gives

```text
q = G a(q),
<q,W> = <a(q),c>.
```

Because `G H` is the identity, `rank(H) >= D`.  Therefore at least `D` rows of
`H` are nonzero, regardless of how overcomplete the dictionary is.

The theorem grants query computation, atom metadata, XOR computation, and all
reuse within the 32-query batch for free.  It also lets the compiler place any
8 GiB of one-bit cells in hot memory and pack every remaining 64 cells into one
cold word.

## 2. Activity lemma

Take one nonzero row `h` of `H` and split it across the disjoint model matrices
as matrices `A_i`.  For independently uniform binary directions `r_i,u_i`,
the corresponding decoder coefficient is

```text
<h,q> = XOR_i r_i^T A_i u_i.                              (2)
```

For one matrix of rank `s`, direct counting gives

```text
Pr[r^T A u = 1] = (1 - 2^-s)/2.                           (3)
```

The signs in (2) are independent across blocks, so

```text
Pr[<h,q> = 1]
  = (1 - 2^(-sum_i rank(A_i)))/2
  >= 1/4                                                  (4)
```

for every nonzero row.  The test suite exhaustively checks (3) for every
binary `2 x 2`, `2 x 3`, and `3 x 2` matrix and independently checks the
two-block character product.

For `K` independent queries, one row is active at least once with probability

```text
1 - (1-p)^K >= 1 - (3/4)^K.                               (5)
```

Averaging then guarantees one `K`-tuple whose union contains at least the
ceiling of the expected number of active cold rows.

## 3. Registered finite substitution

Use only the frozen non-embedding binary coefficient population:

```text
D useful decoder rows at least            403,747,897,344
complete hot one-bit cell grant             68,719,476,736
cold nonzero rows at least                  335,028,420,608
K                                                   32
1-(3/4)^32                         0.9998995475742793.
```

Equations (4)--(5) force one independent 32-query tuple with

```text
cold active cell bits at least              334,994,766,191
best possible packed 64-bit words             5,234,293,222
cold bytes                                   41,874,345,776.
```

The DFloat11-derived complete block allowance is

```text
551.22 GB / 40 = 13,780,500,000 bytes.
```

Thus the fixed linear decoder requires at least

```text
41,874,345,776 / 13,780,500,000 = 3.03866665x
```

the block allowance.  At a favorable `32 GB/s`, granting zero compute and
perfect overlap inside the batch, this is

```text
1,308.5733 ms/block = 40.8929 ms/token.
```

It already exceeds 20 ms/token before native value widths, metadata, decoder
work, KV/state traffic, or nonlinear repropagation.

## 4. Exact scope boundary

The hard query object is an independently selectable rank-one tuple across the
registered matrices, repeated independently 32 times.  This is the same strong
all-query service interface used by the separable-code audit.  This note does
not prove that the 32 causal states of an ordinary Transformer trace realize
that complete Cartesian population.

The proof also assumes the exact answer source consists of the linear cells
`G^T W` and the fixed linear representative `Hq`.  It does not cover:

- a nonlinear minimum-weight representative of the coset `G a=q`;
- arbitrary nonlinear checkpoint advice combined with the cells;
- a causal restriction that provably excludes the hard query tuple;
- native bounded-word summaries whose exact decoder is materially different.

These are real exclusions, not implementation details.  In particular, the
open `OMEGA-FUNCDICT` interface has now become narrower:

```text
near-linear, globally aligned cold dictionary
+ query-dependent nonlinear sparse syndrome decoding
+ succinct atom/address generation
+ native exact arithmetic and complete cost closure.
```

No such construction is present.  Therefore the result is a useful rejection,
not increased target feasibility.

## 5. Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_linear_functional_decoder_activity.py `
  --output-dir results\e0_linear_functional_decoder_activity

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_linear_functional_decoder_activity -v
```

Authority:

```text
results/e0_linear_functional_decoder_activity/summary.json
results/e0_linear_functional_decoder_activity/checksums.sha256
```

