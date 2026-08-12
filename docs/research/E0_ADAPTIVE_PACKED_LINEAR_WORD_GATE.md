# E0 Adaptive Packed-Linear Word Gate

Status: **STRUCTURALLY VALID / BLOCK-LOCAL LINEAR WORD ROUTE REJECTED**

This is an E0 theorem and exact finite calculation. It performs no model or
hardware run and does not claim a general nonlinear cell-probe lower bound.

## 1. The candidate that was tested

The earlier sparse-dictionary Gate charged one selected linear summary at a
time. A real 64-bit load is stronger: one address may return 64 completely
unrelated binary linear summaries of the checkpoint. The decoder is also
allowed to choose its next address from the query and every word already
returned, then apply arbitrary exact postprocessing.

This is the strongest block-local packed-linear version of the idea. It is
strictly more favorable than requiring an extension-field word, because the
64 summaries in one word need not be scalar multiples or obey any field
layout.

## 2. Zero-transcript span theorem

Run the decoder on the all-zero checkpoint. Every returned linear summary is
zero, so this transcript fixes a support `J` of at most `p` word addresses.
If a checkpoint perturbation `h` lies in the common kernel of all summaries
inside these words, the decoder follows the identical address path and sees
the identical values. Exactness of the requested parity `<q,W>` therefore
requires

```text
q in span_F2(all summaries exposed by J).                 (1)
```

One `w`-bit word exposes a subspace of dimension at most `w`, so `j` words
give dimension at most `j*w`. This argument already permits adaptive
addresses, repeated probes, and arbitrary exact output logic.

For an `a x b` binary matrix, the exact largest number of nonzero rank-one
masks in a `d`-dimensional subspace is, after taking `a<=b` and writing
`d=q*b+r`,

```text
M(a,b,d) = (2^q-1)(2^b-1) + 2^q(2^r-1).                 (2)
```

There are `C(L,j)` possible supports of `j` words. Consequently every such
layout must satisfy

```text
sum(j=1..p) C(L,j) M(a,b,min(j*w,a*b))
    >= (2^a-1)(2^b-1).                                  (3)
```

The left side intentionally overcounts overlaps. Failure is a proof of
impossibility; passing only says that raw capacity is no longer enough to
reject the layout.

## 3. Registered `31 x 42` witness

Give the one-bit plane its proportional share of the complete 8 GiB advice.
The 1,302 source bits become 1,523 stored linear bits, rounded upward to 24
fully useful 64-bit words. This padding is favorable to the candidate.

Equation (3) gives:

```text
seven-word support-union / required rank-one masks
    = 0.19975212628...

first probe count not rejected by capacity = 8 words
physical bits read                          = 512
full four-lane Q4 tile bits                 = 5,208
favorable traffic fraction                 = 64/651
                                             = 9.8310...%
registered latency-derived fraction        = 8/675
minimum/target                              = 8.2949...x
```

Eight words are not a construction. They are only the first point where the
necessary union count stops proving impossibility.

## 4. Complete finite rectangle scan

All 8,256 rectangles with `1 <= a <= b <= 128` were granted the same
proportional storage, full 64-bit padding, and the favorable four-Q4-lane
traffic denominator. None reaches `8/675`.

The most favorable case is:

```text
a x b                                      = 118 x 128
source bit-plane dimension                 = 15,104
padded words                               = 277
first probe count not capacity-rejected    = 22
favorable traffic fraction                 = 11/472
                                             = 2.330508...%
minimum/target                              = 1.9663...x
```

Thus merely packing independent linear atoms into ordinary words does not
repair the fixed-linear route. The screen uses the latency-derived resource
fraction, not an assumption that a chosen percentage can be imposed on every
checkpoint.

## 5. Exact boundary and next mechanism

The theorem covers:

- arbitrary binary linear summaries inside every word;
- deterministic data-dependent adaptive word addresses;
- arbitrary exact postprocessing of returned words;
- every block-local rectangle through the registered side-128 scan.

It does **not** cover:

- nonlinear stored functions of the checkpoint;
- one global word construction coupling independently charged matrices;
- a joint 32-query physical support theorem;
- native Q4/BF16/FP32 arithmetic and rounding;
- a complete runtime.

Therefore the next constructor must change the stored information source. A
different packing or address scheduler for linear summaries is no longer a
new mechanism.

## 6. Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_adaptive_packed_linear_word_gate.py `
  --output-dir results\e0_adaptive_packed_linear_word_gate

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_adaptive_packed_linear_word_gate -v
```
