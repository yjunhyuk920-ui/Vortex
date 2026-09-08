# Independent gauge and native schedule audit — preregistration

Date: 2026-09-09 Asia/Seoul. Read-only subject: concurrent preregistration
`dab57747b690ecc4ba0562f3af39e781af01a836`,
`experiments/implicit_nonlinear_direct_query_20260909/PREREGISTRATION.md`.
The other writer owns that experiment. This audit runs only in its own directory.

The unchanged final mission requires arbitrary unmodified HF dense405B,
batch1/single GPU total peak <=8GiB, native logits/RNG/successor-state/KV for
every legal continuation, complete costs, same-machine native4BQ4 p50<=1.2,
p95<=1.5 and original TTFT. The user reports 405B hardware unavailable. No
hardware or whole-theory acceptance can follow from this audit.

This is a bounded constructive check of a specific O3/O4 obligation, not a new
three-principle core round or a lower-bound search. The three routes in the
subject preregistration are not adopted as new qualified principles: I is
explicit block-image tabulation and II retains original row reductions; their
existing failure scopes must be checked before any duplicate implementation.
III has an actual native correctness gap that this audit can repair.

## Frozen constructor and hypotheses

1. For invertible A,B,C over a prime field, construct either a certificate that
   `C(x .* y) = (Ax) .* (By)` for every x,y, or an explicit pair of coordinate
   basis vectors disproving it. No SAT solver or exhaustive input catalog.
   Compare the n^3 bilinear coefficients and verify invertibility by finite
   elimination. Hypothesis: all successful triples are
   `A=Da P, B=Db P, C=Da Db P` for one common permutation P and nonzero diagonals.
   Unlike the subject's same-E lemma, this allows independent gauges on all
   three wires. Compilation O(n^3) field operations, retained certificate
   O(n) words, original gauge matrices O(n^2) words; no native lift assumed.

2. Compile arbitrary row/column permutations of an explicitly declared dense
   reduction into a transformed layout **and a transported original schedule**.
   At each original leaf slot j, read the new column `inverse_perm[j]`.
   Reproduce the same product primitive, same balanced FP32 addition tree and
   +0 padding. Equality follows by literal equality of every leaf/tree node,
   not by associativity. Loops and output permutation maps are finite and paid.

   Frozen numerical counterexample: one BF16-representable row
   `[2^24, 1, -2^24, 0]`, all-one input, column order `[0,2,1,3]`.
   Predicted original balanced FP32 result 0; naive permuted canonical tree 1;
   transported tree 0. Final BF16 stores also differ on the naive path.
   This is not evidence about an unspecified CUDA/HF kernel. Generic proof
   applies only when the compiler transports the actual pinned ABI schedule.

3. Do not turn a per-output bilinear-rank count into a global product count.
   For `H(z,w)=C((A^-1 z).*(B^-1 w))`, output coefficient matrix
   `T_i=A^-T diag(C_i) B^-1` has rank `nnz(C_i)`; nonetheless all outputs share
   n middle products after the paid input transforms, followed by paid C.
   Frozen GF2 n=4 control: A=B=I, C=I+J. C is invertible; each row rank is 3,
   sum is 12, but the explicit shared decoder uses four products plus eight
   XORs to sum the three products in each row. This rejects an invalid summed
   lower-bound inference, not all cheap transformed gates.

## Native/state and 405B connection

The target operation is a dense projection and the Hadamard gate in a gated
MLP. Transporting every native leaf and tree operation preserves that projection's
words. A complete graph compiler would also have to transport all nonlinearity,
normalization, residual, RoPE, aliases, logits, KV/cache and sampler interfaces;
no such whole-HF compiler is supplied here. GF2/prime-field classification is
not native BF16 arithmetic. At mission level O1-O5 remain OPEN and O6 PARTIAL.

The permutation producer reads every coefficient and executes every original
product/addition. Its work fraction is exactly 1 before gather/scatter and
metadata; therefore it is a correctness reference, not a >=10x core. Charge
construction, original and transformed storage, indices, input/output movement,
CPU RAM/SSD/PCIe/HBM, full GPU allocation, KV/state, verification and fallback.
No target hardware or latency measurement, no zero-cost gauge application.

## Frozen finite checks, after this file

- Enumerate all invertible A,B for GF2,n=2 and n=3; C is forced by diagonal
  coefficients, so derive C and test all triples exhaustively by coefficient
  comparison. Predicted accepting pairs: 2 and 6.
- GF3,n=2: enumerate all 48 invertible matrices and all pairs; predicted valid
  triples: `2!*(3-1)^(2*2)=32`.
- Independent evaluate basis-pair equations on every classified pair, rather
  than accepting the classifier's own flag. Tests are not the general proof.
- Numerical witness above plus all 4! column permutations and both row orders
  on a fixed two-row fixture; 3 fixed input vectors, no tuned seeds/thresholds.
- Reproduction outputs are deterministic JSON and SHA-256; Python timings and
  object memory are not target-cost evidence. No GPU, HF run or original source
  writes. Stop on mismatch; retain failure. Independent Sol review is separate.
