# Native fiber obstruction — preregistration, 2026-09-09

Base: independent research checkout 207da384b85e40b5d3e18ef7c53347bce228be1a.
This is a bounded continuation of the rank-normal candidate registered at
52facca, not three new qualifying principles or a reopened admitted core.
The other writer owns its GF2 carrier experiments; no duplicate run is planned.

Final mission and O1-O6 remain unchanged. The local task addresses O3: can an
algebraically full-rank native dense map be made a pure full-coordinate copy by
arbitrary bijective input/output encodings? O4 pays the witness constructor and
reference experiment. Neither a counterexample nor this audit closes O1-O5.

Proposed theorem: fibers of a finite map are invariant under bijective input
and output relabelings. A noninjective native map cannot be conjugated (even with
different input/output encodings) to a full-coordinate copy/permutation. This
does not prohibit encodings that retain side state, global cross-wire encodings,
nonbijective representations with an exact continuation relation, or a different
native operator. No global advice is split per matrix.

Candidate witness: for any 2 <= n <= 16384, use dense W = I + 11^T over the
reals (diagonal 2, all other entries 1), unchanged representable BF16 weights.
Its real determinant is n+1, so algebraic rank is n. Inputs have x_0=1 and
x_j=a_j*delta, j>0, a_j in {0,...,255}, delta=2^(-24-ceil(log2(n))-8-2).
All operands are finite BF16 values. Each native product rounds separately to
FP32 RNE, then a fixed balanced FP32 RNE tree pads with +0 to power-of-two
width. Observed output is the FP32 word vector, optionally followed by BF16 RNE
store. Hypothesis: all 256^(n-1) inputs yield [2,1,...,1]. The proof must bound
every subtree before/after it meets the large leaf, not just the exact sum.

Frozen controls before execution:
- Exact rational proof inequalities and counts for n=2,3,16,16384.
- A fixed non-absorption control: n=2, x=[1,1] returns [3,3], different from
  x=[1,0] returning [2,1]. This detects a constant-output reference mistake.
- Exhaust n=3 (65536 inputs) under the declared FP32 ABI; compare every row word
  to the fixed expected word, not only a second wrapper over the same result.
- Run n=16384 only with two nonzero inputs using an explicit two-leaf/pruned-tree
  evaluator proved identical to the declared tree. Charge positive-zero pruning
  proof and state its restricted input domain; do not claim full dense hardware.
- Compare that pruned evaluator to full dense tree evaluation at n=2,3,16,64
  on fixed distinct inputs. No large dense matrix is allocated for the large
  analytical witness; its full reference operation/storage costs remain paid.
- Full native HF/CUDA/KV/RNG/405B timing is NOT TESTED. No real legal HF trace is
  claimed; this is a native projection-operator obstruction, not a theorem of
  reachability of every witness activation through a specific model.

Stop if finite BF16 representability, native tree induction, or fiber invariance
fails. Correct/refute only this proposed theorem, without threshold tuning.
Universal tiny-test passes do not replace proof. If established, the next core
must supply a native operator/side-state map instead of assuming algebraic
full rank implies native bijectivity. Do not promote this into a universal
impossibility result or rerun toy witnesses as primary progress.
