# Native gauge transport: constructive audit and exact bit-gate continuation

Date: 2026-09-09. Subject: Principle III of
`dab57747b690ecc4ba0562f3af39e781af01a836`,
`experiments/implicit_nonlinear_direct_query_20260909/PREREGISTRATION.md`.
This is an independent bounded O3/O4 construction, not three new admitted core
principles or a completed arbitrary-native dense executor.

## Result and remaining main problem

Two actual finite native procedures now accompany a strengthened field lemma:

1. A dense layout-permutation compiler transports the original logical product
   slots and reduction tree. The naive native permutation claim needs this
   extra condition: a four-leaf counterexample changes 0 to 1 otherwise.
2. An opaque-word butterfly encoder has a finite inverse and an explicit exact
   native Hadamard decoder, including an encoded recurrent gate state. It costs
   three butterfly transforms and n original native products. Thus one may not
   infer that every non-permutation gauge needs a quadratic transformed gate.
3. Independent input and output *field-linear* gauges preserve the raw
   coordinatewise product exactly only with a common permutation and two
   independent nonzero diagonal scales. Per-output bilinear ranks cannot be
   added as a lower bound on a decoder that shares products across outputs.

These results close only the stated local construction/proof questions. No
procedure in this package reduces arbitrary dense native projection work.
The permutation reference executes every original product and addition; the
bit-gauge gate adds XOR work to the original elementwise gate. O1-O5 remain
OPEN, O6 PARTIAL; THEORY_STATUS=NOT_ESTABLISHED, HARDWARE_STATUS=NOT_TESTED,
CORE_ADMISSION=false. The user reports no current hardware able to host 405B.

## 1. Independent gauges: constructive certificate or counterexample

Let A,B,C be invertible n-by-n matrices over a field F. The identity

```
C(x .* y) = (Ax) .* (By)     for every x,y in F^n
```

holds iff `A=Da P`, `B=Db P`, `C=Da Db P`, for one common permutation P and
two nonzero diagonal matrices. The three gauges need not be equal over a
general field. Over GF(2), all nonzero diagonal entries are 1, so A=B=C=P.

Proof. Evaluate on basis vectors x=e_j,y=e_k. For output row i the equations
are `A_ij B_ik=0` when j differs from k and `C_ij=A_ij B_ij` on the diagonal.
Both A and B have a nonempty row. Choose one nonzero entry in each row i; the
off-diagonal identity forces their column indices to coincide. Comparing any
other entry against that fixed nonzero partner forces all other entries zero.
Thus both supports are the same singleton. Invertibility makes these singleton
columns a permutation. The diagonal equation determines C. Direct substitution
proves sufficiency. Evaluating basis pairs also proves this over finite fields;
no invalid polynomial-identity inference from an infinite domain is needed.

`classify_gauges` in [producer.py](producer.py) checks prime-field invertibility
and these finite coefficients. It returns the O(n)-word permutation/diagonal
certificate or an actual basis-vector counterexample; singular triples are
explicitly outside the declared invertible domain. Termination: a bounded
elimination and at most n^3 coefficient checks. Construction is O(n^3) field
operations with O(n^2) input/workspace words, plus the stored certificate.
For an input prime p the actual code also pays up to floor(sqrt(p)) trial
divisions for primality validation and O(n) modular inversions; integer bit
costs depend on log(p) and are not constant-time for unbounded p. The controls
fix p to 2 or 3. This does not compile native checkpoints into a fast executor.

The independently evaluated finite coefficient controls checked every pair of
invertible A,B, with C forced by the diagonal equations:

| Field / n | A,B pairs | Accepted triples | Mismatches |
|---|---:|---:|---:|
| GF(2), 2 | 36 | 2 | 0 |
| GF(2), 3 | 28,224 | 6 | 0 |
| GF(3), 2 | 2,304 | 32 | 0 |

These are bounded validation of the proof, not its universal justification.
The field assumption excludes zero-divisor rings, rounded BF16 multiplication,
SiLU and arbitrary encodings.

## 2. Native column permutations must carry the reduction schedule

For BF16-representable weights `W=[2^24,1,-2^24,0]` and all-one input, separate
FP32 products followed by the fixed balanced FP32 RNE tree give

```
fl(fl(2^24 + 1) + fl(-2^24 + 0)) = 0.
```

Permuting weights and input together by `[0,2,1,3]` and then using a balanced
tree in the **new physical order** gives

```
fl(fl(2^24 - 2^24) + fl(1 + 0)) = 1.
```

FP32 output words are `0x00000000` and `0x3f800000`; final BF16 words are
`0x0000` and `0x3f80`. The algebraic equation `W'=Pout W Pin^-1` alone does
not prove native equality. This counterexample concerns the declared balanced
FP32 ABI; it is not an executed HF/cuBLAS/TensorCore counterexample.

The actual constructor stores a permuted exact FP32 layout, row order,
column order, and inverse-column leaf schedule. At original logical leaf j,
`execute_transport` reads the new physical column `inverse_order[j]` and the
matching encoded activation. The two operand words at each leaf are therefore
the original words. The same product primitive and the same padded reduction
tree return the same word at every node by structural induction. Finally,
`decode_rows` restores the observed row order. Each loop is finite. This proof
does not require associativity and applies to another pinned native primitive
only if its exact operations, order and side effects are transported too.

The 144 fixed cases (two rows, all column/row permutations, three inputs)
checked 288 output words: transported mismatches 0; naive permutation mismatch
cases 48. Additional focused tests cover non-power-of-two padding, signed zero,
and invalid permutation refusal. [Raw cases](results/native_cases.json).

For an m-by-n projection, p=next power of two >=n, the reference retains
exactly mn products and m(p-1) additions. Compilation reads and writes every
coefficient. The declared FP32 packed-data ledger retains 8mn bytes for original
and transformed coefficients, plus 4(2n+m) bytes for three uint32 index vectors
at n,m<=2^32. The Python implementation also pays Python objects/allocator
overhead; these packed-byte terms are not measured process/GPU peaks.
At n=m=16,384 this means 268,435,456 products, 268,419,072 additions,
2,147,483,648 packed source/layout bytes and 196,608 index bytes. Arithmetic
and source reads are 100% of the reference before extra gathers and copies.

## 3. A shared decoder defeats summing separate output ranks

For arbitrary invertible field gauges, define
`H(z,w)=C((A^-1 z).*(B^-1 w))`. Output i has bilinear matrix
`T_i=A^-T diag(C_i) B^-1`, so `rank(T_i)=nnz(C_i)`. This is a valid individual
output rank, but its sum is not a lower bound against shared products.

The frozen GF(2) example uses n=4, A=B=I, C=I+J. Here C is invertible because
J^2=4J=0 and `(I+J)^2=I`. All four output ranks are 3, summing to 12.
The actual decoder computes four products `p_j=z_j w_j`, then XORs the three
selected products in each output using eight XORs. All 256 input pairs match.
The example is implemented in `shared_control`, not supplied as an oracle.

General dense A^-1, B^-1 and C still cost three paid linear transforms. No
linear transform, coefficients, metadata, movement or output conversion is
free. This invalidates only the summed-rank argument; it neither supplies the
arbitrary dense producer nor proves all transformed gates expensive.

## 4. Continue construction: explicit native opaque-bit gauge

Rather than ending at that limitation, [native_bit_gauge.py](native_bit_gauge.py)
implements a non-coordinatewise transformed native gate. For n=2^k b-bit opaque
words, one butterfly stage XORs each upper-half word with its paired lower-half
word. Each stage is an involution because the lower word is unchanged and two
XORs cancel. Applying the stages in reverse order is a finite inverse of the
whole encoder E. Thus E^-1(E(x))=x for every raw bit vector, including encodings
that temporarily resemble infinities or NaNs. Encoded words are never treated
as floating-point operands.

The actual decoder is

```
EncodedMultiply(z,w):
    x = InverseButterfly(z)
    y = InverseButterfly(w)
    r_i = OriginalNativeMultiply(x_i,y_i), in original coordinate order
    return Butterfly(r)
```

For any deterministic native primitive F used unchanged, this returns
`E(F(E^-1 z,E^-1 w))` bit-exactly. This is not the field-linear raw-Hadamard
identity classified above. For a gate-only recurrent state, if z_t=E(s_t),
then the decoded-input equality gives z_(t+1)=E(F(s_t,x_t)) for every input;
initial equality yields induction for arbitrary continuations, including dense
changes. This native primitive and wrapper consume no RNG. Full HF RNG/state,
KV layout/aliasing, norms, RoPE, residuals and actual sampling are not tested
or implemented by the gate-only chain.

The CPU reference uses BF16 operands, separate FP32 RNE multiplication, BF16
RNE store and canonical quiet NaN. Actual other backend semantics are not
silently substituted. Every one of 65,536 BF16 words participated in a
four-coordinate roundtrip vector (262,144 word checks); 144 exceptional/finite
product vectors checked 576 outputs; eight independent known product words
and a 20-step encoded-state chain passed, with no observed mismatch and
unchanged Python RNG state. These checks support the constructive inverse
proof, not an all-HF state or target performance claim.

For already encoded inputs/output, the exact schedule has

```
3*n*k/2 word XORs + n native products,
data reads  = 2*(3*n*k/2) + 5n opaque words,
data writes =   (3*n*k/2) + 4n opaque words,
at most 6n live opaque data words including the encoded inputs,
plus index/control words, native conversion/store and allocator/system costs.
```

The data read/write formula includes the three transform copies and product
input/output words; it excludes instruction/index traffic, which remains paid
O(n log n). At n=16,384: 344,064 XORs, 16,384 native products, 770,048 data
word reads, 409,600 data word writes, and 98,304 live data words. Sixteen-bit
packed live data would be 196,608 bytes before control/allocator/Python objects.
This is an algorithmic scratch term, not a physical GPU/CPU peak measurement.

If the public interface supplies and requires **plain** words, the two input
encodes and final output decode add three more transforms: 688,128 total XORs
at that width, with a conservative 8n live-data-word bound for a literal adapter.
No benefit is credited from moving boundary transforms out of the count.

All CPU/RAM/SSD/PCIe/HBM/GPU/KV/state, initialization, code/index storage,
verification, repair/fallback and synchronization of a real integration remain
additional paid terms. This gate-only routine does not access checkpoint
weights or a cold sidecar and has no certificate oracle. **It is slower in
operation count than the original n-product Hadamard gate.** Its value is that
the transformed gate is now an explicit small program and rules out an
overbroad inference of a mandatory quadratic native gate cost.

## 5. Why the fixed mission remains open

The decisive missing object is still a paid representation and query program
for the arbitrary dense projection `E F_W E^-1`. Calling F_W on decoded input
retains every original coefficient and native operation, then adds encodes.
This package does exactly that if used around a dense projection; it supplies
no hidden fast version. Choosing a checkpoint-specific E that has both a cheap
inverse and a cheap transformed arbitrary native projection is an unresolved
construction, not an assumption or a newly admitted candidate.

The monomial theorem cannot be used to exclude every such byte/bit encoding;
conversely, this cheap gate cannot establish the needed dense projection. The
registered 403,747,897,344 non-embedding coefficients are still fully read in
the literal projection route. No >=90% dense removal, full cost closure, 8GiB
GPU fit, native4BQ4 p50/p95, TTFT or actual 405B run is established.

## 6. Reproduction, independent review and persistence scope

From this experiment directory with Python 3.10+ and standard library only:

```
python -m unittest test_producer -v
python run_audit.py --output-dir <new-empty-audit-directory>
python run_native_bit_gauge.py --output-dir <new-empty-bit-gauge-directory>
python verify_bytes.py
```

Seven focused tests passed. The canonical numeric results are
[gauge/schedule summary](results/summary.json) and
[bit-gauge v2 summary](bit_gauge_results_v2/summary.json). The initial bit-gauge
result is preserved in `bit_gauge_results/`; v2 adds explicit plain-interface
and copy-traffic accounting without changing the algorithm or thresholds.

The two generators explicitly serialize CRLF JSON/checksum lines, preserving
the original Windows evidence hashes across operating systems. Package-local
`.gitattributes` disables text conversion without changing global Git settings.
Before the fix, Git clean changed the canonical summary's blob from `fee3ce1`
to `067ace2`; after the fix both are `fee3ce1`. `verify_bytes.py --index` also
checks the staged bytes against every SHA-256 manifest entry. This repairs a
reproducibility defect; it changes neither the experiments nor admission.

The [independent review](INDEPENDENT_REVIEW.md) requested Sol/high and checked
the scoped theorem, sharing counterexample, native schedule witness and
opaque-bit construction. It did not execute duplicate experiments. Both Sol/high
and the earlier Luna/low extraction requests returned only task identifiers;
actual model selection could not be independently verified. Their agreement is
not proof; the basis coefficient argument, structural native proof and direct
artifacts are the evidence.

At startup the actual persistent Codex goal was registered, original head and
remote `9fad90e` matched, the original tree was clean, 10 prior direct-global
tests passed, and the original canonical JSON regenerated byte-identically at
SHA-256 `a907cb1dbd40b9a7bf71b88a159cf0358c4d3b0656470e71e26f37e2c9185ffe`.
Those are restoration checks, not this construction's new results.

The shared checkout subsequently advanced to `dab5774` and another writer
created uncommitted experiment files. A pre-write HEAD guard stopped the planned
restoration document update before any original file changed. This work was
therefore completed in a separate directory/checkout. The other writer's code,
results, Git index and original canonical artifacts are not overwritten.
The fixed mission, confidentiality rules, no direct main/force push and no
manually dispatched Actions remain unchanged. Persistence is not goal completion.
