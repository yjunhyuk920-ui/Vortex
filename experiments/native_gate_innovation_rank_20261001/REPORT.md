# Exact innovation rank of the native BF16 product cut

## Result and scope first

For m independent BF16 elementwise product lanes, with independently legal
finite operands and **every product output word preserved**, the span of legal
second additive differences over F2 has dimension exactly **15m** under the
signed-zero-preserving multiplication ABI below. Therefore an exact fixed
bit-affine backbone plus a fixed nonlinear output injection

    bits(P(x)) = c XOR L x XOR B eta(x)

requires rank(B) >= 15m. This holds even when eta is an arbitrary function and
L/B are constructed from the source at unlimited computational expense. The
witnesses need only +0, +1 and 16 explicitly described nonnegative normal/zero
BF16 words, and are exact without a transcendental evaluation or rounding test.
A bit-linear change of coordinates cannot reduce this rank.

This settles a concrete necessary native-extension hypothesis: a constant-rank
or o(m)-rank fixed additive innovation cannot reproduce this exposed gate-product
cut on its independent-input ABI. It does **not** establish that the operands
are jointly reachable in the preserved HF model, that the composed HF graph
has this rank, or that a high-rank gate is expensive. No new execution core,
original-HF acceleration, source producer or whole-mission obligation is closed.

THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED;
CORE_ADMISSION=false; FULL_MISSION_O1_O6=OPEN; HANDOFF_STATUS=IN_PROGRESS.

## 1. Exact source boundary and why this is additional evidence

The preserved native graph contains, in order:

    silu = aten.silu.default(gate_projection)
    up = original_up_projection(...)
    mul_12 = aten.mul.Tensor(silu, up)
    down = original_down_projection(mul_12)

See [original graph, lines 139-149](../native_global_transition_20260908/results/probe_graph/graph_0_before.txt).
The [serialized graph metadata](../native_global_transition_20260908/results/probe_graph/graph_0.json)
pins all three relevant projection weights to torch.bfloat16; the original
BF16 projection/SiLU/multiplication chain contains no dtype conversion between
these operations. The input tensors and product at this cut have BF16 dtype in
the declared reference. This note concerns **mul_12 alone, after activation**. It neither
substitutes parity for its multiplication nor claims a theorem for the preceding
SiLU or either dense projection. It introduces no model-specific weight change.

The [Krylov report](../paid_krylov_feedback_20261001/REPORT.md) gave the
second-difference test but no native gate witnesses. The
[signed-orbit report](../signed_orbit_20260907/REPORT.md) treated a real analytic
SiLU transport form; it did not prove this native-bit fixed-port rank. The
[native response-code result](../native_response_code_20260907/REPORT.md)
constructed finite whole-block response matrices at paid enumeration cost; its
rank is a different row/query interface. The
[nonlinear-coordinate report](../nonlinear_coordinates_20260908/REPORT.md)
explicitly retains arbitrary nonlinear-coordinate escapes. None of these scope
boundaries is revised here. No width/seed/activation sweep is repeated.

This is bounded O3 hypothesis falsification and an O6 checkable certificate,
not a new core-selection round. The existing source comparison is unchanged:
fixed additive ports receive this new boundary; nonlinear coordinates still
lack a paid generic constructor/observer; joint whole-program native sources
still lack an affordable causal producer. These are not three newly qualifying
execution principles.

## 2. Original-native cut, legal domain and complete roots

Let BF16 words have one sign bit (bit 15), eight exponent bits (bits 7-14), and
seven fraction bits (bits 0-6). Let D_m contain independently supplied pairs
(a,b) of length-m BF16 vectors with no infinity or NaN input. Signed zeros and
subnormals are admitted. Define P_m(a,b) as ordinary elementwise multiplication
with BF16 outputs, round-to-nearest-even, preserved product sign including
signed-zero underflow, nontrapping arithmetic and no externally observed FP
status flags. Finite products may overflow to signed infinity; finite input
multiplication generates no NaN. A native FP32 promotion/product followed by
BF16 storage also meets the proof: its sign is the input-sign XOR and every
lower-bound witness is exact before both stores.

The lower bound itself only uses nonnegative normal inputs or +0, and normal
outputs or +0, so it does not depend on a transcendental implementation,
subnormal handling, overflow or tie rounding. The matching **15m upper bound**
uses the stated signed-zero/product-sign rule over D_m. This is a symbolic ABI
claim, not newly measured cross-platform conformance. A different ABI must be
checked against those rules; the earlier Windows/Linux replay difference is
not overridden.

A replacement of the exposed product subprogram returns all m BF16 product
words with the original shape/layout contract. This pure operation consumes
no RNG and mutates neither operand nor unrelated live tensors. For an explicitly
state-complete local interface, include unchanged a,b, every unrelated state
word q, and RNG R in the result:

    H(a,b,q,R) = (a,b,P_m(a,b),q,R).

Shapes/layout/control are fixed at the cut or passed through unchanged; output
allocation and writing are still paid. Every witness holds q,R and metadata
fixed. Their second differences vanish in all unchanged roots. Thus H has the
same 15m second-difference rank; nothing is gained by ignoring these roots.
If additional non-affine observable roots are added, the product projection
still gives a >=15m lower bound, but this note does not assert equality then.

D_m is justified as a standalone multiplication-call ABI, not as the set of
reachable inputs to this call from the released checkpoint. In the HF program,
a is the preceding SiLU result and b is the up projection of the same hidden
state. The note has not shown even the small witness set is jointly attainable
from legal prompt/cache histories. Treating them as independent reachable HF
states would be invalid.

## 3. The exact fixed-port hypothesis

Concatenate all original input bits into x. Allow a constant c, any fixed binary
linear L, any fixed binary linear injection B of rank k, and any function eta:

    H_bits(x) = c XOR Lx XOR B eta(x),  for every legal x.

Taking L on **all** input bits is more permissive than a backbone linear only
in persistent state. All source/checkpoint-dependent constants may reside in
c,L,B; they must be fixed across the legal queries being considered. Eta has
no computational restriction for this necessary condition. If even this broad
form fails at a proposed k, a cheap eta cannot repair it.

This hypothesis includes the full observed product in the linearly decoded
output/state relation. It does not cover hiding the product in a separate
arbitrary nonlinear observer. For example, retaining the operands with an
identity transition and multiplying them in an observer has zero transition
innovation but still pays the original product in that observer. Counting only
the transition rank would miss that work. The relevant claim must either apply
the representation to the joint observer/state map or give that nonlinear
observer's actual source and cost separately.

## 4. Exact lower bound: 15 scalar rectangles, then a direct sum

For any legal affine quadruple x, x XOR u, x XOR v, x XOR u XOR v, define

    Delta(H;x,u,v) = H(x) XOR H(x XOR u)
                    XOR H(x XOR v) XOR H(x XOR u XOR v).

The c and L terms cancel. Therefore Delta is in im(B), and the dimension of the
span of all such legal differences is at most rank(B). The four points, not
just two arbitrary legal samples, must be legal in the same domain.

For one lane set

    a0 = 0x0000  (+0),   a1 = 0x3f80  (+1),
    b0 = 0x4000  (+2),   bj = 0x4000 XOR (1 << j),  j=0,...,14.

All four pairs (a0,b0), (a1,b0), (a0,bj), (a1,bj) form an affine rectangle in
the 32 original operand bits. Their exact product words are

    0x0000, b0, 0x0000, bj.

Hence their second difference is b0 XOR bj = 1 << j.

Legality is explicit. For j=0,...,6, bj has exponent 128 and one fraction bit.
For j=7,...,13, its exponent is 128+2^(j-7), at most 192, with zero fraction.
For j=14, bj is +0. These are all finite and nonnegative; nonzero values are
normal. Multiplication by +0 or +1 is exactly representable, with no overflow,
underflow or rounding ambiguity. In particular, the j=14 witness includes +0;
it is not described as an all-nonzero witness.

For m lanes, place each scalar rectangle in lane i and hold every other operand
lane at +0. Its difference is the unit vector e_(16i+j) in the product output,
zero in every other lane and unchanged root. The 15m vectors are independent
because they occupy distinct bit positions. Thus

    dim span {legal Delta(H)} >= 15m,
    rank(B) >= 15m.

This is a proof for every positive integer m, not an extrapolation from a
larger numerical sweep. Only the 15 scalar rectangles need be listed.

## 5. Matching upper bound, constructor and paid baseline

For all finite inputs in the stated multiplication ABI,

    sign(P_i(a_i,b_i)) = sign(a_i) XOR sign(b_i).

This covers signed zeros, underflow and overflow. Thus the sign output bit is
linear, while all nonlinear output variation lies in the other 15 bits of each
lane. All second differences therefore lie in the fixed magnitude-bit subspace,
whose dimension is 15m. With the lower bound,

    dim span {legal Delta(H)} = 15m.

The fixed-port rank is also exactly minimized at 15m: choose L to copy unchanged
roots and compute the output sign XOR, choose B as the canonical injection of
the 15 magnitude bits per lane, and let eta return the original product's
magnitude bits. This is a genuine exact representation, but its explicit eta
executes the original m elementwise products; it supplies no saved producer.

The corresponding finite constructor validates the existing pure multiplication
node, dtype/shape/alias/observer contract, emits the sign-XOR and canonical
magnitude injection, and retains the original multiplication operation for eta.
B is structurally implicit, so it need not allocate a dense (16m)-by-(15m)
binary matrix. The causal update reads both current operands, executes m
original products, extracts/retains the magnitude bits, combines with sign bits,
and writes the original outputs. The observer returns these exact words;
unchanged live state/RNG and metadata retain their original relation.

An unfused direct implementation pays 4m input bytes, 2m output bytes and m
original multiplies, plus sign/mask logic, node/shape validation, allocations,
layout operations and any materialized eta/intermediate buffer and its traffic.
No copy of unchanged logical state is required if original alias/read-only
semantics permit passing it through; otherwise the required copies are paid.
This is an O(m)-work baseline with no dense projections removed. It is not
implemented as a new executor, because merely adding this representation has
no identified saving and no target-qualifying bound.

Importantly, rank 15m alone is **not** an Omega(m^2) computation, dense-B storage,
full-source-read or latency lower bound. This sparse injection illustrates why:
a high-rank innovation can still be cheap to write and apply. The scalar
multi-chain O(kr) bound in the paid-Krylov discussion loses a small-k premise
here, but no lower bound on all structured recurrence schedules follows.

## 6. Exactly which changes escape the exclusion

1. Fixed invertible affine changes of input and output bit coordinates preserve
   the legal quadruples and their span dimension. Injective linear output
   embeddings preserve independence as well. In a linear chart that represents
   all witness points, using an affine input encoding also preserves the
   quadruple relation. Thus source-derived linear rebasing alone does not give
   a narrower port for this cut. A source-dependent fixed chart is still fixed.
2. Arbitrary nonlinear state coordinates, nonlinear output decoders/observers,
   query-dependent injection images or multiple selected charts are not covered.
   They may relocate the product into encoding, observation or chart selection;
   all those algorithms and costs still need construction. This rank theorem
   neither proves them impossible nor supplies a fast one.
3. A checkpoint-specific reachable-domain invariant might exclude some witness
   rectangles. It must prove the exact all-continuation relation; independently
   legal operands cannot substitute for that proof. A smaller reachable domain
   could have lower innovation rank, but it is not established here.
4. The whole HF graph observes a down-projected/continued result, not necessarily
   each product word. Composition can erase independent differences; after a
   nonlinear suffix even transporting these XOR differences linearly is invalid.
   This theorem cannot be applied through the down projection by fiat. A native
   linear suffix would preserve only the rank of its image of these differences;
   no such bit-linear suffix is asserted for original rounded dense MatVec.
5. A wide, structured innovation with a cheap original-source producer remains
   allowed. The missing issue is whether its producer plus the full observer and
   successor-state path can avoid the dominant dense native work with a paid
   bound. Calling B full-width does not construct that producer, but neither
   does full rank alone reject it.

The global mission quantifiers, full logits/KV/layout/RNG obligations and target
budget remain unchanged. This result excludes a representation on an explicit
primitive domain, not arbitrary encodings, LLM executors or possible speedups.

## 7. Validation, status and exact remaining question

[PLAN.md](PLAN.md) records timing honestly: symbolic discovery preceded the
protocol; the exact-word certificate followed it. [verify.py](verify.py) uses
only standard-library integers and Fraction. [certificate.json](certificate.json)
contains every word, exact rational product, zero input-XOR, output basis word,
rank 15 and hashes of the pre-existing sources. It does not execute PyTorch,
SiLU, a model, a GPU kernel or a floating-point backend. The universal-width
result and sign upper bound are symbolic arguments, not empirical measurements.

The remaining useful source question is now narrower: supply a genuinely
source-derived reachable relation or a composed/nonlinear observer-state
representation that avoids the independent exposed-product hypothesis, and
construct its producer and total paid cost. Repeating these witnesses, expanding
the width, testing another seed or rerunning the HF graph would not supply it.

README was reviewed; its mission-not-achieved and missing native producer
statements remain accurate. The parent may add a bounded-result link after
review. No root-ledger/runtime-core/remote changes are made by this worker;
shared dirty files are preserved. The parent independently read the proof and
source cut and reran separately written exact Fraction/bit checks of all 15
rectangles; it reported no material correction. The
[independent review](INDEPENDENT_REVIEW.md) is proof/certificate review, not
native-backend conformance. No independent hardware or runtime test is claimed.
