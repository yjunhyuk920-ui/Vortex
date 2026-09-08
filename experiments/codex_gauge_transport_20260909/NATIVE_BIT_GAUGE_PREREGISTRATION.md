# Follow-on: explicit native bit-gauge gate — preregistration

This follows the failed naive native column-permutation claim and continues the
actual missing transformed-gate construction, not a duplicate table/Gauss-Jordan
experiment. It is bounded auxiliary work: an O3 local native gate and state
correspondence primitive, not the missing arbitrary dense projection or O1-O6.

Define E on n=2^k opaque b-bit words by k butterfly stages. At stage j, for
every index s with bit j set, XOR word s with word s xor 2^j. Each stage is an
involution; inverse executes the same stages in reverse order. No floating
arithmetic interprets encoded words. E is globally mixed over coordinates and
is not an invertible *numeric field-linear* gauge, so the monomial theorem
does not apply to it.

Construct, without any answer catalog,

```
EncodedMultiply(z,w) = E(NativeMultiply(E^-1(z), E^-1(w))).
```

The native callback executes once per original coordinate in the same order.
Exactness is literal bitwise inverse and re-encoding. The same construction
maintains an encoded gate-only recurrent state on arbitrary successive inputs,
including every-coordinate changes; no sparse-delta assumption or future input.
The operation consumes no RNG, and its wrapper likewise consumes none.

Paid schedule: three transforms, exactly 3*n*k/2 word XORs, n native products,
all conversion/store costs of that native primitive, input/output copies, fixed
index calculations, and bounded O(n) opaque-word scratch. The map is procedural
and checkpoint-independent; there is no checkpoint program, free decoder, or
expanded cold sidecar. This adds work over the original Hadamard gate. It is
only a concrete example of a cheap non-coordinatewise transformed gate.

Crucial unresolved dependency: there is no construction showing that this E,
or a checkpoint-derived E with paid cheap inverse, simplifies arbitrary native
dense `E F_W E^-1`. Calling the original dense projection inside that expression
retains 100% dense cost. No native4BQ4 or >=10x core is admitted.

Frozen CPU ABI: BF16 word inputs, separate FP32 RNE multiplication, BF16 RNE
store, canonical quiet NaN. Actual arbitrary HF/CUDA NaN/reduction semantics
are not asserted. The encoding proof would preserve a different native callback
only if that actual callback is invoked unchanged and its complete effects paid.

Frozen checks: all 65,536 BF16 word patterns in 4-coordinate structured vectors
for forward/inverse equality; pairwise product controls from 12 fixed raw words
including signed zero, subnormal, finite extrema, infinities and NaN (144 vectors,
4 outputs each); a deterministic 20-step gate-state chain, comparing each decoded
state; independent known BF16 product words for a small set of exact products.
Output deterministic JSON/hash, no performance/GPU claims. Target hardware is
unavailable per user; fixed mission and all prior rejection scopes unchanged.
