# Direct global finite-word producer frontier — preregistration

Date: 2026-09-09
Parent scientific commit: `6021ff11ca313f304511ceecaa38ac75af22bf8c`

The fixed VORTEX mission and CTC O1--O6 are unchanged.  Completed nonlinear
router, causal KV/right-factor, and Boolean-semiring lift evidence is frozen and
will not be rerun merely to reconfirm it.

## Intended final theorem

For every public unmodified dense HF checkpoint in mission scope, every legal
batch-1 input/continuation and RNG state, construct in finite time a paid
representation `G` and causal program

```text
Compile(checkpoint bytes) -> G
Step(G, input_t, exact_state_t, rng_t)
    -> exact native logits_t, exact_state_(t+1), exact rng_(t+1)
```

such that every selector/address/decoder/state transform is explicit, all
construction/storage/RAM/SSD/PCIe/HBM/GPU/workspace/arithmetic/metadata/
verification costs are charged, the core removes at least 90% of original dense
parameter-effect/query work, and the final sufficient bounds close the fixed
native-4B-Q4 p50/p95 and TTFT contract.  This round is not allowed to promote a
binary necessary condition into native completion.

Frozen registered population used for E0 accounting:

```text
non-embedding coefficients       403,747,897,344
dense matrices                   883
global hot-advice ceiling bits    68,719,476,736   (8 GiB)
binary advice/source ratio              17.020392%
Q4 advice/source ratio                   4.255098%
64-bit global source+advice cells    7,382,302,720
```

405B/CUDA/target <=8 GiB physical execution, PCIe/SSD/HBM service, native 4B Q4
p50/p95 and TTFT remain `NOT TESTED` unless actually executed later.

## Principle P1 — globally coded native reconstruction

Reverse the premise that a global code must be divided per matrix.  Encode the
entire checkpoint jointly with a systematic erasure/network code.

Finite constructor:

1. bit-preserve every checkpoint tensor and metadata word in one canonical
   source string `X`;
2. split `X` into fixed 64-bit cells;
3. generate deterministic XOR parity cells from a fixed full-rank binary
   generator until the complete 8-GiB advice allowance is filled;
4. retain raw `X` in paid cold storage and parity cells in paid advice storage.

Stored representation: raw checkpoint bytes, parity cells, generator metadata,
native layout/ABI metadata.

Runtime address rule: for each requested native weight tile, choose the
lexicographically first erasure pattern recoverable from the parity equations;
read the remaining raw cells, solve the finite GF(2) system, reconstruct every
needed original word bit-exactly, then execute the original native reduction
order.

Exactness/state equation:

```text
Decode(Encode(X), reads) = X exactly
=> Step_P1(X,s,rng) = Step_reference(X,s,rng)
```

and therefore state/RNG correspondence is inductive from equal initialization.
Termination follows from finite Gaussian elimination and the finite reference
forward.

Paid upper bound includes `|X|` cold bits, <=8 GiB parity, generator metadata,
all raw reads, parity reads, decode XORs/workspace, reconstructed tile bytes,
and the unchanged native dense arithmetic.

>=90% route would require the global code to replace >=90% of cold source
information *and* avoid >=90% of dense arithmetic.  No such grant is assumed.

Strongest adversary: an incompressible arbitrary checkpoint with independent
64-bit source cells.

Cheapest falsifier: information conservation for exact reconstruction.  If the
advice contains `R` bits, any fixed run that reconstructs `N` arbitrary source
bits needs at least `N-R` additional source bits in the worst case; moreover the
declared decoder still executes the original dense arithmetic.  Failure of
either >=90% axis rejects P1 without implementing a codec kernel.

## Principle P2 — exact dyadic producer plus native-order rounding witness

Reverse the premise that native non-associative reduction must be performed
while generating every product.  Separate the mathematically exact dyadic dot
from the information needed only to reproduce the reference rounding order.

Finite constructor:

1. decode each finite checkpoint word to its exact sign/exponent/mantissa
   dyadic integer form;
2. compile an exact shared integer/bit circuit for the mathematical dense sums;
3. compile, per reference reduction tree, a finite `RoundWitness` transducer
   whose input is the exact leaf-product stream and whose output is the exact
   reference finite-word accumulator/result.

Stored representation: exact-sum circuit, original checkpoint or lossless leaf
source needed by it, `RoundWitness` program, ABI/reduction-tree metadata.

Runtime address rule: evaluate only circuit gates and witness nodes reachable
from the current dense input.  No desired result/future token is supplied.

Executable query algorithm:

```text
S_exact = ExactCircuit(W, x)
rho     = RoundWitness(W, x)
y       = NativeDecode(S_exact, rho)
```

The witness is *not* a free primitive: its construction/evaluation must be
specified as a finite program and charged.  The reference equality is literal
finite-word equality at every dense boundary; equal exposed state then gives
the normal RNG/state induction.

>=90% route: exact-sum circuit + witness evaluation/traffic together <=10% of
the original dense effect, with finite representation/storage.  The previously
rejected static-circuit result is not reopened unless the witness supplies a new
order-information mechanism rather than merely replaying the dense tree.

Strongest adversary: two legal leaf streams with the same exact dyadic total but
different reference rounded result, plus random high-entropy row weights that
remove static sharing.

Cheapest falsifier: establish whether exact sum plus any declared compact
witness statistic actually determines the native result.  If the proposed
witness must replay one state transition per original contribution or materialize
the full transition table, P2 is rejected as the prior dense/transition family.

## Principle P3 — query-direct global nonlinear word router

Reverse both reconstruction and associative-sum premises.  Store checkpoint-
dependent words whose only purpose is to answer the *current required dense
effects* directly.  Cells may mix matrices and layers, and addresses may depend
on earlier returned values.

Concrete finite compiler searched in this round:

```text
source X = all registered binary checkpoint coordinates
cells     = S deterministic 64-bit functions of X
query     = simultaneous full-Mv right factors for all dense blocks
output    = every block-local GF(2) M_i v_i
```

The constructor is not allowed to say "find good nonlinear cells".  The first
subconstruction is a theorem-guided route compiler: partition simultaneous
query tuples by actual adaptive address transcripts; on every final route store
the finite cell functions and decoder truth tables explicitly.  If route/cell
counts are target-infeasible, the compiler is rejected rather than treated as a
free existence oracle.  If a compact route survives, its cell functions must be
materialized by a finite Boolean circuit whose bytes and evaluation cost are
added before promotion.

Runtime address rule: deterministic adaptive word addresses from current query,
state and previously returned words only.  Decoder returns the complete binary
dense-effect vector; no raw matrix probe is free.

Exactness equation on the E0 binary subclass:

```text
D(q, transcript_E(X,q)) = (M_1 v_1, ..., M_B v_B)
```

for every binary source and every simultaneous query tuple.  A surviving binary
producer still requires an explicit native finite-word lift; bit-plane naming is
not that lift.  Native promotion requires literal logits/KV/state/RNG equality
under the reference reduction ABI.

Termination: bounded route depth `t` plus finite decoder.  Paid costs include
all cell bytes, cell-construction circuits, addresses, probes, decoder work,
raw backing state if retained, native lift and state/KV.

>=90% route: total probe/decoder/cell-evaluation work and physical payload must
be <=10% of the original dense effect before attempting final 4B-Q4 closure.

Strongest adversary: independent arbitrary dense matrices and independently
selectable nonzero right factors for every dense block.  The causal legitimacy
of that complete tuple family is a separate obligation; no interpolation from
the existing small causal bridge is permitted.

Cheapest falsifier/new theorem target: generalize the largest-fiber route-span
argument to **vector-valued simultaneous queries with globally mixed cells**.
For one route, all component coefficient masks must have total linear span
`<=t*w`.  This directly charges global advice synergy instead of dividing advice
per matrix.  Compute the exact registered finite bound before attempting route
synthesis.

## Frozen comparison / execution order

P1 has the strongest native exactness but no plausible arithmetic removal; its
information/compute gate is cheapest and runs first.

P2 is the strongest arithmetic-native idea.  It is promoted only if the
rounding witness is genuinely smaller than dense replay/transition tables.

P3 is the highest-upside direct data-structure principle because it avoids both
weight reconstruction and arithmetic regrouping.  Its global vector-route
theorem is derived after P1/P2 screening.  If the theorem merely fails to reject
the target, that is **not** a constructor; route synthesis/materialization remains
mandatory.

Initial status:

```text
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
O1 OPEN
O2 OPEN
O3 OPEN
O4 OPEN
O5 OPEN
O6 PARTIAL_PRIOR_EVIDENCE_ONLY
```
