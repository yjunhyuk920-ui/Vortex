# Nonlinear Router Frontier — preregistration

Date: 2026-09-08
Base: `866e347540c6aff7595ac81c2d2c6c0cf5680196`

The fixed VORTEX mission is unchanged. This round does not rerun PR #149 and
does not reopen the closed error-transport family.

## Recovered interrupted round

The unreferenced GitHub blob
`6a009505882dabf861b211d71f22ca46994da91d` records a bounded native causal
response attempt. It reports 24/24 exact 97-bit synthetic state transitions,
but 920 native calls versus 24 direct calls (19x to 50.25x by case). It also
records storage/cost rejection of a whole gate-state array and the first
nonzero explicit fused-SwiGLU moment tensor. These are recovered observations,
not yet repository evidence, and they will not be rerun merely to recreate
the text.

## Three materially different information-flow principles

### P1 — globally nonlinear bounded-word routing (checkpoint-space)

Reverse the premise that an exact query must read raw/linear checkpoint
coordinates. Compile arbitrary checkpoint functions into bounded words and
let the query choose/read only a small number of them. The strongest currently
visible concrete target is the already registered binary seed: encode a
`25 x 108` matrix into 50 padded 64-bit cells and answer every rank-one parity
with two probes. The encoder cells and decoder may be nonlinear.

Constructor obligation: an explicit map `E : F2^2700 -> ({0,1}^64)^50`, first
and second address functions, and a finite decoder. No lookup-table existence
claim counts as a constructor.

Stored representation: 50 words = 3,200 bits in the favorable local seed.

Runtime address rule: at most two word addresses. We will distinguish (a)
fixed/nonadaptive pairs and (b) a second address conditioned on the first word.

Exactness equation: for every matrix `W` and nonzero rank-one mask `q=r u^T`,
`D_q(E(W)[a], E(W)[b]) = <q,W> mod 2`, with the adaptive form permitting
`b=b(q,E(W)[a])`.

Termination: exactly two bounded word reads and finite Boolean postprocessing.

Paid favorable query bound: 128 physical bits. This is only a binary source
screen; native Q4/BF16/FP32 lifting, addressing bytes, metadata, build and
whole-model state remain additional.

Dense-effect route: if this object existed at the registered seed, it would
replace 2,700 source-bit incidences by 128 bits of payload, eliminating over
95% of the source traffic before native lifting. It is therefore structurally
capable of clearing the >=90% entry screen rather than merely renaming a full
scan.

Strongest adversary: all `2^2700` binary matrices and the complete rank-one
query set, including matrix units.

Cheapest falsifier registered for this round: use exact Segre/product-code
intersection geometry to test whether fixed pairs, and then affine first-word
adaptive routing, can cover the complete rank-one query family. This must
allow arbitrary nonlinear payload cells and arbitrary final Boolean logic.

### P2 — joint factor-envelope restriction production (query-space)

Reverse the unit of execution from one scalar/matrix query to the full causal
query envelope. For rank-one queries `q_i=r_i u_i^T`, form `R=span(r_i)` and
`U=span(u_i)` and request `W` only through its restriction to `R tensor U`.
For 32 queries this restriction has dimension at most 1,024.

The exact interface is:

`Compile(W) -> C_W`, `(R,U,C_W) -> exact W|_(R tensor U)`.

The stored representation, address generator and decoder must be explicit and
near-source-size. Runtime must output the restriction before evaluating the
original per-query dense contractions. Exact downstream native accumulation
order must then be preserved.

Termination is finite once a generator exists. The required query payload is
at most 1,024 exact summaries plus charged metadata. A literal envelope catalog
is already rejected, and evaluating `R^T W U` from raw `W` is a full/dense
contraction and is therefore not a qualifying constructor.

Dense-effect route: a genuine generator would share one restriction across 32
queries and could remove >=90% of repeated dense effects. No such generator is
currently constructed.

Strongest adversary: independent 32-dimensional left/right factor spaces at a
`16,384 x 16,384` arbitrary dense block. Cheapest falsifier: any proposed
generator whose address or construction expands to a literal factor-space
catalog or reads/contracts the original block is rejected before execution.

This principle is kept as an open interface, not promoted as a candidate with
an undefined generator.

### P3 — native accumulator transition-map composition (arithmetic-space)

Reverse the premise that reference-ordered rounded accumulation must execute
one scalar contribution at a time. A fixed checkpoint chunk induces a finite
map from incoming accumulator word and activation chunk to outgoing accumulator
word. Compose chunk maps while preserving the reference order exactly.

Constructor interface:

`chunk weights -> finite transition representation T`, then
`T(accumulator_word, activation_chunk) -> exact successor accumulator_word`.

Exactness is by equality with the reference finite-word transition at every
chunk boundary; composition terminates after the fixed chunk count. RNG and
causal state are unchanged until the containing native operator completes.

The representation and address rule may not be an exhaustive activation/word
answer table. If it evaluates every activation-coordinate contribution to
construct `T_x` at query time, the original dense work remains and the
principle fails the entry screen. If it materializes all activation patterns,
it becomes the already rejected exponential finite-transition table.

Dense-effect route: only a succinct activation-parameterized transition map
could eliminate >=90% of contribution processing. No such map is currently
constructed.

Strongest adversary: a chunk whose BF16 activation coordinates vary
independently and whose reference FP32/BF16 accumulation exhibits rounding
non-associativity. Cheapest falsifier: count the activation-pattern catalog or
the runtime contribution evaluations of the first explicit representation.

This principle is not promoted unless it supplies a non-table, sub-dense
transition representation.

## Selection

P2 and P3 currently hide exactly the missing object if promoted. P1 is the
only principle with an existing finite target whose generic capacity test does
not already rule it out. This round therefore attacks P1 without assuming a
decoder.

The first theorem targets two powerful submodels that were not closed by the
generic degree-capacity screen:

1. **nonadaptive nonlinear words** — both cells may be arbitrary checkpoint
   functions, but the pair of addresses is fixed by the query;
2. **affine-key adaptive routing** — the first 64-bit cell is affine in the
   checkpoint, the second cell is arbitrary nonlinear, and its address may
   depend on the first value.

For each submodel, derive an exact finite cover implication and combine it with
the exact maximum number of rank-one masks in a low-dimensional subspace.

No claim about the fully nonlinear adaptive-first-word model is preregistered.
If that model remains after the theorem, it remains OPEN rather than being
silently declared feasible or impossible.

## Status before execution

```text
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false
```

405B, <=8 GiB GPU, CUDA, PCIe, SSD, native 4B Q4 p50/p95 and TTFT are not part
of this E0 theorem run and remain `NOT TESTED`.
