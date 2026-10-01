# Bounded native gate innovation audit

2026-10-01 UTC. Local HEAD da2fa6c46163e89b5349395c0916b6bd26ead67e,
branch research/cloud-continuation-20260930. Parent owns remote integration;
shared dirty files and root ledgers must remain unchanged.

This protocol is saved **after the symbolic witness was derived**, but before
running the exact-word certificate. It is not retrospective preregistration of
a numerical discovery. There is no parameter search, timing threshold or model
run. This is an auxiliary test of the existing paid-Krylov native-extension
hypothesis, not a newly admitted core or three newly qualifying principles.

## Fixed question and expected decisive certificate

For the original BF16 elementwise multiplication subprogram, with independently
legal finite operands and every output word exposed, can a fixed bit-affine
backbone plus a rank-k nonlinear injection have k sublinear in lane width m?

Prove/refute the precise form P(x)=c XOR Lx XOR B eta(x), where L and B are fixed
binary linear maps, eta can be arbitrary, and rank means binary output-image
rank. Include unchanged state and RNG, or explicitly track a nonlinear observer
outside this form. Source-dependent matrices are allowed; query-dependent
matrices and arbitrary nonlinear encodings are distinct representations.

The proposed finite witnesses are, in one lane, a in {0x0000,0x3f80},
b in {0x4000,0x4000 XOR (1<<j)}, j=0,...,14. All b words are positive normal
BF16 or +0. The 15 additive output differences should be the 15 magnitude bit
vectors. Rank below 15 refutes that specific single-lane fixed-port hypothesis;
disjoint-lane placement then proves 15m without a width sweep. The product sign
is affine under the declared signed-zero-preserving finite-input multiplication
ABI, giving a matching span upper bound.

## Frozen boundary

Source link: ../native_global_transition_20260908/results/probe_graph/graph_0_before.txt,
lines 139-149. The observed original graph has SiLU, the up projection, then
aten.mul.Tensor(silu, _unsafe_view_5), then the down projection. Inspect existing
signed-orbit, native-response-code, nonlinear-coordinate and whole-program
reports first; do not reframe their existing obstructions as a new result.

This cut permits independently supplied operands. Membership in this primitive's
input ABI does NOT prove joint reachability from the checkpoint's projections,
SiLU, token histories, cache or RNG. The complete graph need not expose this
intermediate. Neither whole-HF nonlinearity rank nor a speed lower bound follows.

Native contract for the symbolic statement: finite BF16 input words, ordinary
IEEE-style multiplication and BF16 round-to-nearest-even output, signed zero
preserved, nontrapping arithmetic, no externally observed FP flags. Native FP32
promotion/product followed by BF16 store is also covered: the witnesses are
exact representable normal/zero products. Overflow is allowed to yield signed
infinity; finite-input multiplication produces no NaN. No transcendental kernel
is evaluated or silently replaced by a mathematical sigmoid.

## Checks and stop

Use Python standard-library exact integers/Fraction only to check the 15 legal
quadruples, exact representability and GF(2) certificate rank. No Torch/model,
CUDA, backend, wider toy sweep, raw replay or target benchmark. Preserve source
hashes and complete small certificate. Seek independent symbolic review. Do not
implement a new executor: after the fixed-port question is settled, report the
precise reachable-domain/composed-observer or representation escape that remains.

Full O1-O6 remain OPEN; THEORY_STATUS=NOT_ESTABLISHED;
HARDWARE_STATUS=NOT_TESTED; CORE_ADMISSION=false; HANDOFF_STATUS=IN_PROGRESS.
