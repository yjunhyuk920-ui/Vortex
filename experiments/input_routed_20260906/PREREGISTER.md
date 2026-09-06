# Input-routed native-output source — registration before execution
Date label: 2026-09-06 (session date). Parent: yjunhyuk920-ui/Vortex @
147dad60b4450d7f32de6e3b24d276b075aed817, PR #128 (open/draft).

## Mission theorem, not claimed
For every original publicly released dense HF checkpoint, every legal causal
input/state/RNG, a uniform constructor and executor must preserve the original
observable output and required successor state while using <=8GiB total GPU
allocation, same-machine native4BQ4 p50<=1.2x, p95<=1.5x, existing TTFT.
No changed weights, quantization, retraining, free construction/fallback or hidden compute.

## Three premise-reversing proposals, screened before choosing
A. Compile the observable output's Boolean residual functions, not the coefficient
stream. Query chooses one child per node; multiple output bits share a sparse
per-query memo. The graph may be cold-backed, with *all* node/edge/root/input/memo
traffic charged. f=(not b) f0 OR b f1 is exact for the final finite-word output.
Unlike the previous bitplane source, a query does not sweep the encoded weights.
Concrete compiler: hash-consed Boolean gates over ripple-carry fixed-coefficient
multiplication/addition followed by native BF16 rounding, then reachable-node
compaction and actual serialization. No truth-table construction, previous-input
cache, numerical similarity, perfect selector, or original weight array at query.
>=10x route requires 12*visited_nodes+root_bytes <=0.1*2mn (before further costs),
and a small enough cold program plus constructible native/state lifting. No small
DAG assumption is granted. Strongest risk: program expansion and address traffic.
Cheapest test is a predeclared exact-integer slice of native BF16 projection.

B. Replace value construction with exact residue/rounding-cell identification.
The residues and a bounded native-rounding defect could determine a BF16 output
without all mantissa bits. But each residue still needs an explicit source; reading
all residue coefficients or evaluating an undefined modular matvec is not <=10%.
Do not implement another bitplane/Mailman variant here. This proposal is withheld
until an actual sublinear residue source, including rounding defect, exists.

C. Solve the causal step as bidirectional finite-word constraints rather than a
forward schedule. Backward propagation from output/RNG/state relations could
collapse unused forward dependencies. A finite bit-vector circuit plus DPLL is
concrete, but coefficient/constraint reading and worst-case search are unpaid by
any current <=10% bound. No theorem permits making exact successor state free.
Do not build a SAT search backend without a small native propagation witness.

These are different proposed information flows, not claims of three newly
invented academic fields or three admitted cores. A has the most concrete missing
source and is selected for a *bounded auxiliary* construction test; no core is
admitted merely from its hypothetical route. Decision diagrams and word circuits
are prior art. We do not re-open a universal entirely hot circuit (F-040), do not
build all FP32-state transition truth tables, and do not enlarge a closed pattern
lookup sweep. The new tested dependency is input-selected cold code access.

## Frozen pilot scope and native lift
Weights: nonzero signed integers with |w|<=127, exactly representable in BF16.
Inputs: canonical signed integers in p=4 bits, also exactly BF16, not quantized
from arbitrary activations. These are explicit restricted test inputs only.
Require sum_j |w_j|*2^(p-1) < 2^24 per row. Thus every separate FP32 product and
any binary FP32 reduction is exact and finite. Starting reduction at +0 makes
exact cancellation +0; -0 inputs are outside this restricted canonical encoding.
Final output is RNE BF16, constructed symbolically. This does NOT claim a generic
CUDA, arbitrary BF16, Transformer, KV or RNG lift. Rejection here does not prove
all larger/native programs fail. No target hardware benchmark.

## Frozen cases and stopping
(n,m,max_abs_weight) = (2,4,7),(4,4,7),(4,16,7),(8,4,7),
(8,16,7),(16,4,7),(4,4,127),(8,16,127).
Seeds 271,811, p=4. Two orders frozen: input-major LSB and bit-plane-major LSB.
Total 32 compiler attempts. Nonzero weights sampled independently, no injected
shared rows, cancellations, low rank, or post-selection. Cap: 100000 constructed
nonterminal nodes and 300000 cached apply entries per attempt; refusal is not a
fast success and no uncharged fallback is permitted. Keep cap outcomes.
For successful builds: n=2 all 256 signed input vectors; otherwise 64 fixed random
vectors plus all min/all max/all zero/alternating extremes. Fixed seeds independent
of constructor. Outputs checked against integer/Fraction rounding and sequential
NumPy FP32 then BF16 reference. Preserve all inputs and raw outputs.
Measure program bytes, peak constructor node/cache counts, and query visited-node,
root/input-map/memo counts. Count unique 64-byte and 4096-byte program address
blocks as *logical placement diagnostics*, not hardware measurements. Do not add
HBM/CPU/SSD tiers together into a latency claim.

## Decision rule / obligations
A successful restricted source must be correct; any mismatch fails closed. Even
if correct, >=10x gate requires *all* registered completed cases' model-code read
ratio <=0.10, no caps, and target/state/construction bounds still needed. If the
first-stage gate fails, stop scaling or GPU work and do not tune widths/orders.
O1-O5 remain OPEN for the full mission. O6: this local source, evidence and remote
handoff may be supplied, but full-theory O6 is not closed by a test suite.
