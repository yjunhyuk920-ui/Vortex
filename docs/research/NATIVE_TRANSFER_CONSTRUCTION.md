# Native ordered-transfer construction — NTC-2026-09-05

Base: `947b307da98b75f7843babbed691c70bd3caa964`, the verified CTC-governed
PR #119 lineage. This is constructive discovery and scoped E1 evidence, **not
an admitted high-speed core and not completion of the VORTEX mission**.

## 1. Final theorem, selection and actual scope

The intended final theorem is unchanged: a uniform finite constructor/executor
for every admitted public unmodified HF dense 405B checkpoint and every legal
prompt/context/RNG state, preserving its original output and required successor
state, with one GPU total peak <=8 GiB and same-machine native-4B-Q4 warm
p50 <=1.2x / p95 <=1.5x, plus the unchanged TTFT requirement. All construction,
CPU/RAM/SSD/GPU/KV/workspace/transfer/verification/repair costs count.

The model-native ABI, baseline and evaluation population are not frozen here.
Consequently this study cannot establish O5 or a full mission theorem even if
its numerical lemmas are sound. The primitive ABI is **binary32 serial fmaf,
round-to-nearest-ties-even, gradual underflow, finite input words**. A different
Torch/cuBLAS/TensorCore reduction order is not silently replaced by this ABI.

The earlier broad alternatives were revisited: (i) a globally encoded cold
query source still lacks a finite native decoder within budget; (ii) exact
ordered-state composition had a missing numerical constructor; (iii) deferring
required successor state does not remove its later paid work. No one of these
has a proven >=10x fully charged route. The present bounded construction tackles
(ii), not an assertion that it passed core admission. Within it, the frozen
three mechanisms are different execution orders: forward guarded phase
composition, suffix-first synchronization, and backward inverse-cell execution.
They belong to one numerical family, not three demonstrated universal cores.
No novelty/priority claim is made for established IEEE or interval principles.

| Mechanism | Concrete route considered | Decisive cost issue |
|---|---|---|
| A: compose native transitions | reuse a length-k descriptor at constant descriptor-evaluation work | construction reads all query-dependent products; useful reuse unproved |
| B: suffix-first synchronization | certify a suffix of k terms; erase the unknown prefix if both native endpoints coalesce | ordinary synthetic rows did not coalesce; all retries paid |
| C: backward rounding-cell inverse | compute exact incoming-state intervals without a 2^32-state table | per-query products/inverse construction and early-certificate coverage still needed |

A and C provide finite numerical subroutines. B failed the preregistered ordinary
synthetic screening population; its positive control establishes soundness,
not population coverage. After that rejection, C was continued with a separately
recorded preregistration and an exact direct constructor rather than more suffix
width/prompt/threshold sweeps. A complete Transformer backend was not built.

## 2. A constructive two-phase summary that preserves ordered rounding

Let `R` be nearest-even integer rounding. It obeys

```text
R(z + 2j) = R(z) + 2j.
```

Odd translation is not generally valid at a tie. For a fixed normal binade
exponent e, set `Delta = 2^(e-23)`. Write a signed accumulator as `a = k*Delta`.
For the **exact**, not separately rounded, product `p = w*x`, set `t=p/Delta`.
For phase `r in {0,1}`, define

```text
d_t(r) = R(r+t) - r.
```

Since `k-r` is even, the fixed-lattice step is exactly
`k -> k + d_t(k mod 2)`. A sequence is represented by six signed integers:
final offsets D[0],D[1], and the minimum/maximum prefix offsets L[0],L[1],
H[0],H[1], including the empty prefix. A leaf has D=d, L=min(0,d), H=max(0,d).
For sequence S followed by T, define for each starting phase r:

```text
q = (r + D_S[r]) mod 2
D_ST[r] = D_S[r] + D_T[q]
L_ST[r] = min(L_S[r], D_S[r] + L_T[q])
H_ST[r] = max(H_S[r], D_S[r] + H_T[q])
```

**Proof of composition and associativity.** S first shifts the integer by D_S
and changes its parity to q. T then contributes its q-dependent shift.
The prefix offsets of ST are exactly those of S together with D_S plus the
prefix offsets of T. The displayed extrema and final shift therefore describe
the concatenated ordered sequence. Concatenation is associative, and this exact
semantic description shows the two parenthesizations have identical summaries.
It does not reassociate floating-point addition.

**Native guard.** For a positive initial accumulator in binade e, require

```text
2^23 + 1 <= k + L[r] <= k + H[r] <= 2^24 - 1.
```

For a negative one use the symmetric interval
`[-(2^24-1), -(2^23+1)]`. Zero, subnormal, nonfinite, wrong-exponent and boundary
initial accumulators are rejected by this primitive. Every rounded prefix is
then strictly inside the same binade, whose adjacent floats form the uniform
Delta lattice. On each step nearest-even lattice rounding equals native fmaf
rounding, including tie parity. Induction establishes exact native words at
all prefixes, hence at the end. A failed guard returns `None`, not a fabricated
answer; an enclosing executable must charge its original exact path.

The guard must cover **every prefix**, not just the final word. With initial
1.5 and products `[1, 2^-23, -1]`, native serial FMA returns word 1069547520,
whereas an unguarded fixed-binade summary returns 1069547521. The actual summary
rejects because an intermediate state crossed the binade boundary.

### Construction and storage are not free

`summarize` is a finite O(n)-term constructor: normalize inputs to binary32,
form all n exact dyadic products, make two phase leaves per term, and compose.
Balanced composition has logarithmic tree depth only with separately charged
parallel workers; total work and operand reads remain linear. The reference is
Python integer/Fraction code, not an optimized GPU implementation.

Two phases do **not** mean two stored bits. A safe signed-field width is
`408 + ceil(log2(n+1))` bits: finite binary32 products have magnitude <2^256,
and division by the smallest allowed Delta can scale by 2^149. Prefix offsets
add at most n such rounded increments. There are six such integer fields plus
length/exponent/header/guard metadata. Small observed values cannot replace this
full-domain bound. Materializing all tree nodes uses O(n) descriptors; streaming
construction can reduce live descriptor storage but not construction reads.
The missing fast checkpoint/query-dependent descriptor source remains open.

## 3. A causal whole-suffix erasure algorithm

This is not a claim that an individual product is zero or absorbed. Let F be
the composition of a selected suffix of the original ordered FMAs. A rigorous
incoming interval `[-R,R]` is obtained without reading the prefix at query time.
Because fixed-operand native FMA is monotone in its accumulator, F is monotone.
If F(-R) and F(R) are **the same finite binary32 word**, every possible incoming
prefix state in the interval has that word as its suffix result. Equal real
values with different signed-zero words are not accepted.

### Paid prefix constructor and bound

The static constructor reads all original row weights once. At each block
boundary (default B=32) it stores a power-of-two upper bound S on the prefix
sum of absolute weights. The query scans all activation words to obtain xmax.
With `u=2^-24`, `eta=2^-150`, `r*u<1`, it uses

```text
|a_r| <= (S*xmax + r*eta)/(1-r*u).
```

The implementation rounds this bound outward to a binary power and declines
the certificate if its exponent exceeds 126. An arbitrary prefix uses the
next stored block-boundary bound, which can only enlarge S.

**Proof.** Nearest-even gradual-underflow FMA has absolute rounding error at
most `u*|z|+eta` for a finite exact result z. Induction gives a bound by
`(1+u)^r * sum|w_i*x_i| + eta*((1+u)^r-1)/u`. For `r*u<1`, these factors are
at most `1/(1-r*u)` and `r/(1-r*u)`. Substitute `sum|w_i*x_i| <= S*xmax`.
The conservative finite-radius guard permits the induction without uncharged
overflow assumptions. If it is not available, execute the fully charged native
path. Finite weights/products in an endpoint suffix can overflow to infinity;
that is not accepted as a finite-word certificate.

### Explicit execution and finite work bound

Try suffix lengths B, 2B, 4B, ... below n. For each try, obtain the prefix
radius and execute the actual suffix once for each endpoint. If the two finite
words match, return it; otherwise double the suffix. If no certificate succeeds,
execute the original n-term serial reduction. There is no future output oracle,
no unbounded search, and no silent approximation.

For n a power-of-two multiple of B, success at k uses `4k-2B` endpoint FMAs;
complete failure uses `3n-2B` FMAs. For general n a conservative bound is `<5n`.
Logical distinct weight reads are k on success and n on failure when cached.
Repeated operand reads, the n-element activation scan, metadata probes, cache
storage, and constructor scan are separate paid terms. Logical reads are **not**
measured SSD pages, PCIe transfers or HBM transactions. This reference keeps
original weights in CPU RAM and implements no physical cold pager.

A serial time bound must add constructor amortization over a declared finite
horizon, activation scan, bound arithmetic, actual metadata/weight/x/cache
service, every endpoint FMA, fallback, and synchronization. No sustained
hardware service bounds or same-machine baseline comparison were measured here.
A 32-value static index uses at least `2*ceil(P/32)` bytes across P coefficients,
before row headers and padding. At the illustrative P=405,000,000,000 this is
25,312,500,000 bytes, already above 8 GiB. Cold placement is possible only with
its storage and query movement charged; it is not free resident advice.

### Measured synthetic screen, not a checkpoint population

The frozen 36 ordinary cases used independent BF16-representable operands,
12 rows each at widths 256, 1024 and 16384. **None certified early.** All read
100% of weights; the counted FMA work was 2.75x to 2.99609375x the original.
The 1024-term alternating +/-1 control also failed to certify and paid 3008
FMAs (2.9375x). All final output words still matched the native reference.

The deliberately synchronizing 1024-term control ends in +2^80 then -2^80.
It certified using 32 distinct weight reads and 64 endpoint FMAs, but also paid
1024 activation reads, a 1024-weight constructor scan and 82 metadata bytes.
The 32x weight-fragment/16x FMA-fragment reductions are **not a measured speedup
or a >=10x fully charged core result**. Ordinary cases were not removed from
the population after observation.

## 4. Direct backward rounding-cell construction

The first finite inverse used two binary searches over the ordered finite
binary32 domain. Monotonicity gives the first input whose output rank is >=L
and the first whose output rank is >H. Their interval is exactly the inverse
of [L,H], including separate signed-zero ranks. Each search needs at most 32
native FMAs, including searches that discover an empty inverse.

After the ordinary suffix screen failed, the continuation constructed the
inverse directly. For finite nonzero target endpoints l,h, let

```text
b_low  = midpoint(prev(l), l)
b_high = midpoint(h, next(h)).
```

The lower/upper boundary is included precisely when the endpoint's encoded
significand is even. At the largest finite magnitudes use the virtual overflow
midpoint at distance 2^103. These describe the exact real rounding cell of the
entire output interval; adjacent cells meet without gaps. Subtract the exact
dyadic product p=w*x, then choose the first and last finite representable
accumulators obeying those shifted inequalities. Exact rational rounding plus
a one-rank directed adjustment implements this selection. A reversed/empty
interval is correctly returned as empty, not as an exceptional success.

Output intervals whose endpoint is +0 or -0 use the bounded native search,
so signed-zero semantics are handled explicitly. Nonzero endpoints need no
native FMA search, but products, rational boundary arithmetic, leading-bit and
rounding operations, addresses, and storage remain paid. Dyadic alignment can
require about 560-bit integers; these are not costless unit-precision GPU ops.

**Proof.** For nearest-even rounding, the preimage of a nonzero finite word is
its half-neighbor-distance interval with tie inclusion fixed by parity. The
union for contiguous output words is the interval between the two extreme
boundaries. Native FMA rounds the exact real value a+p once. Translation by -p
therefore gives its real input interval. Taking its exact representable lower
and upper limits yields all and only finite binary32 input words. Zero endpoint
cases are delegated to the independent monotone-search algorithm. Inverse
composition through a suffix in reverse order is then ordinary set preimage
composition, with no enumeration of 2^32 states.

`suffix_preimage` takes a **proposed** output word. Tests use known native outputs
to validate the primitive; these tests do not supply a deployable cheap selector.
A real caller must produce its candidate causally, build the inverse at paid
cost, and prove its incoming range lies inside it. No theorem supplies early
termination/cheap product generation over arbitrary checkpoint queries here.

## 5. Native substitution and the still-missing Transformer lift

For a frozen deterministic original program, replacing a reduction segment by
an algorithm proven to return exactly its original boundary word leaves every
subsequent deterministic instruction unchanged. Starting with equal state and
preserving every observable write proves state equality by induction. Keeping
the same RNG calls/order then preserves the RNG and output token contracts.

This is a **conditional substitution lemma**, not a constructed HF compiler.
It requires the actual reference reduction graph and all rounding/FTZ/conversion
semantics. The present serial ABI has not been matched to an unmodified public
Transformer runtime; no real layer, logit, KV update or token was replaced.
A rational sum or a different parallel reduction does not discharge that gap.

| Obligation | Full-mission status | What this round actually supplies |
|---|---|---|
| O1 | OPEN | finite row/index constructors only; no budget-closing checkpoint constructor |
| O2 | OPEN | terminating native reduction subroutines; no full causal Transformer executor |
| O3 | OPEN | scoped phase/suffix/inverse proofs and conditional state substitution; actual ABI lift absent |
| O4 | OPEN | explicit primitive work/storage/repair bounds; no complete model service schedule |
| O5 | OPEN | no sufficient target memory/latency/baseline closure; logical index already exceeds hot grant if all resident |
| O6 | OPEN | checkable scoped proof and references, not artifacts closing O1-O5 |

```text
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
SCOPED_RESULT=PHASE_SUFFIX_INVERSE_NATIVE_REFERENCE
CORE_ADMISSION=false
```

Neither target-wide feasibility nor universal impossibility follows from these
synthetic results. The unresolved issue is algorithmic, not merely missing GPU
measurement: cheaply generate exact query-dependent transfer information and
necessary successor state with a full target-size upper bound.

## 6. Validation, provenance and reproduction

The initial preregistration hash is
`4d583f5301b989a29e377f43ff21316d3ec9113c0458ede1e780fda23269a1bd`.
The separate inverse-continuation hash is
`1c63b8960c232844456565a7bc9a54beee90f3e95bb08843709b789c67d26af8`.
No original preregistration was edited after results.

Native fmaf and an independent exact-integer oracle agreed on 4008 controls.
600 guarded synthetic sequences agreed with native serial FMA and composition
parenthesizations. The inverse study attempted 1500 random triples; 1293 had
finite output, plus 35 deterministic boundaries =1328 output intervals.
All complete intervals agreed with independent native searches; 1312 were
nonempty direct-cell intervals, seven nonempty zero-boundary intervals, nine
empty. An additional 100 suffix-inverse cases agreed. These are numerical
controls, not generated LLM tokens or formal proof-assistant certificates.

20 focused unittest cases passed under both pytest and unittest. An initial
test incorrectly assumed a midpoint inverse could not be empty; its failure
log is preserved, and the fixture was corrected to compare valid empty sets.
The primitive constructor was also hardened to normalize Python inputs to
binary32 and reject invalid exponent/output-word arguments. All five earlier
scientific JSON/JSONL captures regenerated byte-identically after these changes.
The full repository suite, public checkpoints and target hardware were NOT run.

```bash
python experiments/native_transfer/audit.py
python experiments/native_transfer/audit_inverse.py
python -m pytest -q tests/native_transfer
python -m unittest discover -s tests/native_transfer -v
python experiments/native_transfer/evidence_codec.py --out results/native_transfer/decoded
sha256sum -c results/native_transfer/checksums.sha256
```

The last command checks stored artifacts; encoded large-record captures are
separately verified by the decoder against their original byte hashes. The
large inverse records are losslessly predictor/residual encoded in
`inverse_recording.json`, with code/input hashes and literal residuals. The
predictor uses the independent integer IEEE specification, **not native fmaf**;
therefore decoding is not a new hardware observation. Decoding both captures
reproduced their original 534690 bytes exactly. Literal raw captures are also
included in the downloadable local bundle. Any prediction difference would be
stored literally; the observed residual maps are empty after JSON normalization.
This evidence encoding is unrelated to an LLM weight-compression claim.

## 7. Decision and next decisive construction

Retain the finite native transfer/inverse references as scoped numerical tools.
Do not promote suffix synchronization as a general core, tune more suffix
lengths after seeing these results, claim phase evaluation excludes its build,
or infer useful Transformer reuse from the constructed positive control.
F-065's local-reuse failure remains intact; the different whole-suffix theorem
does not invalidate that measured failure or establish full-model coverage.

Continuation requires a concrete checkpoint-derived/query-causal mechanism
that avoids generating essentially all dense products, with bounded native
conversion and state-update work, not an undefined compressed-transducer oracle.
Before another native-family core experiment, its fully charged >=10x route
must survive the existing failures. Before model substitution, pin and match
an actual original numerical ABI. Before theory acceptance, close all O1-O6,
including a valid baseline lower bound or coupled latency ratio. Another toy
width, inverse timing number, or commit is not mission progress by itself.

Prior root snapshots are preserved as their exact Git blobs under
`docs/research/history/pre_native_transfer_20260905/`. No existing executable
module is modified, no Actions job is dispatched, no main merge is requested,
and separate EXP-103A..108A branches are untouched. Post-commit remote SHA and
file-hash verification belongs in the actual tool receipt/PR, not a self-hash.

## Source and semantics reference

NVIDIA, *Floating Point and IEEE 754 Compliance for NVIDIA GPUs*, sections on
rounding, fused multiply-add, and operation order:
https://docs.nvidia.com/cuda/floating-point/index.html (consulted 2026-09-05).
The scoped constructions/proofs above are derived here; the source is not a
claim that NVIDIA established VORTEX's mission or these performance results.
