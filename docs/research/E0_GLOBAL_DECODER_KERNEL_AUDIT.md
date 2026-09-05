# E0 global-decoder kernel audit and research handoff

Date: 2026-09-05 (Asia/Seoul). Base: `ff70c1ebbca943161684f33e6afa1dc169fa8f4c`,
`research/e0-route-proof-reach-validation`, PR #118.

**Decision: `SCOPED_2X2_MINIMUM_PROVED_NO_EXECUTOR_PROMOTED`.**
This is an E0/E1 prerequisite, not a new numbered core experiment. It does not
achieve, measure, or prove impossible the fixed 405B / single 8 GiB / same-machine
native-4B-Q4 p50 <=1.2x and p95 <=1.5x target. No checkpoint was downloaded or run.

## 1. Research selection and three candidate principles

The starting point is the actual PR #118 evidence, not the old main-branch
summary. Its 3,474,432 encoder/tag cases rejected only the registered affine
Feistel grammar. Its seven-bit non-systematic problem is still open. Earlier
source-free proofs and identity prefix-state replay are not cheap executors.

Three formulations were compared before further implementation:

| Principle | Premise reversed and exact equation | Conditional >=10x route and complete cost | Cheapest decisive question / disposition |
|---|---|---|---|
| Global response encoding | Store globally encoded responses rather than independently readable coefficients: `D(E(W),q)=q(W)`. Addresses may depend on previous returned data. | A genuinely short decoder could replace an `mn`-term query with `t` word probes plus decode. Charge encoder construction, `S*w+H` storage, physical pages, addressing, decode, activation/state updates, verification and fallback; nothing is free. | Does a concrete code and exact decoder exist at two scales? Implement only the bounded prerequisite and scope checker below. No 10x real-model route has yet been established. |
| Native accumulator-action composition | Replace a sum representation with the ordered action `tau_k o ... o tau_1`, where `tau_i(s)=RN32(s+w_i*x_i)`. For the chosen fused-add reference, function composition preserves order even when addition is not associative; other ABIs need their literal operation tree. | A compact, cheaply constructed action could bypass many internal operations. Charge activation-dependent action construction, guards, representation, composition/evaluation, cache misses and native state. | No subdense constructor was supplied. Merely naming the composition leaves all native operations inside it. Do not implement another page-no-op or exact-input macro sweep. |
| Deferred exact successor-state evaluation | Store a causal executable state description and materialize only future-demanded components; require a simulation relation preserving every legal continuation. | Savings require provably small demand closure, including all pending evaluations, cache growth, repair and fallback. `total = constructor + demands + state + verification + misses`. | No cheap demand-closure evaluator or bounded physical state was supplied. Without it this is the already rejected dense replay / source-free state interface. Not promoted and not implemented. |

These are candidate formulations, not claims of publication novelty or three
successful inventions. The second and third do not earn implementation merely
because their interfaces are mathematically well-defined. Only a bounded
necessary-condition investigation of the first was executed. The result should
not become an excuse for another sequence of tiny negative tests.

## 2. New finite theorem: the 2x2 optimum is five stored bits

**Scope.** Every binary 2x2 matrix W is allowed. Preprocessing is any lossless,
possibly nonlinear map into S one-bit cells. There is no extra checkpoint-derived
advice. The query program is fixed independently of W; any W-dependent program
or embedded constants must count as stored information. For every nonzero u,v in F2^2, a deterministic zero-error query must return
`u^T W v mod 2` using at most two adaptive bit reads, with arbitrary free logic
between reads. This is an information-query model, NOT a CPU-time model.

**Theorem.** S >= 5. Five bits suffice. The lower bound permits arbitrary
nonlinear encoders, arbitrary adaptive address choices, and arbitrary decoder
logic; it is not an enumeration of affine permutations.

### Proof of the four-bit obstruction

The four matrix-unit queries recover W, so any four-bit encoding E is a bijection
of the 16 source matrices onto the complete four-dimensional Boolean cube.
For a binary mask Q define the real-valued character

`chi_Q(W) = (-1)^<Q,W>`.

Let U be the span of the nine rank-one characters. They are orthonormal under the
uniform source measure and have zero mean. Let V be the pullback through E of
all nonconstant Fourier characters of encoded Hamming degree one or two. V has
dimension `C(4,1)+C(4,2)=10`, is orthogonal to the constant function, and has an
orthonormal basis because E is a bijection.

A depth-two bit decision tree has a real multilinear polynomial of degree at
most two. Each required answer has zero mean, so its constant Fourier coefficient
on the complete encoded cube is zero. Therefore successful decoding implies
`U subset V`.

The one-dimensional orthogonal complement `V minus U` has an orthonormal basis
function z. Using kernels without the usual 1/16 normalization,

`K_V(W,W') - K_U(W,W') = z(W) z(W')`.

The diagonal is `10-9=1`, so every `z(W)` is either +1 or -1. For distinct source
matrices, with D=W+W', the rank-one character kernel is

`K_U(W,W') = 2^(4-rank(D)) - 7`,

which is 1 or -3. For two distinct encoded words at Hamming distance h, write
`s=4-2h`. The degree-one/two kernel is

`K_V(W,W') = s + (s^2-4)/2`,

which is 2 or -2. Thus the possible differences are `{-3,1,5}`. Since z takes
unit-magnitude values, its pairwise products must be in `{-1,+1}`; the only
compatible value is +1. Hence z is constant across all 16 sources. But z belongs
to V, which is orthogonal to constants. Contradiction.

The kernel formula follows directly by summing `(-1)^(u^T D v)` over nonzero
u,v: including zeros gives `2^(m+n-rank(D))`; subtract the two zero-vector
families and add back their shared zero. The code checks both this sum and an
independent row-span rank calculation using integers only.

### Five-bit construction

Store `(a,b,c,d,a xor b xor c xor d)` for `W=[[a,b],[c,d]]`.
The nine nonzero rank-one queries consist of four individual cells, four
row/column pair parities, and the full four-cell parity. They require one, two,
and one reads respectively. Thus the lower bound is tight.

This construction costs 5/4 of raw logical bit storage. Both tiny layouts fit
inside one physical byte. It establishes **no physical speedup**.

### Why this is not a 2x3 or 405B impossibility result

The proof needs a bijection onto the COMPLETE encoded cube and a
ONE-dimensional complement. Extra advice, redundant cells, word cells,
restricted checkpoint populations, and other query contracts invalidate its
hypotheses. The scope checker returns `NO_CONCLUSION_FROM_THIS_THEOREM` for the
2x3/seven-bit and 25x108/50-word cases and for native Transformer claims.

The 2x3/six-bit case is already rejected by the prior rank-amplified degree Gate:
`64 > sum(j=0..4) C(6,j)=57`. Its kernel checks are integrity controls, not a new
result. The new item here is the exact nonlinear 2x2 optimum, which the plain
degree-capacity count alone does not reject.

## 3. General seven-bit synthesis: inconclusive, not a rejection

A bounded QF_BV formula represents each of seven cells as an arbitrary 64-bit
truth table over ALL 2x3 source matrices. Each of the 21 queries chooses a first
cell, a separate second cell on each first-answer branch, and Boolean leaf
values. Packed bitwise evaluation is constrained to the exact query truth table.
Thus it is not an affine-Feistel parameter sweep.

Symmetry normalizations are explicit: complement cells to encode source zero as
zero; rename the first query's root and children; sort the remaining interchangeable
cell truth tables. A second read of the already known root can be replaced by an
unused choice with constant leaves. Unused or duplicate/constant cells can be
filled with fresh distinct nonconstant functions after eliminating their use.
The zero-source leaf is fixed to zero. No target labels or answers are supplied
at runtime; this is offline finite synthesis, not a trained model.

Independent control: all 351,232 seven-cell word/tree/leaf combinations agree
between the packed formula and direct branching. A fixed five-bit 2x2 positive
encoder is SAT under the same grammar.

Original bounded run:

```text
libz3                    4.13.3.0
timeout limit            30,000 ms
seed                     0
result                   unknown
reason                   timeout
scientific decision      INCONCLUSIVE
formula SHA-256          5e4e91eb824519b2569b6b9249b245369db21194c5dbf5add679de6f0033c4be
```

The original elapsed wall time was not recorded and is not invented. The formula
regenerates byte-for-byte from `synthesis.py`. Search output can vary by machine;
UNKNOWN is never a proof of nonexistence. No longer-timeout rerun is authorized
by this result. A SAT result would require extracting and independently checking
all encoded states and decoder trees before any positive scientific decision.

## 4. Native semantics and successor-state controls

There are 1,488 exhaustive small binary matrix/query controls in which an
ordered FP32 accumulation equals the corresponding small integer sum. This
only verifies the easy one-way subcase. It does NOT lift a parity decoder:
zero and `[[1,0],[0,1]]` both give parity zero for u=v=(1,1), but their native
sums are zero and two.

A separate native-order counterexample uses BF16-representable contributions
`[2^24, 1, -2^24]`. Left-associated FP32 accumulation gives zero; right association
gives one. A proposed finite-field or function-composition executor must preserve
the actual reference reduction ABI rather than silently substitute an exact real
sum. No target CUDA behavior is inferred from this CPU reference control.

The five-bit positive code is also used for 8,192 transitions of the nontrivial
toy state machine `x_next = W x mod 2` over all 16 matrices and all four initial
states, 128 transitions each. All state comparisons pass. These are **GF(2) toy
state transitions, not LLM tokens**, and provide no native KV-state construction.
Five single-cell corruption controls all change a legal answer.

## 5. Fully charged resource boundary

For any proposed executor use resource-specific measured quantities:

`T >= max(SSD_bytes/B_SSD, PCIe_bytes/B_PCIe, HBM_bytes/B_HBM,
          instructions_i/F_i, dependency_time)`

with construction, metadata, decode, state, verification and fallback included
on the actual path. The original resource goal is not changed.

The inherited favorable 25x108 diagnostic assigns 50 64-bit stored words (400
bytes) and two logical word reads (16 bytes). Against its favorable Q4-equivalent
1,350-byte denominator, `16/1350=8/675`: it already consumes the entire nominal
allowance before any extra bytes. One *additional charged* byte would produce
`17/1350`, a factor `17/16` above that diagnostic line. This is conditional; the
audit does NOT prove an extra physical byte mandatory. Cached metadata, page
coalescing, native lifting, address work and state must be measured or proved,
not filled in with zeros. The parity/Q4 comparison remains diagnostic only.

## 6. Validation and reproduction

```bash
python experiments/e0_global_decoder/audit.py --output results/e0_global_decoder/summary.json
python -m pytest -q tests/e0_global_decoder/test_audit.py
python -m unittest discover -s tests/e0_global_decoder -v
python experiments/e0_global_decoder/synthesis.py
sha256sum -c results/e0_global_decoder/checksums.sha256
```

The audit and test controls use the Python standard library; pytest is only a
test runner. Optional `synthesis.py --solve` needs the recorded local libz3.
Do not run a search just to reproduce the deterministic proof result.

Local validation: 18 pytest tests passed and 18 unittest tests passed. Summary
regeneration is byte-identical. There are 4,352 ordered-pair kernel checks,
1,063 orthogonality-pair checks, 144 positive queries, five corruption witnesses,
8,192 toy state transitions, and 1,488 binary/native subcase checks. Full-repository
regression was NOT run because no complete local checkout was available. No
existing executable module is changed. GitHub Actions was neither required nor
dispatched; local validation precedes remote persistence.

Deterministic core SHA-256:
`e21f7c4e3aad75727d31b6cb318feabf9fc572e7a4d92106dca4b538522ef7eb`.

## 7. Decision, assumptions, and next handoff

Retain the new kernel obstruction and exact decoder-grammar checks as auxiliary
prerequisites. Do not promote an executor, assign a new core experiment number,
claim a speedup, or infer a general nonlinear cell-probe lower bound.

The active constructive requirement remains a redundant, non-entrywise GLOBAL
code with an explicit encoder, causal addresses, decoder and nontrivial native
successor-state path. The same grammar must work at two sizes WITHOUT separate
truth tables and must close the physical storage/query/compute ledger before
public-checkpoint layer replacement. Merely solving the seven-bit seed would not
satisfy this requirement. Do not continue with more tiny zero-redundancy theorems,
Feistel parameter sweeps, longer SMT timeouts, or a proof-checker-only runtime.
A new core round must instead provide the missing constructor or a materially
different, fully charged causal execution source.

Unchanged final status:

```text
actual 405B execution                      NOT TESTED / NOT ACHIEVED
physical complete allocation <=8 GiB       NOT TESTED
same-machine native-4B-Q4 p50/p95           NOT TESTED
actual public Transformer layer replacement NOT PERFORMED THIS ROUND
seven-bit 2x3 encoder                      UNRESOLVED
native numerical/state lift                NOT CONSTRUCTED
general impossibility                      NOT ESTABLISHED
```

This round narrows a finite prerequisite and improves verification. It does not
supply positive evidence that the E7 performance goal has become more likely.

## 8. Source map and history preservation

Read alongside `AGENTS.md`, `MISSION_AND_WORKING_PRINCIPLES.md`, the local-first
commit mandate, and these base-commit sources:

- `docs/research/E0_ADAPTIVE_NONLINEAR_PROBE_DEGREE_GATE.md`
- `docs/research/E0_ROUTE_PROOF_REACH_VALIDATION.md`
- `results/e0_route_proof_reach_validation/summary.json`
- `FAILED_APPROACHES.md`, `FAILED_APPROACHES_RECENT.md`
- `docs/PROOF_FIRST_CONTRACT.md`, `docs/RESEARCH_EFFICIENCY_CONTRACT.md`

The refreshed README/state/next-gate/validation snapshots retain exact copies
of their old blobs under `docs/research/history/pre_global_decoder_20260905/`.
Those copies are historical, not new instructions; relative links inside them
are interpreted against the original repository root at the pinned base commit.
Append-only decision/failure ledgers and all prior experimental evidence remain
unchanged. This document is the scoped decision and validation addendum for the
new auxiliary result; it does not reverse an existing core decision.
