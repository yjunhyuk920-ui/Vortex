# E0 Causal Bilinear Query Restriction

## Status and evidence boundary

- Date: 2026-08-09 Asia/Seoul
- Status: SHAPE THRESHOLD CLOSED; CHEAPEST E1 NECESSARY GATE PREREGISTERED;
  NO MODEL ROW EXECUTED
- Evidence: finite resource accounting, a rank-to-miss theorem, deterministic
  calculator, and ten independent/exhaustive controls
- Evidence ceiling: E0; a future run can be only an E1 favorable structural
  observation
- Model/checkpoint/hardware action: none

This note asks whether actual causal Transformer decision queries are much
smaller than the arbitrary Cartesian rank-one query set used by the preceding
lower bounds. It identifies a precise statistic that could matter and freezes
the cheapest real-checkpoint rejection Gate. It does not claim that the
statistic is favorable, that a current runtime can expose it cheaply, or that a
Core Candidate survives.

## What is new and what is not reopened

For matrix instance `i` and causal token state `t`, the post-Atlas exact
decomposition leaves

```text
b_i,t = r_i,t^T W_i u_i,t
q_i,t = vec(r_i,t u_i,t^T).
```

EXP-069 tested whether later **activation vectors** lie in the exact span of
earlier activation vectors. The object here is instead the outer-product
coefficient query `q=r tensor u` after both primal and decision-dual side
components have been removed. A new `r` can in principle compensate for a new
`u`, so EXP-069 does not logically determine its span.

The screened execution interface is named a **Causal Bilinear Span Ledger**:

```text
calibration residual pairs and exact scalar answers
    -> factorized basis q_j = r_j tensor u_j
current causal pair (r,u)
    -> exact coefficients q = sum_j c_j q_j
    -> certified sum_j c_j b_j
non-member / failed witness / corrupt state
    -> unchanged matrix fallback or abort
```

This is not response replay: a hit may be a new pair and a new outer product,
provided an exact linear-combination witness exists. It is also not a promoted
runtime. A finite-field rank nonincrease is never called a hit, and a dense
reference trace is never called a free pair source.

## The rank-to-miss theorem

Let `S={q_1,...,q_N}` be an evaluation population with certified linear rank
`R` over the exact dyadic/rational query values. Grant a future-aware oracle
the best `B`-dimensional subspace after seeing all of `S`. If only `m` rows lie
outside that subspace, then

```text
rank(S) <= B + m.
```

Therefore every `B`-dimensional ledger has

```text
m >= max(0, R-B).                                    (1)
```

Equation (1) is stronger than evaluating one unlucky build basis: it grants
the best post-hoc subspace and is used only to reject. Exhaustive GF(2)
controls enumerate every two-generator subspace in four ambient bits and every
four-vector population; none violates (1). Separate rank-one controls show
that a small outer-product population can attain full observed rank and that
duplicates/zero factors receive no false rank credit.

## Registered 405B shape threshold

The frozen non-embedding population has:

```text
matrix instances K                         883
dense coefficients D                       403,747,897,344
packed Q4 bytes                             201,873,948,672
sum_i (m_i+n_i)                             39,109,888 coordinates
p50 allowance alpha                        8/675 = 1.185185185%
```

The favorable ledger stores each basis pair coordinate in two bytes. Exact
residual values can require more, so this is a grant. For span dimension `B`,
the every-token basis scan is

```text
factor bytes       = 2 B sum_i(m_i+n_i)
factor operations  = B sum_i(m_i+n_i).
```

The registered metadata includes, per matrix, a four-byte `B x B` Gram
inverse, four-byte cached scalar answers, eight-byte pivot descriptors, and
six two-byte basis fingerprints. The retained six-check verifier contributes

```text
operations                            234,659,328
logical traffic                       538,678,272 bytes
resident state                        458,686,464 bytes
```

Building one aligned basis tuple is granted one dense forward equivalent and
one dense reverse equivalent. The resulting `2B` dense-equivalent build is
amortized over an explicit `20,000,000` checkpoint-service tokens.

The complete favorable pre-fallback equations are:

```text
C_common(B) = [B S + K B^2 + K B + 6 K B + C_verify] / D
              + 2B / 20,000,000

T_common(B) = [2 B S + metadata(B) + T_verify] / Q4_bytes
              + 2B / 20,000,000.
```

Pair extraction, a local trace/proposal source, native rounding repair, KV,
workspace, fallback overlap, and allocator headroom are still positive in a
real executor. They are set to zero only for this necessary screen.

Exact enumeration gives:

```text
largest span dimension fitting all component Gates       B = 23
B=23 common logical traffic                              1.159413233%
B=23 common logical operations                           0.281295494%
B=23 component state                                     2.104879502 GiB
B=24 common traffic                                      above 1.185185185%
```

Thus a full factor scan cannot be rescued by increasing its span after seeing
the result: `B=24` fails even with perfect hits and every omitted system cost
free.

At `B=23`, the factor payload alone is `1,799,054,848` bytes/token. Favorable
contiguous layout still exposes:

```text
left/right vector segments                 40,618
64-byte lines                          28,110,232
4-KiB vector-rounded pages                442,129
```

These are logical address counts, not SSD, PCIe, HBM, or latency measurements.

## Weighted fallback threshold for the cheapest real Gate

The E1 screen observes only the pinned small checkpoint's last
`down_proj=[1024,3584]`. A miss is nevertheless charged as one registered
405B `down_proj=[16,384,53,248]` instance:

```text
registered coefficients per down instance      872,415,232
fraction of D                                   208/96,261
evaluation rows                                 36
traffic headroom after common cost              0.025771953%
maximum favorable down fallbacks                4 / 36
```

Combining this with (1), a certified evaluation rank of at least

```text
B + allowed_misses + 1 = 23 + 4 + 1 = 28       (2)
```

rejects every `B<=23` factor-scan ledger, even when the best subspace is
chosen after evaluation and all other 882 matrix instances hit. The strongest
counterexample is generic independent arrival: rank `36` forces at least
`13/36` last-down fallbacks and raises favorable total traffic to
`1.237441833%`, or `1.044091547x` the complete p50 allowance.

## Exact leakage-free residual-pair extractor for the Gate

The future evaluator must use the unchanged pinned checkpoint:

```text
Qwen/Qwen3.5-0.8B
revision 2fc06364715b967f1860aea9cf38778875588b17
weight SHA-256 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
prompt SHA-256 46c9779c24e59f356247145f7ef870a1b82ba4907eb9dffcd48652e519f7612a
```

For each prompt and causal position:

1. Run the unchanged dense target on the exact committed prefix and capture
   the last layer's `down_proj` input `x`, output `y`, residual branch before
   the addition, final pre-RMSNorm state, and logits.
2. Choose the native greedy winner `w` and strongest different competitor `c`;
   ties use the lowest token ID.
3. Detach `y`, replay only `residual_add -> final RMSNorm -> tied LM head`, and
   compute

   ```text
   v = derivative(logit_w-logit_c) / derivative(y).
   ```

   This suffix is exact for the captured native branch and does not traverse a
   target-future token. It is nevertheless a same-token dense-reference oracle
   because `w,c,y` are known only after the native result.
4. On prompt positions only, build rank-at-most-16 primal and dual bases
   `Q,P` by the frozen two-pass MGS order. The current decode position is
   excluded. A rank-deficient, non-finite, or mismatched prefix fails closed.
5. In the frozen float32 operation order form

   ```text
   u = x - Q(Q^T x)
   r = v - P(P^T v).
   ```

   The exact IEEE bit patterns of `r,u`, not a rounded tolerance, define the
   query tensor for certification.

Prompt-token batched capture is permitted only if every registered prefix-only
replay control matches the corresponding batched activation, gradient, winner,
and competitor exactly. Otherwise the run is an infrastructure/control failure,
not scientific evidence.

## Frozen build/evaluation population

Reuse the already pinned EXP-076 prompt split; do not select new prompts from
the result.

```text
build              6 prompts, one per family
build positions    first 4 genuine post-prefill decode positions
build query rows   24
ledger rule        first 23 exact-rational independent rows in file/position order

evaluation         18 disjoint prompts, three per family
evaluation rows    first 2 genuine post-prefill positions
total rows         36; six per family
```

The build code and all pivot choices are immutable before evaluation. The
separate oracle-rank rejection in (2) may inspect the evaluation population
only because it grants the best post-hoc subspace and can only make rejection
harder.

## Finite-field and exact-hit contract

Each float32 factor is decoded as its exact IEEE dyadic rational. Outer-product
coordinates are reduced into the same three odd fields used by EXP-069:

```text
65,521
65,519
65,497.
```

A rank increase under any registered prime certifies rational independence;
an integer minor nonzero modulo that prime is nonzero over the rationals. A
modular rank nonincrease is **not** an exact hit.

For a hit, coefficients are solved from frozen pivot coordinates in exact
dyadic/rational arithmetic. The identity

```text
r u^T = sum_j c_j r_j u_j^T
```

must then be verified exactly by a factorized rank test, or the row is a miss.
Six independent rank-one field fingerprints protect the implementation path.
For a nonzero residual matrix over `F_p`, one check collides with probability
at most `2/p-1/p^2`; over `883 * 20,000,000` service queries the registered
six-check union bound is below `1.43e-17`. This probabilistic check does not
replace the exact rational witness.

## Preregistered E1 decision

Control requirements:

```text
zero checkpoint/config/prompt/hash mismatch
zero current-position inclusion in Q or P
zero later-token read by extraction or basis construction
zero prefix-replay mismatch
zero non-finite/malformed pair
zero false exact hit or fingerprint failure
```

Scientific rejection occurs immediately when either condition becomes
inevitable:

```text
certified held-out query rank >=28
or exact build-ledger misses >4 of 36.
```

The canonical run may stop when rank 28 is certified. Complete promotion
requires all 36 rows, at most four exact misses globally, at least five exact
hits in each six-row family, oracle rank at most 27, and every exact-hit
witness/verification control passing.

Decision on pass:

```text
PROMOTE_ONLY_TO_PAID_PAIR_EXTRACTOR_AND_NATIVE_NUMERICAL_SEMANTICS_GATE
```

Decision on scientific failure:

```text
REJECT_CAUSAL_BILINEAR_FACTOR_SPAN_LEDGER_AS_CORE
```

No experiment number is assigned by this preregistration. The next map ticket
may execute only this frozen last-down Gate. It may not change span dimension,
side rank, primes, prompt split, positions, decision direction, or exact-hit
definition after evaluation.

## Why a pass would still not be a Core Candidate

The Gate deliberately isolates population structure from result production.
For a general internal matrix, obtaining the current decision residual `r`
requires an exact pullback through a nonlinear suffix. The known raw-checkpoint
VJP route is a dense traversal and consumes the work being avoided. At the last
down slice the derivative is cheap only because the remaining suffix is
`residual add -> RMSNorm -> LM head`; that slice alone is not the model-wide
executor.

Likewise, a coefficient identity over exact dyadic query tensors does not by
itself prove that a linear combination of native rounded scalar answers
reproduces the unchanged BF16/Q4 reduction order. A pass would therefore open
only these two missing constructions:

```text
paid causal pair extraction without a dense trace proposer
native exact/outward-corrected coefficient and answer semantics
```

Until both close with selector, verification, miss, fallback, KV, workspace,
traffic, and state, the project remains `NO_SURVIVING_CANDIDATE`.

## E0 scorecard

1. Target-scale upside: an exact span hit replaces a dense bilinear query; the
   abstract effect is large enough.
2. Novelty: the query is `r tensor u`, not an activation replay or another
   Atlas rank/page choice.
3. Evidence basis: only the exact rank-to-miss theorem and shape equation
   exist; real query rank is unmeasured.
4. Scaling: factor scan cost grows with `sum(m+n)` while dense work grows with
   `sum(mn)`, but required exact coverage and pair extraction remain open.
5. Universality: build order is automatic and checkpoint/prompt split is
   fixed; arbitrary held-out prompts still require fail-closed misses.
6. Correctness: exact rational witness plus independent fingerprints or
   fallback; modular nonincrease is not accepted.
7. Resource closure: the favorable component closes only through `B=23` and
   leaves a four-row down-fallback allowance; mandatory runtime state and the
   pair/result source are explicitly absent.
8. Cheapest falsification: rank 28 among 36 last-down queries rejects the
   complete factor-scan class before another model-wide extractor or backend.

## Decision and claim boundary

```text
DERIVE_B23_SHAPE_LIMIT_FOR_A_FULL_FACTOR_SCAN
PREREGISTER_LAST_DOWN_CAUSAL_BILINEAR_RANK_GATE_ONLY
DO_NOT_CALL_MODULAR_NONINCREASE_AN_EXACT_HIT
DO_NOT_REOPEN_ACTIVATION_REPLAY_OR_ATLAS_SWEEPS
KEEP_PAIR_EXTRACTION_AND_NATIVE_NUMERICAL_RECONSTRUCTION_UNSOLVED
KEEP_NO_SURVIVING_CANDIDATE
```

The result neither proves nor disproves that real causal pairs have low span.
It also does not rule out a nonlinear or more compact query code. It proves
which finite statistic the cheapest run must cross and why a rank sweep cannot
rescue the declared full-factor ledger after failure.

## Reproduction

No model is used:

```powershell
$env:PYTHONPATH = ".;.deps"
python scripts/derive_causal_bilinear_query_restriction.py `
  --output-dir results/e0_causal_bilinear_query_restriction
python -m pytest -q tests/test_causal_bilinear_query_restriction.py
```

Observed validation: `10` focused tests, all `486` repository tests, and the
standard `scripts/run_validation.py` run passed. The standard runner uses only
its existing synthetic fixtures and is not causal-query population evidence.
Authority:

```text
results/e0_causal_bilinear_query_restriction/summary.json
results/e0_causal_bilinear_query_restriction/checksums.sha256
summary SHA-256 bb8456553848d10a3b4128cedcb24f9f0d444d1c9d880f0cf9bc7a2d4996309c
```
