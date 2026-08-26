# E0 validation — ROUTECELL, PROOFWAVE, and REACHQUOTIENT

Date: 2026-08-26 Asia/Seoul  
Validated base: `f5a432a5152ee6d9e1399583a4829a39aacdf5e0`  
Deterministic core: `ab149d78b9e04e504a1c46e1a40cd928e9d9d9c518ad14197cfe9637e7861587`

## Question

Validate three newly proposed execution principles before assigning another
numbered experiment:

1. **ROUTECELL** — lossless checkpoint cells whose returned content selects the
   next cold address;
2. **PROOFWAVE** — a local proof-producing executor that commits a token only
   after proving every better alternative impossible;
3. **REACHQUOTIENT** — an exact behavioral quotient that stores a compact state
   class instead of native hidden/KV bytes.

The fixed p50 target-equivalent fraction remains:

```text
8 / 675 = 1.185185185185...%
```

The authoritative result is an E0/E1 cheapest-kill audit. No checkpoint forward,
checkpoint download, CUDA/backend action, target-host action, or physical
latency measurement occurred. Synthetic controls are falsification tools only;
they are not Transformer success evidence.

## Source boundaries reused by this audit

The audit does not reopen already closed families.

- `docs/research/E0_ADAPTIVE_NONLINEAR_PROBE_DEGREE_GATE.md` leaves a fully
  non-systematic nonlinear adaptive cold encoding formally open. Its smallest
  unresolved seed is a 2-by-3 binary source encoded in seven bits and queried
  with two adaptive reads; the target-scale capacity survivor is the
  25-by-108/50-word/two-read interface. Neither is a constructor.
- `docs/research/E0_NONLINEAR_RECURSIVE_LIFT_GATE.md` proves that a nonlinear
  local seed cannot be composed entrywise for free. A useful lift must expose
  and charge cross-child correlations or use a non-entrywise global encoding.
- `FAILED_APPROACHES_RECENT.md` F-048 closes proof-carrying execution when the
  local trace generator remains dense or is omitted from accounting. It leaves
  a genuinely cheap causal local generator open.
- `docs/research/EXP_089A_LATEST_RESULT.md` accepts exact prefix-plus-RNG replay
  as a state witness. It explicitly does not establish a fast compressed
  quotient.

---

# 1. ROUTECELL

## Registered grammar

The proposed general idea is broader than can be exhaustively searched. This
Gate therefore freezes one concrete uniform grammar before evaluating it.

Source and query:

```text
source                       every 2 x 3 binary matrix
source bits                  6
queries                      all 21 nonzero GF(2) rank-one parities
```

Encoder:

```text
3+3 Feistel network
same affine 3-bit round function repeated
round counts                 1, 2, 3
linear round maps            512
all 8 affine offsets covered
```

Redundancy and decoder:

```text
six encoded source bits
+ one route-tag bit
route tag                    every Boolean function of any <=3 encoded bits
unique tag partitions        2,262
decoder                      arbitrary deterministic adaptive decision tree
read budget                  2 cells
```

This is favorable to the candidate. The route decoder is not restricted to the
originally suggested ARX instructions; after the encoder/tag are fixed, the
search grants every exact two-read decision tree independently to each query.
The seventh bit is also more generous than a strict proportional allocation of
8 GiB over a 405B-Q4 source at this tiny scale.

Affine offsets alter only output constants, not decision-tree partitions. The
implementation nevertheless checks every offset and reports zero invariance
failure.

## Exact exhaustive result

Complete encoder/tag pairs:

```text
512 linear maps x 3 round counts x 2,262 tags
= 3,474,432
```

Results:

```text
full 21/21 two-read encoders              0
raw-coordinate two-read coverage         15 / 21 = 71.428571%
best registered two-read coverage        16 / 21 = 76.190476%
```

The lexicographically first best candidate used one zero-linear Feistel round
and a three-input tag with truth-table integer 60. Its exact minimum query-depth
population was:

```text
depth 1     7 queries
depth 2     9 queries
depth 3     2 queries
depth 4     2 queries
depth 5     1 query
```

The one-coordinate and two-coordinate parity controls required exactly one and
two reads, respectively. Thus the rejection is not caused by a decoder that
cannot recognize simple fast paths.

## Decision

```text
REJECT_REGISTERED_AFFINE_FEISTEL_SINGLE_TAG_ROUTECELL_GRAMMAR
```

This rejects the concrete uniform grammar, not the unrestricted adaptive
nonlinear cold-source class. In particular, it does not reject:

```text
fully non-systematic seven-bit encodings
non-affine or multiword route cells
global cross-matrix advice
more general adaptive word probes
native numerical partial-output encodings
```

None of those objects is currently constructed. ROUTECELL therefore does not
survive as an executable core, while the broader theoretical gap remains open.

---

# 2. PROOFWAVE

## Structural accounting

A proof checker can verify a supplied trace more cheaply than executing the
original computation. That fact does not remove the work required to generate
the trace locally.

The complete charged equation is:

```text
total work
  = local causal proof generator
  + proof emission
  + independent checker
  + exact successor-state update
  + miss/fallback
```

All terms are nonnegative. Therefore PROOFWAVE becomes a new core only when the
**local causal proof generator itself** is independently specified and is
subdense. Calling that generator a SAT/SMT/bit-vector solver does not supply a
new information source.

## Exact finite lazy-decision controls

The Gate grants free logic between reads and computes zero-error stopping depth
on 12 binary source contributions.

Positive controls:

```text
one-coordinate decision        p50/p95 = 1 / 1 reads
OR certificate                 p50/p95 = 1 / 5 reads
```

The OR control confirms that the model recognizes genuinely short proofs when
one decisive contribution is present.

Balanced dense-margin control:

```text
output                         sign of 12 equal-magnitude contributions
threshold                      at least 6 positive contributions
mean reads                     10.0673828125 / 12
p50 reads                      10 / 12 = 83.333333%
p95 reads                      12 / 12 = 100%
p50 miss factor vs 8/675       70.3125x
p95 miss factor vs 8/675       84.375x
```

Because every unread contribution is exchangeable in this control, adaptive
address choice cannot improve on the exact stopping distribution. A parity
control also requires 12/12 reads, but parity is retained only as an integrity
diagnostic and is not the primary scientific result.

## Real-checkpoint evidence boundary

The closest committed evidence is
`docs/research/EXP_096A_LATEST_RESULT.md`, blob
`fe9247a2173e95a4ad64912913a34b38eb3403bf`.

On pinned SmolLM2-135M, a target-seeing residual-margin audit found sparse
coefficient certificates for every registered build and holdout row:

```text
build certificates                 384 / 384
untouched holdout certificates      384 / 384
holdout support p50/p95             5 / 13
perfect-block compressed traffic    0.619070788%
fine dense arithmetic               100%
causal coefficient generator        NOT CONSTRUCTED
```

That is useful evidence that a compact certificate may exist **after** the
relevant target information is known. It does not give PROOFWAVE a causal
method for obtaining the certificate without the dense arithmetic.

## Decision

```text
REJECT_PROOFWAVE_AS_STANDALONE_NEW_INFORMATION_SOURCE
RETAIN_PROOF_CHECKER_AND_FAIL_CLOSED_SOLVER_AUXILIARY
```

This does not reject every checkpoint-specific SAT/SMT proof search. Reopening
requires an explicit causal local trace generator with finite-word operations,
physical checkpoint addresses, proof/state bytes, and complete fallback cost
below the target. The generator—not proof packaging—is the core invention.

---

# 3. REACHQUOTIENT

## Exact contract

The principle is mathematically valid. A compact state code is acceptable only
when it is an exact Moore/bisimulation quotient:

```text
same class
  -> same declared output
  -> same next class for every legal future input/RNG continuation
```

The Gate uses exact partition refinement, not approximate state similarity.

## Positive control

A duplicated 3-bit shift machine contains 32 native states but only eight
behavior classes:

```text
native states       32
minimal classes      8
class fraction      25%
```

This proves the minimizer finds real quotient structure when redundant copies
exist.

## Strong exact adversary

For a k-bit shift register, the current output exposes the oldest bit and future
legal input symbols shift every remaining bit into the output. Consequently any
two distinct states are separated by a suffix of length at most k.

Exact results for k=2 through k=8:

```text
4/4, 8/8, 16/16, 32/32, 64/64, 128/128, 256/256 classes
```

Every registered adversarial case retained 100% of its native state classes.

## Random deterministic machines

One hundred seeded two-input/two-output machines were minimized at each size.

| Native states | Minimum classes | p50 classes | p95 classes | Fully minimal samples |
|---:|---:|---:|---:|---:|
| 16 | 13 | 16 | 16 | 70% |
| 32 | 29 | 32 | 32 | 73% |
| 64 | 61 | 64 | 64 | 83% |
| 128 | 126 | 128 | 128 | 83% |

The p50 class fraction was 100% at every size. This is not evidence about one
Transformer checkpoint; it shows why a small exact quotient cannot be assumed
from state continuity or finite precision alone.

A separate control confirms that restricting the future-input contract merges
more states. Such merging is invalid for VORTEX unless the declared contract is
actually restricted; a fixed observed continuation cannot be substituted for
all legal future suffixes.

## Real-checkpoint evidence boundary

`docs/research/EXP_089A_LATEST_RESULT.md`, blob
`aed26836718caf953949599361c0df66cd7a1709`, established an exact state witness
on pinned SmolLM2-135M:

```text
compiled state                    exact prefix tokens + exact RNG bytes
transitions                        64
all token/logit/cache/RNG mismatch 0
reference target calls             68
replay target calls                612
replayed token positions           3,791
```

That witness reconstructs the official state by replaying the checkpoint. It
is an identity representation of causal history, not a compressed H/lambda/delta
transition program and not a fast executor.

## Decision

```text
VALID_BISIMULATION_INTERFACE
NO_COMPRESSED_REACHABLE_QUOTIENT_CONSTRUCTOR
DO_NOT_PROMOTE_REACHQUOTIENT_AS_CORE
```

A checkpoint-specific quotient remains logically possible. Promotion requires
an automatically compiled `H`, output function, and next-class transition,
plus a machine-checkable all-suffix congruence proof and a complete 8-GiB/byte/
instruction equation. None exists in the current proposal.

---

# Overall result

```text
NO_ROUTE_PROOF_REACH_CORE_PROMOTED
```

| Candidate | Valid part | Decisive result | Classification |
|---|---|---|---|
| ROUTECELL | adaptive data-dependent addresses are a genuine open execution class | registered 3,474,432 encoder/tag pairs produced zero 21/21 two-read solution | concrete grammar rejected; broad class open without constructor |
| PROOFWAVE | independent proof checking can be exact and fail closed | local generator unspecified; balanced exact control retains p50/p95 83.33%/100%; real certificate evidence still has 100% dense arithmetic | standalone core rejected; checker auxiliary |
| REACHQUOTIENT | exact bisimilar successor state is allowed | positive quotient works, but all shift adversaries and random p50 retain 100% classes; real witness is dense replay | valid interface, no compressed constructor |

No numbered experiment, checkpoint run, backend, or hardware action is
authorized by this result.

## Next constructive Gate

The only materially open class among the three is a more general adaptive
nonlinear cold encoding. Another local seed or renamed router is not enough.
The next candidate must first provide one **non-entrywise global route grammar**
that works at two source scales with the same finite-word rules:

```text
scale 1     complete tiny exact synthesis/falsification
scale 2     same grammar, no per-size truth table
output      exact query result or native reduction-boundary partial
state       explicit nontrivial successor-state path
cost        encoder storage + metadata + addresses + probes + decode + fallback
```

Only after both scales pass may a native Q4/BF16/FP32 lift or public-checkpoint
operation replacement be attempted.

Registered next-Gate label:

```text
NON_ENTRYWISE_GLOBAL_ROUTE_GRAMMAR_WITH_TWO_SCALE_EXACT_DECODER_AND_NATIVE_LIFT_PREREQUISITE
```

## Reproduction

```bash
python experiments/e0_route_proof_reach_validation/run_validation.py \
  > results/e0_route_proof_reach_validation/summary.json
python -m pytest -q tests/e0_route_proof_reach_validation/test_validation.py
sha256sum -c results/e0_route_proof_reach_validation/checksums.sha256
```

Locally validated result:

```text
summary regeneration      byte-identical
focused tests             5 passed
checksum verification     pass
checkpoint/model run      not performed
hardware run              not performed
```

## Claim boundary

```text
405B execution                                      NOT TESTED
physical complete 8-GiB allocation                 NOT TESTED
native Q4/BF16/FP32 lift                            NOT TESTED
actual Transformer operation replacement           NOT TESTED
same-machine 4B-class p50/p95                       NOT TESTED
general nonlinear adaptive cell-probe impossibility NOT ESTABLISHED
```
