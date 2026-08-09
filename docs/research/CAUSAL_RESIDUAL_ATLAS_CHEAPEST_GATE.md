# Causal Residual Atlas Cheapest Real-Weight Gate

Date: 2026-08-08
Status: PREREGISTERED, NO MODEL RESULT
Evidence ceiling if executed: E1 favorable-oracle small-real-checkpoint falsification

## Question

Before building a residual-bound propagator, cold runtime, backend, or kernel,
does the most favorable fixed rank-16/64-column Causal Residual Atlas retain the
unchanged greedy output at even the first post-prefill decode call on a pinned
required pair of real projections?

This is the smallest decisive real-weight test of the premise admitted by
`E0_CAUSAL_RESIDUAL_ATLAS_SOURCE.md`. It deliberately grants the candidate an
illegal exact-reference center, enumerates every permitted page, changes only
one projection at a time, and leaves the rest of the model dense. A failure
therefore occurs before selector, numerical enclosure, simultaneous error,
fallback, physical page service, and complete-memory costs can make the path
worse.

No experiment number is assigned and no checkpoint is run in this
preregistration commit.

## Frozen decision vocabulary

```text
PROMOTE_CAUSAL_RESIDUAL_ATLAS_TO_CAUSAL_PAIR_AND_BOUND_GATE
REJECT_RANK16_PAGE64_CAUSAL_RESIDUAL_ATLAS_FAST_PATH
INVALID_CAUSAL_RESIDUAL_ATLAS_ORACLE_CONTROL_FAILURE
INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION
```

A promotion is only permission to test the legal pair-constructed center and
an outward-rounded residual bound. It is not a Core Candidate promotion, an E2
operation replacement, or evidence about physical speed.

## Why this is the cheapest decisive population

The already pinned evaluation split has 18 prompts, three in each of six
families. Chat-template lengths are 25 through 46 tokens, so every prompt can
construct rank 16 before the measured decode call. The frozen EXP-076 trace
contains eight exact teacher positions per prompt.

The Gate uses only:

```text
evaluation prompts                  18
teacher position                     1
measured positions/prompt            1
layer                               11
projections                  q_proj, down_proj
projection branches                 36
token states                         18
```

Teacher position zero is the logit produced by the final prompt token. Using
that same prompt position in both the Atlas basis and the query would make the
residual trivially zero. Teacher position one is instead the first genuine
post-prefill module call: the first exact generated token is processed to
predict the second generated token. The basis is already fixed and cannot see
that current projection input or any later generated token.

Layer 11 is the registered middle full-attention layer. `q_proj` exercises the
1,024-wide shared attention input and `down_proj` the 3,584-wide MLP
intermediate. Both are mandatory unchanged-checkpoint operations. If either
required operation fails while tested alone, the registered per-projection
Atlas fast path cannot certify the complete token. A pass expands depth,
positions, and simultaneous composition later; it does not infer them.

## Frozen inputs

```text
checkpoint             Qwen/Qwen3.5-0.8B
revision               2fc06364715b967f1860aea9cf38778875588b17
weight SHA-256         04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
prompt file            experiments/exp_076/prompts.json
prompt SHA-256         46c9779c24e59f356247145f7ef870a1b82ba4907eb9dffcd48652e519f7612a
teacher trace          results/exp_076/raw/case_rows.jsonl
trace SHA-256          1e921698ce8ee522c0d3cb9b8b9004139beb54cc2dee82e1a2aa0a08fa245e4f
trace evidence commit  55b79937c1f21887ae76b7e56ad61ba7dde8322a
runtime                Python 3.12.13, Torch 2.6.0+cpu,
                       Transformers 5.12.0, BF16 eager, CPU, 8 threads
```

Only the 18 `evaluation` prompts determine the scientific result. The six
`build` prompts may be used for implementation smoke tests but cannot select a
rank, page width, layer, projection, token position, threshold, or ordering.

## Frozen causal prefix policy

For each prompt and each of the two registered projections:

1. execute the unchanged prompt prefill and capture every prompt-position
   projection input `X` and its unchanged image;
2. convert the immutable BF16 weight values and captured inputs to float64 for
   this favorable geometry Gate;
3. compute the thin SVD of prompt inputs only;
4. define the numerical-rank threshold as
   `max(rows, columns) * eps_float64 * sigma_max`;
5. retain the first `min(16, numerical_rank)` right singular vectors;
6. canonicalize each vector sign by making its largest-absolute coordinate
   nonnegative, breaking an equal-coordinate tie by the lowest index;
7. freeze the basis for the measured post-prefill call.

The current decode activation, current dense image, teacher token at position
one, later teacher tokens, other prompts, and evaluation outcomes may not enter
the basis. A prompt with fewer than 16 usable directions is not excluded; its
available rank is recorded and the resulting token remains in the denominator.

The top-16 SVD is more favorable than an incremental implementation because it
uses all legal prompt inputs to minimize projection residual energy. Its build
cost is granted free in this Gate. A failure cannot be attributed to an
inferior online orthogonalizer.

## Frozen page policy

Use contiguous native-column pages of exactly 64 coefficients:

```text
q_proj input width       1,024 -> 16 candidate pages
down_proj input width    3,584 -> 56 candidate pages
page candidates/prompt               72
evaluation prompts                    18
total oracle page candidates       1,296
```

Exactly one page is revealed for each projection branch. Every possible page
is enumerated; there is no learned, causal, heuristic, or post-result subset.
The lower page index breaks exact ties.

This applies the same `ceil(0.2% * page_count)` rule as the registered target
screen. Small-model rounding grants a much larger coefficient fraction than at
405B (`6.25%` for `q_proj`, `1.785714%` for `down_proj`), so it is intentionally
favorable. Rank 16 with the same one-page set also weakly dominates rank 8 in
input-residual L2 at these widths. Rank 32 already fails the target traffic
screen. No rank/page sweep is permitted after observing the result.

## Native-anchored favorable candidate

Let `Q` be the legal prompt-only basis, `x` the current post-prefill projection
input, and `u = x - QQ^T x`. Let `y_native` be the unchanged dense module
output. The evaluator converts the unchanged BF16 weight values to float64 and
constructs, for every page `G`,

```text
y_G = y_native - W u + W[:,G] u[G].
```

This is deliberately stronger than the deployable `Z Q^T x + W[:,G]u[G]`:
it grants the exact current dense image to anchor the center, removes native
rounding error from pair construction, and scans every unread weight in the
evaluator. It exists only to upper-bound the one-page residual premise. The
candidate path may not cite this center as causal or deployable.

For each page, patch `y_G` only at the current token position and only at the
one registered projection. All other positions, projections, layers,
attention, MLPs, norms, the tied LM head, and cache work remain unchanged and
dense. Candidate values are converted to the module's native output dtype at
the patch boundary. The unchanged path and an all-page identity arm must
reproduce the frozen baseline before the scientific result is valid.

The Gate also records, without promotion, the exact unread L2 error and the
Frobenius page-bound radius from the existing reference. Neither is treated as
an end-to-end certificate.

## Frozen exact-reference page oracle

For each projection branch, run every page candidate to the unchanged final
logits. The evaluator may inspect the exact baseline logits for that same
position. It may not inspect a later generated token when constructing `Q`,
`u`, or a candidate projection output.

The page is selected lexicographically:

1. prefer candidates whose greedy top-1 equals the unchanged target;
2. among them, minimize `KL(softmax(target) || softmax(candidate))`;
3. break equal KL by the lowest page index;
4. if no page preserves top-1, retain the minimum-KL page for diagnostics and
   mark the branch failed.

This exact-reference enumeration is a non-deployable favorable oracle. A pass
does not supply a causal selector. A failure means no one-page choice in the
registered matrix can meet the frozen output condition even after seeing the
answer.

## Frozen output and coverage Gate

A token state succeeds only if both its layer-11 `q_proj` branch and its
layer-11 `down_proj` branch have at least one top-1-preserving page. The two
branches are tested independently with the rest of the model dense. This is a
necessary condition, not a simultaneous partial-model execution result.

The target rank-16 accounting requires traffic-governed coverage

```text
rho >= 0.998998405303215166463
```

over the finite 18-token population. Therefore:

```text
required token successes       ceil(18 * rho) = 18
maximum token failures                              0
one failure coverage                         17/18 = 94.444444%
required successes/family       ceil(3 * rho) = 3
```

All of the following are required for promotion:

- checkpoint, prompt, trace, dependency, and population hashes match;
- unchanged baseline and all-page identity controls have zero top-1 mismatch;
- basis/current/future audit has zero leakage;
- malformed or non-finite state count is zero;
- token-state success is 18/18;
- every family succeeds 3/3;
- all 36 selected projection branches preserve top-1;
- mean target-to-selected-candidate KL is at most `0.02`;
- NumPy linear p95 KL is at most `0.05`.

The KL distribution is over the 36 independently selected branch candidates.
Token coverage is the conjunction over the two branches and is the controlling
fallback metric. Report branch, token, family, projection, page-index, residual
norm, radius, KL, prompt-rank, and prompt-length rows even on early scientific
failure.

The finite corpus cannot statistically prove 99.89984% population coverage.
Zero failures only allow the mechanism to survive this E1 falsification and
advance to a broader preregistered Gate.

## Controls and failure classification

Control failure, with no scientific interpretation:

- any pinned hash or dependency mismatch;
- any frozen teacher-trace mismatch;
- the hook changes a non-target position or module;
- unchanged, dense-patch, or all-page identity changes the frozen top-1;
- basis includes the current decode input, another prompt, or later teacher
  position;
- repeated runs change page selection or deterministic scientific payloads;
- malformed shape, rank, page, cache, or non-finite state is not rejected.

Infrastructure failure, with no scientific interpretation:

- missing pinned local payload;
- dependency/import incompatibility;
- memory exhaustion, timeout, filesystem failure, or incomplete output bundle.

Scientific rejection occurs immediately after valid controls if any promotion
condition fails. Complete all already computed rows, write checksums, and stop.
Do not respond with another rank, page width, layer, token, prompt subset,
numerical tolerance, or page-order sweep.

## Core twelve-question audit

1. **Operation replaced:** one required dense `q_proj` or `down_proj` MatVec at
   the first post-prefill decode call.
2. **New information:** prompt-only rank-16 input subspace; its legal image is
   supplied by committed prefix pairs in the eventual candidate.
3. **Selector cost:** granted free and stronger than deployable by enumerating
   all 1,296 page candidates.
4. **Wrong-skip detection:** final top-1/KL exact-reference comparison in this
   Gate; a later runtime still needs an outward-rounded certificate.
5. **Fallback:** unchanged dense module/token path; any unresolved branch is a
   token fallback in coverage accounting.
6. **Output contract:** no candidate branch may silently change greedy top-1;
   distribution drift is bounded by the frozen mean/p95 KL thresholds.
7. **Scaling:** the Gate uses the target rank/page policy but grants favorable
   small-width rounding; a pass makes no scale claim.
8. **Sequential work:** the current page candidate is derived from prompt state
   and current activation, not future generated tokens.
9. **Movement:** target E0 accounting remains a `0.980995178 GiB` capsule plus
   `1,009` logical target pages/token; this Gate measures no physical movement.
10. **405B minimum:** the registered favorable path remains
    `1.085025716%` amortized traffic and `0.557958575%` operations.
11. **Gap:** the path has only `0.100159470%` fallback headroom at the traffic
    Gate and therefore permits zero failures in this finite population.
12. **Strongest falsification:** a single valid token state for which either
    required projection has no top-1-preserving page, despite exact-reference
    center and exhaustive page enumeration.

## Stop rule and authorized continuation

On scientific failure:

```text
REJECT_RANK16_PAGE64_CAUSAL_RESIDUAL_ATLAS_FAST_PATH
```

This closes the registered rank-16/page-64 per-projection Atlas from the
primary track. A globally coordinated cross-layer cancellation mechanism would
need a new information source, sound global contract, and fresh E0 equation;
it is not a page/rank rescue.

On pass:

```text
PROMOTE_CAUSAL_RESIDUAL_ATLAS_TO_CAUSAL_PAIR_AND_BOUND_GATE
```

Only the following next work is authorized:

- replace the illegal native-anchored center with `Z` built from captured
  committed prompt pairs only;
- add outward-rounded native numerical bounds;
- expand to all eight frozen positions and early/middle/late q/down matrices;
- test simultaneous bound composition and charge selector/build/fallback.

No pass authorizes a cold scheduler, decoder integration, CUDA/kernel work,
larger checkpoint, target Ubuntu command, E2 claim, or 405B feasibility claim.

## Reproduction of the preregistered arithmetic

```powershell
$env:PYTHONPATH = ".;.deps"
python scripts/derive_causal_residual_atlas_gate.py
python -m pytest tests/test_causal_residual_atlas_gate.py -q
```

Expected invariants:

```text
token states                         18
projection branches                  36
q/down pages                      16/56
oracle page candidates             1,296
required token successes              18
maximum token failures                 0
required successes/family               3
```

## Claim boundary

This commit preregisters a cheap Gate and validates only its arithmetic/oracle
selection reference. It contains no real-weight result. Small-checkpoint
residual behavior, legal pair-center construction, end-to-end certification,
actual operation replacement, physical I/O, peak VRAM, latency, 122B/405B,
and E2-E7 remain `NOT TESTED`.
