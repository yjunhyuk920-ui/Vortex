# Experiment 046 — Exact Future-Behavior DAG Quotient

Last updated: 2026-08-03 (Asia/Seoul)

## Evidence level and purpose

This is an E1/E2 finite-horizon compiler compression Gate. It consumes the authoritative Experiment 045 TinyLlama traces and performs no new model calls.

Experiment 045 proved that complete token-prefix identity gives sound replay but no reuse across distinct compiled prompts: 64 path records produced 64 unique nodes at horizon eight, and held-out start coverage was zero.

Experiment 046 asks a narrower but stronger compression question:

> If the compiler is allowed to see the complete exact future continuation of every already compiled path, how small can the deterministic decision graph body become without changing any checked token?

This is an optimistic offline upper bound on graph-body compression. It is not a causal start router for unseen prompts.

## Exact behavioral equivalence

For a finite continuation suffix, define nodes backward:

```text
terminal = -1
signature_t = (exact_token_t, successor_signature_(t+1))
```

Two states merge only when their signatures are exactly equal. Equivalently, their complete remaining token suffixes are identical.

This relation is sound for the declared finite horizon:

- the current token is identical;
- the successor node is identical;
- by induction, every remaining token is identical.

No hidden-state approximation, semantic class, learned router, or probability threshold is used.

## Input evidence

The Gate reads:

```text
results/decision_index_compiler_gate.json
experiments/decision_index_compiler_grammar.json
```

Authoritative source:

```text
PR #52 head: 5fb32b30ceda3e362da7b6ee9ed2dee0c93231e5
compiler workflow: 30786618783
```

The source traces contain eight compiled A/B grammar paths, four held-out C paths, one exact duplicate-control mapping, and eight generated tokens per path.

## Quotient construction

For every measured horizon `H` in `2, 4, 8`:

1. take the first `H` exact generated tokens of each compiled path;
2. scan each path backward;
3. intern `(token_id, successor_address)` pairs;
4. store one start address per prompt;
5. separately add the exact duplicate-control start, which must equal its source start;
6. export the maximum-horizon DAG to compact40 using the Experiment 044 VM.

The four-bit codebook remains exact and collision-free.

## Compression accounting

Report:

```text
raw path records = compiled prompt paths * horizon
minimal quotient nodes
node reduction
compression fraction
unique suffixes by remaining length
full-continuation equivalence classes
start-router entries
compact40 file bytes
```

At each remaining length `r`, the number of unique suffix tuples is the exact number of quotient nodes at that depth. Summing those counts must equal the total DAG nodes.

Compression attribution distinguishes:

```text
exact duplicate prompt savings
cross-distinct-prompt future-suffix savings
terminal-token merging
longer identical suffix merging
```

## Exact replay

Every compiled and duplicate-control path is replayed from its start address through `DecisionVMReader`.

Promotion requires:

```text
all paths exact
all tokens exact
duplicate control shares source start
compact40 checksum verified
node accounting exact
```

## Held-out body coverage versus start routing

Held-out C traces remain outside the graph.

Two different metrics are mandatory:

### Future-aware body oracle

For each held-out step, check whether its complete remaining token suffix exists somewhere in the compiled DAG. This oracle sees future ground-truth tokens and is not deployable. It measures only whether the graph body already contains the behavior.

### Causal start-router coverage

A held-out prompt has no compiled prompt-to-start entry. Therefore its deployable exact start coverage is zero unless a separately proven router exists.

The Gate must not convert future-aware suffix existence into a causal routing claim.

## Decision boundary

Possible outcomes:

```text
little or no quotient compression:
    reject finite-horizon suffix DAG as a meaningful storage mechanism

strong graph-body compression but zero held-out start routing:
    accept quotient compiler; identify the start-router barrier as primary

strong compression and a separately sound held-out router:
    advance to broader grammar and horizon
```

Experiment 046 does not contain such a router. Its purpose is to isolate whether graph-body duplication or start selection is the dominant remaining problem.

## Strict scope

- Future continuations are used only for offline compilation/evaluation.
- The quotient is exact only for compiled paths and measured horizons.
- A small DAG does not imply arbitrary prompt coverage.
- Start mapping cost and held-out misses remain explicit.
- No TinyLlama result is projected to 405B as proof.
- No CI timing is target hardware evidence.

The next research direction must follow the measured result. If the body compresses strongly, work moves to a sound start router or certified state equivalence. If it does not, even future-aware path minimization is insufficient and index enumeration remains the dominant barrier.
