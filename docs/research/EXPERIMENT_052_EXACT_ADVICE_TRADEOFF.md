# EXP-052 — Runtime-Only Exact Advice Tradeoff and Constraint-Closure Gate

## Status

```text
Implementation branch: research/exp-052-exact-advice-tradeoff
Gate registration: COMMITTED BEFORE REAL-CHECKPOINT RUN
Scientific result: PENDING
Phase: A/B with small-checkpoint exact-state observation
Evidence ceiling: E1
Complete real operation replacement: false
Phase D: NOT TESTED
```

## Question

Can automatic target-specific exact advice avoid enough target execution on unseen held-out states while its build calls, exact collision witness, hot index, cold storage, lookup probes, fallback, and required reuse are fully charged?

This Gate evaluates enumerative exact advice. It does not claim an unconditional lower bound for every possible program compiler or symbolic representation.

## Exact runtime contract

```text
Compile(target revision, decode contract, build states)
    -> immutable exact advice entries

Query(target revision, decode contract, exact state)
    -> exact hit with collision witness
    OR exact target fallback
```

No approximate hit is allowed. A digest match without the complete registered witness is a miss. A conflicting value, namespace mismatch, malformed digest, corruption, or accounting inconsistency fails closed.

## Conditions

### P0 — exact full-prefix table

For each pinned target, construct six leave-one-family-out folds. Build on all 64 measured states from five prompt families and query all 64 states from the held-out family.

Key and witness:

```text
exact target revision
exact decode contract
complete uint32 prefix token sequence
```

A miss performs one exact target fallback. Build and evaluation families remain disjoint.

### S0 — exact KV-state table

Use the same folds. Canonically serialize the target KV cache tensors including dtype, shape, tensor order, and bytes.

Index and witness:

```text
SHA-256 state bucket
SHA-512 state digest
current token
position
complete exact prefix
```

The full prefix is retained as an exact deterministic-state witness. Raw KV byte size is measured separately. A hash collision without the complete witness is a miss.

### R0 — same-state replay positive control

Query every build state against its own table. Expected result:

```text
100% exact hits
zero wrong hits
zero target fallback
```

This validates implementation only. It is not held-out generalization. Every distinct entry still costs one target build call, so one replay has target-forward component 100%; at least 85 exact repetitions are required even under perfect coverage.

### I0 — independent-state coverage and storage adversary

For a deterministic family of `N` independent exact states, store exact advice for `M` states and verify measured hit rate `M/N` with zero wrong hits.

Executable control:

```text
N = 262,144
coverage = 0%, 10%, 50%, 90%, 99%, 99.9%, 100%
```

Storage closure uses:

```text
N = 2^48 independent states
8 GiB hot-index budget
48-byte hot slot
1 TiB cold-advice budget
```

This construction evaluates the enumerative table family only. It does not prove that every possible target admits no shorter program.

## Corpus

Pinned tokenizer and Dense checkpoints:

```text
EleutherAI/gpt-neo-125M @ 21def0189f5705e2521767faed922f1f15e7d7db
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Families:

```text
English narrative
Korean
code
mathematics
structured JSON
identifier boundary
```

For each model/prompt:

1. exact prefill;
2. one exact warm-up token;
3. 64 exact incremental states with target KV cache;
4. record prefix, next token, cache digests/bytes, target calls, elapsed CPU, and prompt hash.

Expected natural state rows:

```text
3 models * 6 families * 64 states = 1,152
```

## Charged target equation

For exact hit rate `h`, total logical build calls `B`, evaluation query count `Q`, and exact repetitions `R`:

```text
fully_accounted_target_fraction
    = B / (Q * R) + (1 - h)
```

Final required fraction:

```text
<= 0.011851851851851851
```

Consequences fixed before the run:

```text
infinite build reuse requires h >= 98.8148148%
R = 85 requires h >= 99.9912854%
perfect coverage with one build per state requires R >=85
```

No amount of build reuse can rescue a fallback rate already above 1.185185%.

## Advice storage

Report for every fold:

```text
entry count
serialized prefix-table bytes
serialized state-table bytes with prefix witness
raw KV bytes represented
average and maximum prefix length
lookup probes
hot index slots required
cold advice bytes
```

Storage projections are compared against 8 GiB hot index and 1 TiB cold advice. Metadata bytes are never relabeled as model traffic or latency.

## Required raw evidence

```text
raw/state_records.jsonl
raw/fold_rows.jsonl
raw/replay_controls.json
raw/independent_state_audit.json
raw/checkpoint_manifest.json
processed/aggregate.json
summary.json
logs/run.log
artifacts/environment.json
artifacts/contract.txt
checksums.sha256
```

## Pre-registered rejection Gate

Reject enumerative exact advice as the primary runtime if any condition holds:

```text
wrong exact advice hits >0
build/evaluation family leakage >0
any held-out family exact hit rate <98.8148148%
any held-out family online fallback fraction >1.185185%
median observed natural exact-state reuse <85
p90 fully-accounted target fraction >1.185185%
1 TiB independent-state minimum fallback fraction >1.185185%
```

Failure decision:

```text
REJECT_ENUMERATIVE_EXACT_ADVICE_AS_CORE_RETAIN_FAIL_CLOSED_TABLE_AUXILIARY
```

A failed natural Gate also forbids enlarging only the table, changing hash width, or hiding build/fallback cost as a rescue.

## Positive result boundary

A positive held-out result would still not prove 405B success. Promotion would require:

```text
causal automatic compiler from the unmodified checkpoint
real unseen-state target operation replacement
measured physical query/storage cost
complete generation with exact fallback
p90 fully accounted fraction <=1.185185%
hot state <=8 GiB
medium/large non-degrading scaling
Phase-D profiler evidence
```

## Claim boundary

```text
405B execution: NOT TESTED
8 GiB VRAM: NOT TESTED
CUDA/PCIe/SSD/TTFT/tokens per second: NOT TESTED
complete real operation replacement: false
Phase D: NOT TESTED
```

## Commands

```bash
python -m pytest -q tests/exp_052
bash experiments/exp_052/run_current_env.sh
bash experiments/exp_052/reproduce.sh
```

<!-- EXP-052-AUTHORITATIVE-FINAL -->
## Final authoritative result

Authority: `results/exp_052/summary.json`; workflow `30811429049`; source head `d4c2328027a5377b997e9ee1d8df0f55190fb652`; artifact `8854946309`; ZIP SHA-256 `1beb137e1ee14fe80ded0a3309c4ed297035d552a46bf901b2e4233ab95549ca`.

1,152 exact warm states and 36 leave-one-family-out rows produced zero wrong hits and zero build/evaluation leakage, but P0 prefix and S0 KV-state held-out hit rates were 0% in every family. Fallback was 100%, natural exact reuse median/max was 1/1, and p90 fully-accounted target fraction was 6.0 (600%). Same-state replay was 100% exact and required at least 85 repetitions. Under 8 GiB hot index plus 1 TiB cold advice, combined coverage of 2^48 independent states was 6.357828752356909e-7, leaving fallback 0.9999993642171248.

Decision: `REJECT_ENUMERATIVE_EXACT_ADVICE_AS_CORE_RETAIN_FAIL_CLOSED_TABLE_AUXILIARY`.
