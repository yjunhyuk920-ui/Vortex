# Next Experiment

## Closed Gate — EXP-051

Authoritative PR and workflow record:

```text
PR #61
workflow 30808285251
source head SHA 01c2cde4ab6b1122d50392cdfb08ad82923524f8
workflow merge SHA 8b6b164c80a828aa9827fd6b4ea85cd659917e15
artifact 8853920561
artifact ZIP SHA-256 50062060f9f31495ed6b4d96df0f2cb12b38478b1e8583a4ecdd15e621b317d7
main merge commit 7aad7b3e1b3d783857406ebb6435058a2aecc724
```

MEASURED in the successful Gate and recorded in PR #61:

```text
3 targets * 6 families * 64 warm states = 1,152
final-depth reconstruction mismatches 0
future generated token uses 0
suffix-stable median block fraction 25%
suffix-stable p90 block fraction 37.5%
median favorable logical byte fraction 82.2068758%
p90 favorable logical byte fraction 99.8010577%
transient first-match failures 428/1,152 =37.1527778%
late-final-layer adversary stable depth 8/8
```

Decision:

```text
REJECT_LAYER_FINALIZATION_TAIL_SKIP_CORE_RETAIN_ORACLE_AUXILIARY
```

The original EXP-051 artifact is no longer retrievable through the current connector and `results/exp_051/` was not installed before merge. This is a provenance defect, not a change to the recorded scientific decision. EXP-052 must freeze evidence into the repository immediately after the authoritative run.

## EXP-052 — Runtime-Only Exact Advice Tradeoff and Constraint-Closure Gate

### Mechanism

Automatically compile exact target-specific advice from an unmodified checkpoint and a finite set of build states:

```text
Compile(target revision, decode contract, build states)
        -> immutable exact prefix/state advice

Query(exact runtime state)
        -> exact witnessed hit
        OR one exact target fallback
```

The tested family is enumerative exact advice. Digest-only or approximate matches are forbidden.

### Conditions

```text
P0 exact full-prefix table, six leave-one-family-out folds
S0 actual KV SHA-256 bucket + SHA-512/full-prefix witness
R0 same-state replay positive control
I0 independent-state exact M/N and storage-budget control
```

Natural corpus:

```text
3 pinned TinyStories targets
6 held-out families
1 warm-up token
64 exact incremental states per prompt
1,152 exact natural states
```

### Exact cost closure

For hit rate `h`, build calls `B`, query count `Q`, and identical evaluation repetitions `R`:

```text
fully_accounted_target_fraction = B/(Q*R) + (1-h)
```

Required final fraction:

```text
<=0.011851851851851851
```

Pre-registered consequences:

```text
infinite build reuse requires h >=98.8148148%
85 repetitions require h >=99.9912854%
perfect one-build-per-state advice requires at least 85 exact repeats
```

### Storage closure

```text
hot index budget: 8 GiB
hot slot: 48 bytes
cold advice budget: 1 TiB
independent storage adversary: 2^48 exact states
```

Prefix and actual KV-state serialized sizes, raw KV bytes represented, probes, target build calls, fallback calls, and natural reuse are all charged separately.

### Pre-registered rejection Gate

Reject enumerative exact advice as the primary runtime if any condition holds:

```text
wrong exact advice hit >0
build/evaluation family leakage >0
any held-out family exact hit rate <98.8148148%
any held-out family online fallback >1.185185%
median observed natural exact reuse <85
p90 fully-accounted target fraction >1.185185%
1 TiB independent-state minimum fallback >1.185185%
```

Failure decision:

```text
REJECT_ENUMERATIVE_EXACT_ADVICE_AS_CORE_RETAIN_FAIL_CLOSED_TABLE_AUXILIARY
```

### Claim limit

EXP-052 can reject the implemented enumerative exact-advice family. It does not prove an unconditional lower bound for every conceivable symbolic compiler or program representation.

```text
Phase A/B with small-checkpoint observation
Evidence ceiling E1
complete real operation replacement false
405B / 8 GiB / CUDA / PCIe / SSD / TTFT / tokens/sec NOT TESTED
```

### Next exact action

1. implement one-pass pinned state-corpus generator with canonical KV hashing;
2. run P0/S0 six-fold held-out queries and R0 replay controls;
3. execute I0 M/N and storage closure;
4. freeze all raw evidence and checksums into `results/exp_052/`;
5. update all durable research documents;
6. merge only after Python 3.10/3.12 CI and evidence validation pass.
