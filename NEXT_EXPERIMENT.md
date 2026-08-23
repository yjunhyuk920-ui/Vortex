# Next Experiment — EXP-104A

## Checkpoint-Native Resident Slice Transducer Reality Gate

### Why this is next

EXP-102A rejected independent external drafts as a long exact segment source. EXP-103A rejected exact native-rounding page skipping as a universal weight-stream core. The next Gate must therefore change the **causal information source** and remain executable under the real 8-GiB budget.

### Three materially different principles

1. **Checkpoint-native resident slice transducer — selected.** Keep a fully charged subset of target-derived draft-only weights inside 8 GiB, stream only the exact token embedding rows needed online, generate a causal draft, and use the unchanged target for exact verification.
2. **Streaming page-shadow self-draft — secondary.** Generate draft evidence only while the corresponding exact target pages are physically transferred. Every duplicate read, synchronization, branch, and target position remains charged; transfer overlap is never called free.
3. **Deferred mismatch state closure — auxiliary.** Commit the exact target mismatch token and materialize its target state at the next sweep. This may remove a repair call but cannot promote a candidate whose accepted segment or dense arithmetic remains inadequate.

### Static Gate before model execution

Freeze a concrete representation and report:

```text
draft resident bytes
draft LM-head bytes
embedding-row traffic
KV/cache bytes
workspace/fragmentation
metadata/packing bytes
peak total <= 8 GiB
```

No unmeasured compression, free transforms, future target state, or perfect selector.

### Causal and exactness Gate

On build-only and untouched holdout prompts, measure:

```text
actual accepted prefix A
actual target positions N
N/A
draft generation time
target verification time
repair/rebuild time
exact committed tokens
exact terminal target state
```

The target-scale raw-BF16 p50 floor is:

```text
A >= 339
```

unless the implemented executor measurably reduces exact target sweep bytes. Merely increasing configured K does not count; actual accepted prefix is authoritative.

### Stop rule

Reject immediately when any of the following holds:

- the static resident ledger exceeds 8 GiB;
- the source cannot causally generate an actual segment approaching 339 tokens;
- exact target state requires an uncharged extra sweep;
- the mechanism is only an external draft under a new name;
- `r=1` and the block/throughput ledger cannot meet the final latency equation.

Do not run GitHub Actions after local validation. Commit the locally validated source, evidence, checksums, decision, ledgers, and README, then verify the remote SHA.
