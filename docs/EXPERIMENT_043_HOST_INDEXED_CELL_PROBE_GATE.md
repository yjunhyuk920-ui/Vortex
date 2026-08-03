# Experiment 043 — Host-Indexed Exact-Decision Cell-Probe Gate

Last updated: 2026-08-03 (Asia/Seoul)

## Evidence level and purpose

This is an E1 algorithmic and prototype Gate. It is not a real 405B GPU benchmark and does not use CI timing as target-hardware evidence.

Experiment 042 proved that a complete exact final-decision representation for a constructed Q4 Llama-style family needs 26.1585866455 GiB and cannot be entirely resident in 8 GiB. It did not prove per-token traffic because a host-indexed representation might return only the currently required code.

Experiment 043 tests that escape directly.

## Explicit adaptive pointer-table model

The host representation is an array of exact records:

```text
record[address] = (q4_value, next_address)
```

A prompt selects one chain start. Autoregressive execution is:

```text
value_t       = record[address_t].q4_value
next_token_t  = value_t
address_(t+1) = record[address_t].next_address
```

The next address is unavailable until the current exact record returns. Therefore logical probes are serial. Skipping the currently addressed record is unsafe: two tables can be identical on every previously observed record and differ only at the current record, producing different current tokens and different next addresses.

## Multi-chain resident-cache lower bound

Construct `S` disjoint chains of length `T`. The table has:

```text
M = S * T records
```

Let a resident cache contain at most `C` complete raw records. The sum of cached records across all chains is at most `C`, so at least one chain contains at most:

```text
floor(C / S)
```

cached records. A prompt choosing that chain forces at least:

```text
host_misses >= T - floor(C / S)
```

serial host record probes.

This theorem applies to a cache of complete explicit pointer records. It does not prove the same bound for an arbitrary compressed representation carrying equivalent information.

## Record and storage equations

For `M` records and 16 signed-Q4 values:

```text
address_bits = ceil(log2(M))
record_bits  = 4 + address_bits
table_bits   = M * record_bits
cache_records = floor(cache_bits / record_bits)
```

A compact Q4-only table uses four bits/cell but needs a public or separately represented transition mechanism. The explicit pointer table charges the next-address field and therefore uses more host storage.

## Target projection

Use the Experiment 042 coefficient count:

```text
M = 56,175,137,076 cells
T = 256 tokens/chain
S = floor(M / T) = 219,434,129 chains
address bits = 36
record bits = 40
```

Explicit pointer table:

```text
total host storage = 261.5858664549887 GiB
8 GiB resident raw-record capacity = 1,717,986,918 records
floor(C / S) = 7 cached records on the worst-chain bound
minimum host misses = 249 / 256 tokens
host-miss fraction = 97.265625%
logical bytes per explicit record = 5
minimum logical host bytes = 1,245 bytes / 256 tokens
minimum logical host bytes/token = 4.86328125
```

The serial miss count is high, but the logical byte count is tiny. Neither fact establishes target latency without measuring or proving the cost of a dependent host/GPU lookup.

## Executable Gate

The workflow must:

1. build deterministic disjoint Q4 pointer chains;
2. decode exact token traces using only addressed records;
3. validate the cache pigeonhole lower bound for balanced and arbitrary sampled caches;
4. construct indistinguishable table pairs at early, middle, and late chain positions;
5. prove identical prefixes before the changed record, then different current token and next address;
6. verify logical dependency depth equals the decoded token count;
7. run a packed 64-bit host-memory pointer-chasing prototype;
8. record timing only as nonrepresentative CI evidence;
9. calculate target storage, cache capacity, serial misses, and logical bytes.

## Promotion and rejection

The Gate advances if the explicit pointer model proves serial adaptivity and the target projection forces the stated worst-chain host misses under an 8 GiB raw-record cache.

It must not claim the physical runtime target fails unless a target-relevant latency or communication lower bound is added.

Expected interpretation:

```text
explicit raw-record pointer table: nearly one serial host miss/token
logical bytes/token: small
bandwidth-only impossibility: not proven
latency impossibility: not proven
arbitrary compressed host index: not closed
```

If the prototype demonstrates that one dependent lookup is algorithmically sufficient, the next research direction is a constructive host-indexed exact-decision VM with real CPU/GPU integration and charged construction cost, not another metadata-size theorem.
