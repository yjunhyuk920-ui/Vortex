# Experiment 044 — Host-Indexed Exact-Decision VM

Last updated: 2026-08-03 (Asia/Seoul)

## Evidence level and purpose

This is an E1/E2 constructive CPU prototype Gate. It does not run a released 405B checkpoint, does not integrate with a GPU, and does not treat CI timing as target hardware evidence.

Experiment 043 proved that an explicit pointer representation can require nearly one serial host record probe per generated token, but only about 4.86 logical bytes/token. The probe count did not close the runtime target. Experiment 044 therefore builds the host-indexed path instead of adding another abstract impossibility argument.

## VM record semantics

Every exact record contains:

```text
q4_value:    4 bits
next_address: exact record index or terminal sentinel
```

Execution is:

```text
record_t = vm.read(address_t)
token_t  = record_t.q4_value
address_(t+1) = record_t.next_address
```

The VM must reproduce the original explicit pointer-table trace exactly.

## Binary format v1

The file begins with a fixed 64-byte little-endian header:

```text
magic
version
header bytes
record bytes
format flags
record count
start count
chain steps
records offset
starts offset
payload CRC32
header CRC32
reserved
```

Two record encodings share identical semantics:

### compact40

```text
5 bytes/record
low 4 bits: Q4 code
high 36 bits: next_address + 1
zero pointer code: terminal
```

This supports up to `2^36 - 1` addressed records and directly matches the Experiment 043 36-bit target address envelope.

### aligned64

```text
8 bytes/record
low 4 bits: Q4 code
high 60 bits: next_address + 1
zero pointer code: terminal
```

The aligned format spends more host storage but may decode faster on ordinary CPUs.

Chain starts are stored as little-endian unsigned 64-bit addresses after the records.

## Builder requirements

The builder must:

1. validate every Q4 value and next pointer;
2. write a unique temporary file in the destination directory;
3. stream records while computing payload CRC32;
4. append starts and include them in the payload checksum;
5. write the final header with its own checksum;
6. `fsync` the file;
7. atomically replace the destination;
8. attempt to `fsync` the parent directory;
9. remove temporary files after failure.

Build time, final bytes, format, checksums, and counts are evidence.

## Reader requirements

The mmap reader must reject:

```text
bad magic
unsupported version
wrong header or record size
invalid offsets
truncation or trailing bytes
bad header checksum
bad payload checksum
out-of-range chain start
out-of-range next pointer
invalid Q4 code
```

Exact checksum verification may be disabled only for benchmark decomposition and must be reported.

## Resident cache

A bounded LRU cache stores decoded records by address. Evidence must separate:

```text
logical probes
mmap record reads
cache hits
cache misses
start-table reads
```

A repeated chain with enough cache capacity must replay identically while moving from all misses to all hits.

## Benchmark matrix

For both compact40 and aligned64:

1. build the same deterministic table;
2. verify checksum and exact replay;
3. measure sequential record reads;
4. measure shuffled random reads;
5. measure dependent pointer chasing;
6. measure repeated replay with an empty and then warm cache;
7. close and reopen the mmap and measure again;
8. report p50, p95, p99, mean, minimum, maximum, operations/second, build time, and file bytes.

The OS page cache cannot be reliably dropped in GitHub Actions. Reopen measurements are not called physically cold. The certificate records:

```text
OS cache state controlled: false
CI timing target representative: false
```

## Target storage projection

Using:

```text
records: 56,175,137,076
starts: floor(records / 256) = 219,434,129
```

Projected files are:

```text
compact40 records: 261.5858664549887 GiB
aligned64 records: 418.53738632798195 GiB
start table: 1.6349116638302803 GiB
compact40 total: 263.220778118819 GiB
aligned64 total: 420.17229799181223 GiB
```

This is host/disk storage, not VRAM. Timing is not projected from CI to these file sizes.

## Promotion criteria

The VM advances toward CPU/GPU integration only if:

- both formats build atomically and pass corruption tests;
- exact traces match the source table;
- cache accounting is exact;
- benchmark JSON is complete and finite;
- compact and aligned storage equations are correct;
- timing is explicitly marked nonrepresentative;
- no conclusion substitutes mmap lookup for a real 405B decision compiler.

## Strict scope

A successful VM proves that the Experiment 043 host-indexed representation is concretely executable with mmap and bounded caching. It does not prove that arbitrary released model decisions can be compiled into this table, that the table can be built cheaply, or that CPU/GPU lookup fits the final latency target.

The next step after a successful CPU VM is pinned-memory or GPU-facing integration plus a real decision-index compiler Gate. Both representation construction and inference must be charged.
