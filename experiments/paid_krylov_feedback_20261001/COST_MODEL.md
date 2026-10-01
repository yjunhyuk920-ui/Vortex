# Conservative finite instruction bound, distinct from Python/hardware timing

The executable Python/C reference establishes correctness and logical events.
The following is a derived bound for an explicit fixed-array 64-bit word-RAM
rendering of the same constructor, not a claim that Python costs one instruction.
It closes a finite bookkeeping gap for the bounded primitive, not O5.

## Machine and services

1<=n<=64, 0<=r<=n. A word read/write, bounded integer/bit operation, shift,
index calculation, comparison, branch and 64-bit popcount each cost one unit.
A byte or uint16 access can be rendered by word loads/shifts/masks within the
per-iteration budgets below. Inputs and output buffers are explicitly allocated
in charged memory. External file service, allocator calls, code/runtime loading,
physical memory latency, and optional logging/readback have separate charged
service terms. A unit-priced popcount is a model instruction, not a latency
assertion; without it replace that primitive by its finite software cost.

Use arrays row[64], K[64], pivot_vector[64], pivot_label[64] and valid[64], plus
fixed scratch. A zero valid flag skips an unused pivot. Carry exact dependency
labels as in verify.py. Find the highest set bit by a fixed six-stage scan over
32/16/8/4/2/1-bit blocks, rather than a free arbitrary-precision bit_length.
For r=64 set mask=UINT64_MAX; never evaluate 1ULL<<64. For r=0 never access
z_(r-1). Left shifts discard bit 64, exactly as the later mask requires.

## Debit schedule

Set Q=n^2+4n. Each of the following units is implemented with at most 64 of the
word instructions above; this deliberately leaves large slack for control,
addressing, assertions, stores and setup:

- Q original BF16 validation/packing iterations, including input/output offsets
- rn packed matrix rows for constructing the r powers
- n(r+1) pivot-slot visits for at most r independent candidates and one dependence
- rn packed matrix rows for direct AK=KJ verification
- r^2 basis-column visits for its literal K*(J e_j) decoder checks
- 10r additional units cover 3r transported-observer iterations, 3r direct
  observer rechecks, column/pivot creation, decoder serialization and outer loops
- 10n units cover fixed-array initialization and dimension-bounded loop setup
- 64 constant units cover headers, masks, rank-zero handling and hot serialization

Thus an explicit sufficient internal-instruction count is

    U_build = 64[Q + 2rn + n(r+1) + r^2 + 10r + 10n + 64].

A pivot update's pair loads, two XORs and conditionals fit within its visit.
The six-stage pivot discovery fits within an independent-candidate unit. An
observer row uses bounded loads, AND, popcount, parity mask, bit placement and
comparison. Serialization uses at most eight byte-store/shift iterations per
word; include them in the 64-instruction unit. All integer operands are bounded
64-bit words; there is no unit-cost long integer or source oracle.

For the displayed hot step a conservative total is U_step<=96 word instructions,
including <=32 arithmetic/bit/select operations, static/current-state loads,
index/branch work, state and original output writes. Record load/initialization
uses <=128 instructions after memory is provided. A requested full-state readback
uses <=16r+64 instructions and exactly r basis-word reads, plus output service.
No solver, original A, K or current-input-dependent preprocessing enters U_step.

A simple data-buffer bound for this rendering is

    M_build <= 2Q + 32n + 24r + 1024 bytes,

which covers the original input buffer/header, packed rows, both pivot arrays,
validity flags, K, decoder output, hot output and fixed scratch simultaneously.
Fixed executable code, external I/O buffers and service/allocator overhead are
additional; this is not a complete Python process or GPU allocation measurement.
Hot execution can retain <=256 data bytes including its record, state, scratch
and outputs, again plus fixed code/runtime and physical allocation overhead.
Archive storage includes the unchanged original file and the serialized records.

For n=64,r=63 the conservative derived internal bound is 1,396,160 instructions
and 13,288 data bytes. These are sufficient model counts, not measured speed,
actual RSS, or an asserted 8-GiB target proof. The corresponding full T-step
bound adds actual initialization/service costs, at most 96T internal query
instructions, and each requested (16r+64)-instruction readback plus its service.
Neither a native-4B baseline ratio nor a numerical break-even horizon follows
without comparable service/operation prices and the prescribed workload.

The concrete logical counters now include the previously omitted postcheck
decoder and observer events. Their sum still is not this conservative bound:
interpreter, file and allocator costs remain separately accounted, not measured.
