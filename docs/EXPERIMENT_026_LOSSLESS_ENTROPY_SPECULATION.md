# Experiment 026 — VORTEX-ZIPTREE

## Lossless Entropy-Stationary Speculative Verification

Evidence level: **E2 sampled pretrained codec measurement / E0 405B projection**

## Candidate

The target checkpoint is never quantized or approximated. Exact FP16 weight bits
are divided into independently decodable tiles. Each tile chooses the smallest
reversible representation among:

- raw byte order plus zlib;
- byte-plane shuffle plus zlib;
- adjacent FP16-bit XOR prediction, byte-plane shuffle and zlib.

Every selected tile is decoded and byte-compared with the source before its
statistics are accepted.

A resident drafter proposes a straight token sequence or token tree. One exact
target pass streams and decompresses every target tile once, then applies it to
all verified positions as a batch. Only tokens on the accepted path count as
committed work.

## Complete pass accounting

```text
T_transfer = compressed_exact_weight_bytes / host_bandwidth
T_decode   = exact_FP16_output_bytes / decompression_output_bandwidth
T_compute  = target_FLOPs_per_position × verified_positions / tensor_throughput

T_ideal/pass = max(T_transfer, T_decode, T_compute)
T_serial/pass = T_transfer + T_decode + T_compute

T/token = T/pass / committed_tokens
```

A verification tree is explicitly charged through:

```text
verification expansion = verified_positions / committed_tokens
```

Thus a wide tree cannot appear fast merely because one path is eventually
accepted.

## Pretrained entropy measurement

The workflow loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` at FP16 and samples
unique matrix tensors across the checkpoint. Sampling uses beginning, middle and
end segments of every matrix instead of one contiguous model prefix. Each tile
reports:

- exact compressed bits/value;
- reversible transform selected;
- byte-plane zero-order entropy;
- FP16-symbol zero-order entropy;
- XOR-symbol zero-order entropy.

The measured TinyLlama bit rate is only a codec diagnostic. It is not assumed to
be the 405B bit rate.

## 405B falsification frontier

The workflow projects:

- the measured sampled bit rate;
- theoretical exact bit rates 0.125, 0.25, 0.5, 1, 2, 4 and 8 bits/weight;
- straight accepted lengths 12, 32, 64, 128, 256, 512 and 1,024;
- verification expansion factors 1, 2 and 4.

The currently observed causal Q4 candidate upper bound of 12 accepted levels is
included only as a comparison point. A real 8 GiB resident drafter remains a
separate mandatory gate.

## Promotion rule

ZIPTREE can promote only when all of the following hold:

```text
bit-exact codec round-trip passes
compressed exact target state fits the declared resident budget
serialized target pass <= 1.2 × native-4B latency
real resident drafter acceptance >= calculated minimum commit length
```

If the sampled codec saves storage but cannot approach the required bit rate or
commit length, it remains a useful systems optimization but is rejected as the
405B/8 GiB/4B-latency solution.
