# EXP-057 — Pinned Real-Checkpoint Weight-Structure Extraction Gate

This Phase C observation measures whether unchanged public TinyStories checkpoints contain the exact repeated-column or exact prototype-plus-residual structure required by the retained EXP-055/056 mechanisms.

## Pinned checkpoints

- `roneneldan/TinyStories-1M` at `77f1b168e219585646439073245fe87e56b3023e`
- `roneneldan/TinyStories-3M` at `cfaf26ec85ecdfc1bd7c2638104cce55cb67f894`
- `roneneldan/TinyStories-8M` at `8612e3b15c66ffa94eaa6ee0de5c96edd2d630af`

Every learned tensor is manifested and checksummed. Every 2-D tensor is analyzed.

## Representations

- exact loaded FP32 bit patterns: grouping observation only;
- deterministic symmetric per-output-row Q8;
- deterministic symmetric per-output-row Q4.

Q8/Q4 quantization error is recorded separately. This Gate does not claim model-output preservation after quantization.

## Run

```bash
bash experiments/exp_057/reproduce.sh
```

The artifact contains model, tensor, representation, and control rows; pinned snapshot manifests; aggregate summaries; environment metadata; and checksums. Model weight files are not copied into the artifact.

## Claim boundary

No Transformer operation is replaced. 405B, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, tokens/sec, and Q8/Q4 model-output preservation are not tested.
