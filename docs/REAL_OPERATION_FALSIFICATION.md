# Real-operation falsification runner

Evidence level: **E1 real-operation falsification**

`scripts/run_real_operation_falsification.py` replaces selected `torch.nn.Linear` modules inside an actual Hugging Face causal model with `AtlasLinearModule`.

Unlike the previous hook-only trace runner, generated tokens pass through the replacement operator.

## Purpose

This runner quickly rejects the assumption that disjoint prompts remain inside a compact activation span.

It measures:

- exact greedy token equality against the same loaded model before replacement;
- hot-path vector fraction;
- rank growth on disjoint evaluation prompts;
- logical cold-weight reads;
- managed and full-model repair fractions;
- tokens per full-model-equivalent repair;
- actual elapsed time;
- CUDA peak allocation when run on CUDA.

## Command

```bash
pip install transformers

python scripts/run_real_operation_falsification.py \
  --model <local-path-or-hf-repo> \
  --device cpu \
  --max-new-tokens 16 \
  --max-rank 256
```

Default replaced suffixes:

```text
self_attn.o_proj
mlp.down_proj
```

Additional suffixes may be supplied repeatedly:

```bash
--suffix self_attn.q_proj \
--suffix self_attn.k_proj \
--suffix self_attn.v_proj \
--suffix self_attn.o_proj \
--suffix mlp.gate_proj \
--suffix mlp.up_proj \
--suffix mlp.down_proj
```

Build and evaluation prompts are disjoint by default and contain English, Korean, code, and structured-analysis tasks.

## Repair metric

For an evaluation segment:

```text
rho_full = logical exact-weight bytes read / full model weight bytes
E_full   = generated tokens / rho_full
```

The Gate 0 analytic minimum is approximately:

```text
E_full >= 491.3
```

The promotion threshold is deliberately higher:

```text
E_full >= 600
```

## Important boundary

The current Transformers model remains physically resident while the wrapper measures **logical** cold-weight use.

Therefore this runner can falsify activation-span reuse and token equality, but it does not yet prove NVMe/host/GPU traffic or the 8 GiB residency target.

A result from this runner remains E1. Physical cold-weight streaming is the next implementation only after the logical repair metric survives disjoint prompts.

## Rejection

The candidate should be rejected or redesigned when:

- committed token output diverges;
- full-model-equivalent repair efficiency remains below 300;
- rank grows to the configured maximum across disjoint prompts;
- most evaluation vectors require exact fallback;
- capsule growth cannot remain within the Gate 0 memory envelope.
