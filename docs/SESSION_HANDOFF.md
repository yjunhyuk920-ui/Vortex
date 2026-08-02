# Session handoff

Last updated: 2026-08-02 (Asia/Seoul)

## Current verified result

The repository now contains two execution experiments:

1. disk-backed progressive LM-head certification;
2. a base-free internal projection fast path named `OnlineAtlasLinear`.

Validation command:

```bash
python -m pytest -q
python scripts/run_validation.py
```

Observed result:

```text
10 passed
validation script completed successfully
```

## New milestone completed

`OnlineAtlasLinear` caches an input basis `U` and exact operator image `WU`. Inputs inside the learned span execute without loading the original weight. Inputs outside the span use exact cold fallback and expand the atlas.

Validated results:

- synthetic rank-8 trace: 93.75% fast path after eight cold reads;
- tiny Llama O/down projection atlas persistence;
- fresh-runtime replay with identical generated tokens;
- zero cold reads for the managed internal projections on replay;
- 84,480-byte persisted capsule for the validated tiny trace.

See `docs/ATLAS_FAST_PATH_2026-08-02.md` and `validation_results.json`.

## Current code boundary

```text
HF safetensors -> exact cold linear operator
                    | miss
                    v
              learn U and WU
                    |
                    v
         persistent AtlasLinear capsule
                    |
                    v
       base-free exact-on-span fast path
```

Integrated projection suffixes in the validation run:

```text
self_attn.o_proj.weight
mlp.down_proj.weight
```

## Exact next task

Do not return to the full-base progressive design as the default internal path.

1. Add a real-model trace runner for a small pretrained Llama-family checkpoint.
2. Capture per-layer O/down inputs during prompt prefill and continuation.
3. Build atlases from the first trace segment.
4. Test continued generation and neighboring prompts without rebuilding.
5. Record rank curves, cold streams/token, hit rate, capsule bytes, token equality, and wall-clock.
6. Add Q/K/V and gate/up only after O/down pass the real-trace falsification gate.
7. Add an FP16/BF16 capsule mode and compare numerical/token behavior.

## Decisive unknown

The project now has a real base-free path. The next measurement determines whether real-model activation reachable sets remain low-rank enough across new tokens and prompts for cold reads to become rare.
