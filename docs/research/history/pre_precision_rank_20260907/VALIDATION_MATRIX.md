# Validation matrix — 2026-09-07

[Report](experiments/selector_adjoint_20260907/REPORT.md), [replay](experiments/selector_adjoint_20260907/README.md), [fixed validation](experiments/selector_adjoint_20260907/results/validation.json).
[Prior matrix unchanged](docs/research/history/pre_selector_adjoint_20260907/VALIDATION_MATRIX.md).

| Item | Evidence and exact scope |
|---|---|
| Constructor/extractor | Typed coefficient/selector DAG; no selector-bearing products; finite source/serialization/forward+reverse execution |
| Proof | Reverse path coefficient induction over F2; NOT real differentiation through native rounding |
| Native scope | FP32 word->BF16 RNE plus unchanged raw32bit state; canonical NaN ABI; not a neural layer |
| Native cases |331776inputs,0output/state mismatch vs independent C;512Fraction checks; not all2^32words |
| Source work |225direct Boolean gates ->463extraction bit operations;6472B file; no latency claim |
| GF2 |8matrices/2048queries/122880bits match;128x128 file134584–134680B vs2048B bitpacked original |
| Boolean checks |12circuits/3072inputs/49152outputbits match direct evaluation and selector probes |
| Alternative gates |Unique-energy greedy trap; exact fixed-frame rank; neither is a universal no-go theorem |
| Replay |16tests,84sciencefiles; manifest b3a1f11efec95418855c985478e9dbb58ff10b7003528e0444345389d3b6c96e |
| Full mission |O1-O6OPEN;CORE_ADMISSION=false;threequalifyingnewprinciplesfalse |
| HF/fullKV/RNG/CUDA/405B/8GiB/4BQ4/TTFT/latency |NOT TESTED / NOT CONSTRUCTED |
| Actions/fullrepositorysuite |NOT RUN |

Code, preregistration, bounded proof, fixed validation and expected manifest are committed; raw binary inputs/programs/outputs regenerate and are in the user ZIP. Some detailed Korean prose/development logs are ZIP-only, not falsely described as Git-embedded. Remote persistence and scientific acceptance remain independent.
