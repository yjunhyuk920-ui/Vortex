# Whole-block transport research — 2026-09-06

**No mission achievement, core admission, or GPU performance result.** This is a bounded constructive comparison of actual bias-free SwiGLU and projected-key attention transformations. Read [the Korean report](docs/REPORT_KO.md) before interpreting tests.

The executable components are an exact rational quadratic-component compiler and a fully serialized dyadic monomial program. The report also gives explicit native counterexamples and complete cost terms for these chosen paths. Three qualifying materially new core principles were **not** obtained.

```bash
python src/research.py --out results
python -m unittest discover -s tests -v
```

See `PREREGISTRATION.json`, `results/summary.json`, `results/raw.jsonl`, `results/serialized_programs.json`, `results/environment.json`, `results/validation.json`, and `HANDOFF.md`. PyTorch and NumPy versions are pinned in `requirements.txt`; actual environment is recorded separately. The whole VORTEX test suite and GitHub Actions were not run.

Raw words are also archived as `results/raw.jsonl.gz` for repository persistence. `gzip -dk results/raw.jsonl.gz` restores the byte-identical JSONL, or rerun the frozen generator. The uncompressed SHA is in SHA256SUMS.txt.
