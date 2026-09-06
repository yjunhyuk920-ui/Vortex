# Handoff — source budget screen, 2026-09-06

Parent: `4a6d16487002c2756785378b0d0f5a1d73d3fca2`, PR #129, verified using the connected GitHub reader.
Branch intended: `research/source-budget-theorem-20260906`, based on the existing research branch, not main.

The fixed 405B/8-GiB/native-output-state-RNG/native4BQ4 p50<=1.2x,p95<=1.5x/TTFT mission is unchanged.
THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED; CORE_ADMISSION=false; O1-O6=OPEN.
THREE_QUALIFYING_NEW_PRINCIPLES=false. E0 analytical attempt, not an executor or target performance result.

Read `docs/REPORT_KO.md` for three candidates, exact domains, fixed-format storage/read/cache theorem,
independent-sign interval theorem and why none is a universal lower bound. No giant table or new
matrix backend was built after the cost gates. No BDD cap/order sweep, public checkpoint, GPU,
KV/RNG implementation, baseline/TTFT experiment or Actions dispatch was run.

Reproduce:

```
python src/unpack_evidence.py results/recorded.json.gz.b64 recorded_results
python src/analyze.py --out regenerated_results
python -m unittest discover -s tests -v
```

Compare the three deterministic JSON/JSONL files to `recorded_results/validation.json` hashes.
The unit-test log contains nondeterministic elapsed time; it is preserved but not byte-regenerated.
The packed file contains all 21 derived cost cases, five interval cases, both native witness paths,
validation and original test log. It is merely storage for evidence, not an inference codec.

README/RESEARCH_STATE/NEXT_EXPERIMENT/VALIDATION_MATRIX additions preserve the old bytes.
No previous local ZIP is retrospectively marked as part of this commit. The next primary obligation
remains the concrete cheap native causal source, not more variants of these negative screens.
Remote handoff status is set only by actual commit/ref/PR read-back, not by this note's existence.
