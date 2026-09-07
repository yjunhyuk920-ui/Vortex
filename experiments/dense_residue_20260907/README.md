# Dense interval–residue source (bounded auxiliary experiment)

This is NOT a 405B/8GiB executor or a complete theory. See [Korean report](docs/REPORT_KO.md), [preregistration](PREREGISTRATION.json), [fixed results](RESULTS.md) and [validation](results/validation.json).

Requires Python3.13, NumPy and a C11 compiler. Run from this folder:

```sh
python -m unittest discover -s tests -v
python run.py
python verify.py
```

`run.py` compiles the independently written C reference, creates18 actual integer/BF16 input matrices, serializes source programs, runs288 queries/19968 coordinates, and saves original values, original BF16 bytes, input vectors, produced outputs/residues, field costs and scope counterexamples. `verify.py` checks the frozen111-file manifest hash after regeneration (and additionally compares file dictionaries when the local manifest is present). The runtime Program receives only its serialized blob and current input. No original matrix or oracle residual is passed to the runtime.

The compact syndrome decoder can correct dense output errors, but the implemented source reads every lossless residual bitplane. The L1-based modulus does not discard original coefficient information. Generic dense128 source cost is51.60–51.70% of BF16 payload and is larger than a bitpacked representation of the actual original integer weights. No latency speedup is claimed. Original-domain restrictions and the zero-sign guard are explicit and are not changes to the mission.

`results/reference.so` is a locally compiled reference, regenerated rather than a portable binary. It is not included in the scientific manifest. Raw fixtures and traces are included in the user ZIP; the source and expected manifest suffice to regenerate them. Logs are not promised bit-identical. No upstream checkpoint weights or secrets are included.
