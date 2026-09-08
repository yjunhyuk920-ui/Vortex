# Exact output-envelope screen

[한국어 보고서](docs/REPORT_KO.md), [사전등록](PREREGISTRATION.md),
[원시 결과](results/science/summary.json), [비용](results/costs.json).

Public original BF16 weights, synthetic inputs, a declared reference reduction.
This is not an HF operation replacement or a 405B performance result.
Python3.12, NumPy and requests are required. The exact environment is recorded
in results/science/environment_observation.json. No torch loader is used.

From this directory:

```powershell
py -3.12 -m unittest discover -s tests -v
py -3.12 portable_replay.py --output replay/from_git
```

The measured byte-for-byte replay is Windows/Python3.12.10. Linux/macOS replay
was not performed; JSON newline differences must not be called numerical errors
or silently replace the frozen baseline. The fresh output must not exist.
About18MB of pinned tensors may be acquired. Original weight files and derived
min/max .npz arrays are not committed. Checksums, input arrays, full output words,
row-packet traces, source and scientific manifests are committed. Envelope files
regenerate exactly. Acquisition/reference checks are not free inference work.

In the original local checkout, verify.py --rerun checks the39-file initial
manifest and regenerates in replay/verified. That path has been used and preserved;
choose portable_replay with a fresh path next time. Additional audit scripts alter
only the new source snapshot;38numerical/scientific result hashes remain frozen.

No HF forward, CUDA, fullKV/RNG,405B,8GiB,4BQ4,TTFT or full repository suite ran.
Coefficient payload and arithmetic counts are not physical traffic or latency.
