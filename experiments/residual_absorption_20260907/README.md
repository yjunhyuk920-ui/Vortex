# Exact residual absorption / live-KV history quotient

Bounded constructive research, not the arbitrary405B/8GiB/4BQ4 engine.
[Full Korean proof/report](REPORT.md), [initial registration](PREREGISTRATION.md),
[follow-on registration](FOLLOWON_PREREGISTRATION.md), [scoped ledger](LEDGER.md),
[validation](VALIDATION.json).

Restore into an empty directory from a checkout of this exact branch:

```sh
python experiments/residual_absorption_20260907/restore.py /tmp/vortex-absorption --run
```

Requires Python3.13, NumPy2.3.5, torch2.10.0+cpu and gcc. No package installation,
network/model download, GPU, credentials, Actions or repository-wide tests needed.
The 7 exact source/test/replay files are archived in CODE_CAPSULE.txt. Decode with
restore.py and verify CAPSULE_MANIFEST.json. XZ/base64 is archival transport, NOT
model/inference compression. The script reconstructs the original455 run files,
2 derived files and checks the original manifest after19 tests. A fresh restore
already passed. Scientific fixtures are synthetic, high-scale cases deliberately
favorable. The theorem's exp range is an explicit ABI assumption, not a complete
proof of NumPy's implementation. Full HF/RoPE/GPU not constructed or measured.

All actual checkpoints/traces/full raw KV/tables, prior development source and
initial failure logs are in the user ZIP. They are reproducible scientific files,
not all binary traces embedded in this Git tree. Do not claim unarchived logs can
be regenerated. The source/hash/report preserve enough to rerun scientific results.
Remote ref/PR readback is separately recorded after commit; a commit is not theory success.
