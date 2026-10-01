# Paid Krylov nonlinear feedback primitive

[Report](REPORT.md) gives the explicit source constructor, causal hot-state
update, exact native/RNG/state proof and costs. [Plan](PLAN.md) predates results.
This is a synthetic BF16/FP32 parity primitive, not an HF executor or target gain.

Reproduce into a fresh path:

    python experiments/paid_krylov_feedback_20261001/verify.py --out /tmp/vortex-krylov-replay

Python 3 and a C11 compiler are required. No model, GPU, network, paid resource,
repository-wide suite or hosted action is used. Compare numerical files using
CHECKSUMS.sha256, excluding the environment/path-dependent summary fields and
compiled executable. Original frozen numerical files are never overwritten by
replay.
