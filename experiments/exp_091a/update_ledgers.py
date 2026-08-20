from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- EXP-091A:START -->"
END = "<!-- EXP-091A:END -->"


def replace_block(path: Path, body: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"mandatory ledger missing: {path}")
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{body.rstrip()}\n{END}"
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise RuntimeError(f"malformed EXP-091A marker block in {path}")
        prefix, remainder = text.split(START, 1)
        _old, suffix = remainder.split(END, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def percent(value: float) -> str:
    if value == float("inf"):
        return "inf"
    return f"{100.0 * float(value):.9f}%"


def latest_markdown(result: dict[str, Any], result_path: Path, source_commit: str) -> str:
    selection = result["selection"]
    holdout = result["holdout_aggregate"]
    target = result["target_equation"]
    decision = result["authoritative_decision"]
    return f"""# EXP-091A latest result — Exact Jacobi Fixed-Point Gate

## Identity

- source commit: `{source_commit}`
- raw result: `{result_path.as_posix()}`
- public checkpoint: `{result['checkpoint']['model_id']}@{result['checkpoint']['requested_revision']}`
- checkpoint tensor SHA-256: `{result['checkpoint']['checkpoint_tensor_sha256']}`
- deterministic core: `{result['deterministic_core_sha256']}`
- mechanism fingerprint: `{result['mechanism_fingerprint']}`

## Authoritative decision

`{decision}`

## Build selection

- selected seed: `{selection['mode']}`
- minimum/total first-sweep exact release: `{selection['minimum_first_sweep_accepted']} / {selection['total_first_sweep_accepted']}`
- minimum/total best release over eight sweeps: `{selection['minimum_best_accepted']} / {selection['total_best_accepted']}`
- worst fixed-point sweep: `{selection['worst_fixed_point_sweep']}`

## Untouched holdout

- first-sweep exact releases: `{holdout['first_sweep_accepted']}`
- minimum/p50/maximum: `{holdout['first_sweep_min']} / {holdout['first_sweep_p50']} / {holdout['first_sweep_max']}`
- best releases over measured sweeps: `{holdout['best_accepted']}`
- fixed-point sweeps: `{holdout['fixed_point_sweeps']}`
- trajectory-before-target control: `{holdout['all_trajectories_precede_target']}`
- exact-prefix theorem controls: `{holdout['prefix_theorem_controls']}`
- fixed-point/AR controls: `{holdout['fixed_point_controls']}`

## Target equation

- required tokens per sweep, no compression: `{target['no_compression_required_tokens_per_sweep']}`
- required tokens per sweep, favorable lossless compression: `{target['favorable_required_tokens_per_sweep']}`
- observed holdout minimum first-sweep release: `{target['observed_holdout_minimum_first_sweep_accepted']}`
- observed logical fraction, no compression: `{percent(target['observed_first_sweep_fraction_no_compression'])}`
- observed logical fraction, favorable compression: `{percent(target['observed_first_sweep_fraction_favorable_compression'])}`
- two-sweep full-block fraction, no compression: `{percent(target['two_sweeps_full_block_fraction_no_compression'])}`
- target arithmetic per block sweep: `100.000000000%`

## Meaning

A self-consistent prefix is exact by causality: the guessed token and unchanged target proposal agree at every released position, and every preceding released token is already exact. The ordinary AR continuation was delayed until all trajectories were complete and served only as an independent integrity control.

A rejection means the unchanged checkpoint does not release enough exact tokens per target sweep from any frozen causal seed to approach the 405B traffic target. It does not reject trained consistency models or arbitrary new exact token certificates; retraining remains outside the fixed mission.

## Claim boundary

TARGET-W acceptance, target KV/storage behavior, 8-GiB residency, physical streamed checkpoint execution, target arithmetic throughput, and same-machine 4B-class p50/p95 remain `NOT TESTED`.
"""


def update(result_path: Path, source_commit: str) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    decision = result["authoritative_decision"]
    selection = result["selection"]
    holdout = result["holdout_aggregate"]
    target = result["target_equation"]
    gates = result["gates"]

    (ROOT / "docs/research/EXP_091A_LATEST_RESULT.md").write_text(
        latest_markdown(result, result_path, source_commit), encoding="utf-8"
    )

    state_body = f"""## EXP-091A — Exact Jacobi Fixed-Point Gate

- decision: `{decision}`
- evidence: E3 causal held-out exact-prefix Gate
- selected seed: `{selection['mode']}`
- build minimum first-sweep release: `{selection['minimum_first_sweep_accepted']} / 128`
- holdout first-sweep releases: `{holdout['first_sweep_accepted']}`
- holdout best releases over eight sweeps: `{holdout['best_accepted']}`
- holdout fixed-point sweeps: `{holdout['fixed_point_sweeps']}`
- required no-compression/favorable release: `{target['no_compression_required_tokens_per_sweep']} / {target['favorable_required_tokens_per_sweep']}`
- observed no-compression logical fraction: `{percent(target['observed_first_sweep_fraction_no_compression'])}`
- target arithmetic per sweep: `100.000000000%`
- raw evidence: `{result_path.as_posix()}`
- source commit: `{source_commit}`

Every released prefix was checked against the delayed official AR continuation, and every full fixed point, if present, was required to equal it. All Jacobi trajectories were complete before that AR control began.
"""
    replace_block(ROOT / "RESEARCH_STATE.md", state_body)

    if decision == "PROMOTE_ONE_SWEEP_JACOBI_PREFIX_TO_EXACT_BLOCK_EXECUTOR_GATE":
        next_action = """Implement the physical one-sweep streamed checkpoint executor. It must commit only the measured self-consistent prefix, preserve the official cache for that prefix, and charge lossless bytes, decompression, all 128-position arithmetic, KV traffic, RAM/SSD/PCIe/HBM movement, workspace, and p50/p95."""
    elif decision == "PROMOTE_COMPRESSED_ONE_SWEEP_JACOBI_PREFIX_TO_BLOCK_EXECUTOR_GATE":
        next_action = """First reproduce a lossless checkpoint compression ratio at least 1.261972 on the target tensor inventory and implement decompression fused with the one-sweep Jacobi block. Then measure the complete physical resource equation and exact committed cache. Without the compression result this promotion is conditional only."""
    else:
        next_action = """Do not sweep seeds, n-gram order, cycle width, block length, iteration count, prompts, tie rules, or thresholds. The next mechanism must change first-sweep exact-release information: a sound checkpoint-static token certificate, a globally shared nonseparable token solver, or another unchanged-model operator that can certify at least 85 exact tokens per checkpoint sweep."""
    next_body = f"""## Post EXP-091A handoff

Current decision: `{decision}`.

{next_action}

Later Jacobi iterations are diagnostic only: at K=128, two complete checkpoint sweeps already exceed the registered traffic fraction even under the favorable compression ratio.
"""
    replace_block(ROOT / "NEXT_EXPERIMENT.md", next_body)

    decision_body = f"""## EXP-091A decision

`{decision}`

The frozen build compiler selected `{selection['mode']}`. Untouched first-sweep exact releases were `{holdout['first_sweep_accepted']}` against required values `{target['no_compression_required_tokens_per_sweep']}` without compression and `{target['favorable_required_tokens_per_sweep']}` under the favorable lossless ratio. Best releases over eight sweeps were `{holdout['best_accepted']}`. Exact target arithmetic remained one full 128-position checkpoint evaluation per sweep.
"""
    replace_block(ROOT / "DECISION_LOG.md", decision_body)

    assumption_body = f"""## EXP-091A assumptions and grants

- `MEASURED`: DEV-W unchanged-target Jacobi proposals, self-consistent exact prefixes, fixed points, delayed AR controls, target calls, block sweeps, and processed positions.
- `DERIVED`: exact-prefix soundness by causal induction and checkpoint weight fractions from committed tokens.
- `GRANTED`: the target-scale projection treats one logical checkpoint sweep as the full weight cost and does not yet add decompression, KV, workspace, or synchronization.
- `PROJECTED`: the observed first-sweep release applied to the registered 405B checkpoint fraction equation.
- `UNVERIFIED`: TARGET-W release population, maximum-context KV handling, physical SSD/PCIe/HBM traffic, 8-GiB allocation, and target latency.
- The exact boundary token is supplied by the preceding accepted block; prompt prefill calls are measured but excluded from steady-state block amortization.
"""
    replace_block(ROOT / "ASSUMPTION_REGISTER.md", assumption_body)

    validation_body = f"""## EXP-091A validation row

| Boundary | Contract | Result | Evidence |
|---|---|---|---|
| Unchanged target Jacobi sweep | official BF16/eager block operator | `{gates['integrity_passed']}` | E3 control |
| Self-consistent prefix | exact delayed-AR equality | `{holdout['prefix_theorem_controls']}` | MEASURED |
| One-sweep release | holdout minimum >=85 / >=67 | `{holdout['first_sweep_accepted']}` | MEASURED DEV-W |
| Two or more sweeps | logical traffic <=1.185185185% | `FAIL by frozen equation` | DERIVED |
| Physical 405B block executor | full state/bytes/MACs/p50/p95 | `NOT TESTED` | — |
"""
    replace_block(ROOT / "VALIDATION_MATRIX.md", validation_body)

    architecture_body = f"""## EXP-091A architecture status

A Jacobi block may commit only the prefix where input guesses equal unchanged-target proposals consecutively from position one. The official block cache is valid for exactly that prefix. Architecture promotion without compression=`{gates['no_compression_signal']}`; conditional compressed promotion=`{gates['favorable_compression_signal']}`. Later sweeps are not admitted into the final core at K=128 because their checkpoint traffic exceeds the target.
"""
    replace_block(ROOT / "ARCHITECTURE.md", architecture_body)

    hardware_body = f"""## EXP-091A hardware status

No target GPU run was performed. The observed minimum first-sweep release gives a logical checkpoint fraction `{percent(target['observed_first_sweep_fraction_no_compression'])}` without compression. This excludes KV reads/writes, decompression, allocator/workspace, PCIe/HBM movement, and synchronization. Exact block arithmetic remains `100%` of the target graph for 128 positions. All physical target metrics remain `NOT TESTED`.
"""
    replace_block(ROOT / "HARDWARE_VALIDATION_PLAN.md", hardware_body)

    reproduce_body = f"""## EXP-091A reproduction

```bash
python -m pip install --disable-pip-version-check -r requirements/fixed-public-dynamic-executor.txt
pytest -q tests/exp_091a
python experiments/exp_091a/run_experiment.py \\
  --config experiments/exp_091a/config.json \\
  --output-dir results/exp_091a/<source-commit>
```

Checkpoint: `{result['checkpoint']['model_id']}@{result['checkpoint']['requested_revision']}`. Source commit: `{source_commit}`. Verify `checksums.sha256` before using summaries.
"""
    replace_block(ROOT / "REPRODUCIBILITY.md", reproduce_body)

    if decision.startswith("REJECT_"):
        failed_body = f"""## EXP-091A — native exact Jacobi fixed-point decoding

EXP-091A rejects the frozen unchanged-target Jacobi family as a 405B core. The selected seed was `{selection['mode']}`; untouched first-sweep exact releases were `{holdout['first_sweep_accepted']}` against the no-compression/favorable requirements `{target['no_compression_required_tokens_per_sweep']}/{target['favorable_required_tokens_per_sweep']}`. Best releases after up to eight full checkpoint sweeps were `{holdout['best_accepted']}`.

Do not reopen with seed, n-gram order, cycle width, block length, iteration count, prompt, tie-rule, shallow-draft initialization, or threshold sweeps. Training a consistency model changes the checkpoint and is outside the mission. Reopening requires a materially new exact token information source that changes first-sweep certified release.
"""
        replace_block(ROOT / "FAILED_APPROACHES.md", failed_body)
        replace_block(ROOT / "FAILED_APPROACHES_RECENT.md", failed_body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    update(args.result, args.source_commit)
