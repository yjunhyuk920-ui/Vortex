from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- EXP-092A:START -->"
END = "<!-- EXP-092A:END -->"


def replace_block(path: Path, body: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"mandatory ledger missing: {path}")
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{body.rstrip()}\n{END}"
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise RuntimeError(f"malformed EXP-092A marker block in {path}")
        prefix, remainder = text.split(START, 1)
        _old, suffix = remainder.split(END, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def pct(value: float) -> str:
    return f"{100.0 * float(value):.9f}%"


def details(result: dict[str, Any]) -> dict[str, Any]:
    h = result["holdout_aggregate"]
    p = result["target_projection"]
    g = result["gates"]
    return {
        "decision": result["authoritative_decision"],
        "integrity": bool(g["integrity_passed"]),
        "threshold": int(g["maximum_extra_directions_per_operator"]),
        "p50": float(h["extra_rank_p50"]),
        "p95": float(h["extra_rank_p95"]),
        "maximum": int(h["extra_rank_max"]),
        "fraction_p50": float(h["extra_fraction_p50"]),
        "fraction_p95": float(h["extra_fraction_p95"]),
        "over": int(h["over_threshold_count"]),
        "reports": int(h["report_count"]),
        "multiplier": float(p["observed_operation_multiplier_lower_bound"]),
        "sidecar": int(p["observed_sidecar_bytes_lower_bound"]),
        "threshold_sidecar": int(p["threshold_sidecar_bytes"]),
    }


def latest_markdown(
    result: dict[str, Any], result_path: Path, source_commit: str
) -> str:
    d = details(result)
    return f"""# EXP-092A latest result — One-Sweep Block-Span Correction Closure Gate

## Identity

- source commit: `{source_commit}`
- raw result: `{result_path.as_posix()}`
- public checkpoint: `{result['checkpoint']['model_id']}@{result['checkpoint']['requested_revision']}`
- checkpoint tensor SHA-256: `{result['checkpoint']['checkpoint_tensor_sha256']}`
- deterministic core: `{result['config_sha256'] if 'deterministic_core_sha256' not in result else result['deterministic_core_sha256']}`
- mechanism fingerprint: `{result['mechanism_fingerprint']}`

## Authoritative decision

`{d['decision']}`

## Measured DEV-W Gate

- integrity controls: `{d['integrity']}`
- frozen extra-direction budget per operator: `{d['threshold']}`
- untouched holdout reports: `{d['reports']}`
- holdout certified extra-rank lower bound p50/p95/max: `{d['p50']} / {d['p95']} / {d['maximum']}`
- holdout certified extra-fraction lower bound p50/p95: `{pct(d['fraction_p50'])} / {pct(d['fraction_p95'])}`
- holdout reports already over the frozen budget: `{d['over']} / {d['reports']}`
- optimistic correction-operation multiplier lower bound: `{d['multiplier']:.9f}x`

## Favorable target projection

- sidecar at frozen eight-direction budget: `{d['threshold_sidecar']:,}` bytes
- sidecar lower bound at the observed maximum certified requirement: `{d['sidecar']:,}` bytes

The projection grants true target blocks, exact coefficients, basis discovery, nonlinear recomputation, routing, coefficient work, and native-order repair for free. It is `DERIVED`, not a TARGET-W measurement.

## Meaning

The first guessed block supplies 128 arbitrary projection-input basis rows. The Gate asks whether the exact teacher-forced block remains inside that span plus only eight static directions. A modular rank increase on a frozen 192-coordinate restriction is a certified lower bound on the full rational rank increase. Therefore a lower bound above eight cannot be repaired by a full-rank solver or kernel implementation.

## Claim boundary

This result covers first/middle/last complete DEV-W layers and four dominant linear-operator input roles. Full-coordinate ranks when the lower-bound Gate survives, a causal coefficient constructor, all TARGET-W layers, 8-GiB residency, physical traffic, block arithmetic, and same-machine 4B-class latency remain `NOT TESTED`.
"""


def update(result_path: Path, source_commit: str) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    d = details(result)
    latest = latest_markdown(result, result_path, source_commit)
    (ROOT / "docs/research/EXP_092A_LATEST_RESULT.md").write_text(
        latest, encoding="utf-8"
    )

    rejected = d["decision"] == "REJECT_ONE_SWEEP_BLOCK_SPAN_CORRECTION_AS_405B_CORE"
    if rejected:
        next_action = (
            "Do not enlarge or retune a linear correction basis. The next admissible mechanism must propagate token corrections through a nonlinear finite-word symbolic program, exact branch closure, or sound token certificate whose information is not a small linear extension of the guessed-block activation span."
        )
        failure = (
            "EXP-092A rejects one-sweep linear block-span correction under the frozen scope. Even an oracle true block and free coefficients require more than eight new directions on untouched operator inputs."
        )
    else:
        next_action = (
            "Compute full-coordinate exact ranks, synthesize one build-derived static basis, freeze it before holdout, construct causal coefficients from corrected activations, and verify native projection bytes on all DEV-W layers."
        )
        failure = (
            "EXP-092A did not reject the lower-bound Gate; no failed-family closure is registered until the full exact coefficient Gate runs."
        )

    replace_block(
        ROOT / "RESEARCH_STATE.md",
        f"""## EXP-092A — one-sweep block-span correction closure

- decision: `{d['decision']}`
- evidence: E2/E3 public-checkpoint oracle rank Gate
- holdout certified extra-rank lower bound p50/p95/max: `{d['p50']} / {d['p95']} / {d['maximum']}`
- frozen budget: `{d['threshold']}` directions per operator
- over-budget holdout reports: `{d['over']} / {d['reports']}`
- correction-operation multiplier lower bound: `{d['multiplier']:.9f}x`
- raw evidence: `{result_path.as_posix()}`
- source commit: `{source_commit}`

Guessed captures precede target generation. The exact AR target and teacher-forced block are oracle controls only. Rank is computed on exact BF16 dyadic integers over three primes after a frozen coordinate restriction; an over-budget lower bound is decisive before coefficient or kernel work.
""",
    )
    replace_block(
        ROOT / "NEXT_EXPERIMENT.md",
        f"""## Post EXP-092A handoff

Current decision: `{d['decision']}`.

{next_action}

Do not treat the oracle true block, coefficients, or basis discovery as deployable inputs. TARGET-W and physical latency remain `NOT TESTED`.
""",
    )
    replace_block(
        ROOT / "DECISION_LOG.md",
        f"""## EXP-092A decision

`{d['decision']}`

The frozen Gate granted the complete 128-row guessed input block as an arbitrary projection basis, then measured the certified rank increment required by the exact teacher-forced block on first/middle/last complete layers. Holdout p50/p95/max lower bounds were `{d['p50']}/{d['p95']}/{d['maximum']}` against a budget of `{d['threshold']}`.
""",
    )
    failed_body = f"""## EXP-092A — one-sweep linear block-span correction

{failure}

Frozen fingerprint: K=128 EXP-091A selected seed; official guessed block before target; exact teacher-forced oracle; layers 0/15/29; qkv/o/gate-up/down input roles; exact dyadic integer encoding; deterministic 192-coordinate restriction; three modular rank lower bounds; eight extra static directions granted.
"""
    replace_block(ROOT / "FAILED_APPROACHES.md", failed_body)
    replace_block(ROOT / "FAILED_APPROACHES_RECENT.md", failed_body)
    replace_block(
        ROOT / "ASSUMPTION_REGISTER.md",
        f"""## EXP-092A assumptions and grants

- `MEASURED`: DEV-W guessed/true operator-input blocks, delayed AR equality, exact modular ranks, run ordering, and population controls.
- `DERIVED`: full rational rank-increment lower bounds from restricted-coordinate modular ranks; `{d['multiplier']:.9f}x` optimistic correction multiplier lower bound.
- `GRANTED`: true target block, exact coefficients, basis discovery, routing, nonlinear recomputation, coefficient application, and native-order repair.
- `UNVERIFIED`: a causal nonlinear correction source, full-coordinate ranks if the Gate survives, all layers, TARGET-W, target traffic/VRAM/latency.
""",
    )
    replace_block(
        ROOT / "VALIDATION_MATRIX.md",
        f"""## EXP-092A validation row

| Boundary | Contract | Result | Evidence |
|---|---|---|---|
| Guessed block ordering | capture completes before AR target | `{d['integrity']}` | MEASURED |
| Teacher-forced block | 128-token delayed AR equality | `{d['integrity']}` | MEASURED |
| Block-span correction | every holdout rank increment <= `{d['threshold']}` | p95 `{d['p95']}`, max `{d['maximum']}` | E2/E3 lower-bound Gate |
| Causal coefficients/native equality | all layers, no target oracle | `NOT TESTED` | — |
| 405B / 8-GiB / 4B-class p50/p95 | complete target contract | `NOT TESTED` | — |
""",
    )
    replace_block(
        ROOT / "ARCHITECTURE.md",
        f"""## EXP-092A architecture status

The block-span object is an oracle correction feasibility test, not a runtime component. It may enter the architecture only after a surviving lower-bound result is followed by a build-frozen basis, causal exact coefficient solver, all-layer native-byte equality, and complete online resource trace. Architecture promotion: `{not rejected}`.
""",
    )
    replace_block(
        ROOT / "HARDWARE_VALIDATION_PLAN.md",
        f"""## EXP-092A hardware status

No target GPU run was performed. `{d['threshold_sidecar']:,}` bytes at the eight-direction budget and `{d['sidecar']:,}` bytes at the observed certified requirement are dimension-derived sidecar values only. SSD, PCIe, HBM, CUDA/SASS, workspace, peak VRAM, TTFT, and p50/p95 remain `NOT TESTED`.
""",
    )
    replace_block(
        ROOT / "REPRODUCIBILITY.md",
        f"""## EXP-092A reproduction

```bash
python -m pip install --disable-pip-version-check -r requirements/fixed-public-dynamic-executor.txt
pytest -q tests/exp_092a
python experiments/exp_092a/run_experiment.py \\
  --config experiments/exp_092a/config.json \\
  --output-dir results/exp_092a/<source-commit>
```

Checkpoint: `{result['checkpoint']['model_id']}@{result['checkpoint']['requested_revision']}`. Source commit: `{source_commit}`. Verify `checksums.sha256` before using summaries.
""",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    update(args.result, args.source_commit)
