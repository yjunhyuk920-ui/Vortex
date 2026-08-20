from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- EXP-096A:START -->"
END = "<!-- EXP-096A:END -->"


def replace_block(path: Path, body: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"mandatory ledger missing: {path}")
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{body.rstrip()}\n{END}"
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise RuntimeError(f"malformed EXP-096A marker in {path}")
        prefix, remainder = text.split(START, 1)
        _old, suffix = remainder.split(END, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def pct(value: float) -> str:
    return f"{100.0 * float(value):.9f}%"


def details(result: dict[str, Any]) -> dict[str, Any]:
    build = result["build_aggregate"]
    holdout = result["holdout_aggregate"]
    resources = result["target_projection"]
    gates = result["gates"]
    return {
        "decision": result["authoritative_decision"],
        "integrity": bool(gates["integrity_passed"]),
        "build_fraction": float(build["certificate_fraction"]),
        "holdout_fraction": float(holdout["certificate_fraction"]),
        "build_certified": int(build["certified_positions"]),
        "holdout_certified": int(holdout["certified_positions"]),
        "build_total": int(build["total_positions"]),
        "holdout_total": int(holdout["total_positions"]),
        "build_infeasible": list(build["decisive_infeasible_cases"]),
        "holdout_infeasible": list(holdout["decisive_infeasible_cases"]),
        "build_l1_p50": build["coefficient_l1_p50"],
        "build_l1_p95": build["coefficient_l1_p95"],
        "holdout_l1_p50": holdout["coefficient_l1_p50"],
        "holdout_l1_p95": holdout["coefficient_l1_p95"],
        "holdout_linf_p50": holdout["coefficient_linf_p50"],
        "holdout_linf_p95": holdout["coefficient_linf_p95"],
        "holdout_support_p50": holdout["support_p50"],
        "holdout_support_p95": holdout["support_p95"],
        "holdout_min_margin": holdout["minimum_certified_margin"],
        "hot_bytes": int(resources["total_hot_bytes"]),
        "hot_gib": float(resources["total_hot_gib"]),
        "hot_pass": bool(resources["hot_limit_passed"]),
        "coefficient_bytes": int(resources["float64_coefficient_block_bytes"]),
        "score_work_fraction": float(resources["scalar_work_fraction_of_405b"]),
        "raw_fraction": float(resources["one_fine_sweep_fraction_at_full_block_raw"]),
        "compressed_fraction": float(resources["one_fine_sweep_fraction_at_full_block_compressed"]),
        "promoted": bool(gates["build_all_positions_certified"])
        and bool(gates["holdout_all_positions_certified"]),
    }


def latest_markdown(
    result: dict[str, Any], result_path: Path, source_commit: str
) -> str:
    d = details(result)
    return f"""# EXP-096A latest result — Online Residual Direct-Margin Certificate Gate

## Identity

- source commit: `{source_commit}`
- raw result: `{result_path.as_posix()}`
- public checkpoint: `{result['checkpoint']['model_id']}@{result['checkpoint']['requested_revision']}`
- checkpoint tensor SHA-256: `{result['checkpoint']['checkpoint_tensor_sha256']}`
- deterministic core: `{result['deterministic_core_sha256']}`
- mechanism fingerprint: `{result['mechanism_fingerprint']}`

## Authoritative decision

`{d['decision']}`

## Measured DEV-W Gate

- integrity controls: `{d['integrity']}`
- build certificates: `{d['build_certified']} / {d['build_total']}` (`{pct(d['build_fraction'])}`)
- untouched holdout certificates: `{d['holdout_certified']} / {d['holdout_total']}` (`{pct(d['holdout_fraction'])}`)
- build decisive infeasible cases: `{d['build_infeasible']}`
- holdout decisive infeasible cases: `{d['holdout_infeasible']}`
- build coefficient L1 p50/p95: `{d['build_l1_p50']} / {d['build_l1_p95']}`
- holdout coefficient L1 p50/p95: `{d['holdout_l1_p50']} / {d['holdout_l1_p95']}`
- holdout coefficient Linf p50/p95: `{d['holdout_linf_p50']} / {d['holdout_linf_p95']}`
- holdout coefficient support p50/p95: `{d['holdout_support_p50']} / {d['holdout_support_p95']}`
- minimum holdout certified FP64 margin: `{d['holdout_min_margin']}`
- promotion Gate: `{d['promoted']}`

Every accepted row was scanned against the complete vocabulary with the registered `gamma_(2K+4)` FP64 error enclosure. Every rejected row reports HiGHS status `2` on an explicit constraint subset or full matrix.

## Favorable target projection

- coefficient block: `{d['coefficient_bytes']:,}` bytes
- residual scoring and full margin scan work: `{pct(d['score_work_fraction'])}` of 405B parameter-equivalent scalar work/token
- projected total hot state: `{d['hot_bytes']:,}` bytes (`{d['hot_gib']:.6f} GiB`)
- projected 8-GiB Gate: `{d['hot_pass']}`
- perfect-block raw/compressed fine-sweep traffic/token: `{pct(d['raw_fraction'])} / {pct(d['compressed_fraction'])}`
- fine dense arithmetic/token: `100%`
- causal coefficient generator: `NOT CONSTRUCTED`

## Meaning

This is a target-seeing coefficient-capacity Gate. A pass means the online residual bank contains finite FP64 coefficient words that certify every target token. It does not make those coefficient words available before target generation.

## Claim boundary

A causal coefficient generator, target-independent certification, TARGET-W execution, physical 8-GiB allocation, fine arithmetic reduction, CUDA/SASS, and same-machine 4B-class p50/p95 remain `NOT TESTED`.
"""


def update(result_path: Path, source_commit: str) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    d = details(result)
    (ROOT / "docs/research/EXP_096A_LATEST_RESULT.md").write_text(
        latest_markdown(result, result_path, source_commit), encoding="utf-8"
    )

    if d["promoted"]:
        next_action = (
            "Freeze the finite FP64 coefficient/certificate ABI and construct a causal checkpoint-derived coefficient generator from the corrected shallow state and online bank. It must reproduce all certificate words before target continuation, then charge coefficient generation, full-vocabulary scoring, prefix-state commitment, and the still-unreduced fine sweep."
        )
        failed = (
            "EXP-096A did not reject the residual-margin capacity source. No failed-family closure is registered until the causal coefficient Gate runs."
        )
    else:
        next_action = (
            "Do not tune the linear residual-margin LP. The next admissible mechanism must add a genuinely new information dependency: branch-generated nonlinear residual rows, a checkpoint-derived higher-order coefficient program available before target continuation, or an exact symbolic transition that reduces the fine dense arithmetic."
        )
        failed = (
            "EXP-096A rejects the frozen complete online residual bank as an exact direct-margin source whenever an explicit target-token LP is infeasible. The favorable target-seeing minimum-L1 compiler therefore cannot certify every required position even before a causal coefficient generator is charged."
        )

    replace_block(
        ROOT / "RESEARCH_STATE.md",
        f"""## EXP-096A — online residual direct-margin certificate

- decision: `{d['decision']}`
- evidence: E2/E3 public-checkpoint target-seeing finite-word margin-capacity Gate
- build certificates: `{d['build_certified']} / {d['build_total']}`
- untouched holdout certificates: `{d['holdout_certified']} / {d['holdout_total']}`
- build/holdout decisive infeasible cases: `{d['build_infeasible']} / {d['holdout_infeasible']}`
- promotion: `{d['promoted']}`
- projected hot state: `{d['hot_gib']:.6f} GiB` (`PROJECTED`)
- raw evidence: `{result_path.as_posix()}`
- source commit: `{source_commit}`

One K=128 online residual bank was compiled with target-seeing minimum-L1 FP64 coefficients. Accepted rows exclude the complete vocabulary under a conservative FP64 roundoff margin. This is capacity evidence, not a causal executor.
""",
    )
    replace_block(
        ROOT / "NEXT_EXPERIMENT.md",
        f"""## Post EXP-096A handoff

Current decision: `{d['decision']}`.

{next_action}

The favorable hot ledger is `{d['hot_gib']:.6f} GiB`; fine target arithmetic remains `100%`, and target hardware remains `NOT TESTED`.
""",
    )
    replace_block(
        ROOT / "DECISION_LOG.md",
        f"""## EXP-096A decision

`{d['decision']}`

The frozen target-seeing compiler certified `{d['build_certified']}/{d['build_total']}` build positions and `{d['holdout_certified']}/{d['holdout_total']}` untouched holdout positions before cheapest-kill termination. Explicit infeasible cases were `{d['build_infeasible']}` and `{d['holdout_infeasible']}`.
""",
    )
    failed_body = f"""## EXP-096A — complete online residual direct-margin LP

{failed}

Frozen fingerprint: one K=128 fine guessed sweep; official one-layer/Q4 coarse path; complete float64 residual bank; target-seeing minimum-L1 FP64 coefficients; unique `2^-20` margin; HiGHS dual-simplex active set plus full fallback; complete-vocabulary `gamma_(2K+4)` roundoff certificate; cheapest-kill-first build and untouched holdout.
"""
    replace_block(ROOT / "FAILED_APPROACHES.md", failed_body)
    replace_block(ROOT / "FAILED_APPROACHES_RECENT.md", failed_body)
    replace_block(
        ROOT / "ASSUMPTION_REGISTER.md",
        f"""## EXP-096A assumptions and grants

- `MEASURED`: DEV-W online residual banks, delayed target tokens/coarse states, LP statuses, coefficient words and complete-vocabulary FP64 margin scans.
- `DERIVED`: minimum-L1 coefficient metrics, roundoff lower margins, `{d['hot_gib']:.6f} GiB` hot-state projection and `{pct(d['score_work_fraction'])}` score-work fraction.
- `GRANTED`: target token and true shallow state are available to the offline coefficient compiler.
- `UNVERIFIED`: causal coefficient generation, target-independent certificate, TARGET-W behavior, physical kernels, fine arithmetic reduction, target VRAM and latency.
""",
    )
    replace_block(
        ROOT / "VALIDATION_MATRIX.md",
        f"""## EXP-096A validation row

| Boundary | Contract | Result | Evidence |
|---|---|---|---|
| FP64 coefficient word | complete-vocabulary positive lower margin | build `{d['build_certified']}/{d['build_total']}` | target-seeing E2 |
| Untouched holdout capacity | every position certified | `{d['holdout_certified']}/{d['holdout_total']}` | target-seeing E3 |
| Causal coefficient generator | words available before target | `NOT TESTED` | — |
| Fine arithmetic reduction / 405B / target latency | complete target contract | `NOT TESTED` | — |
""",
    )
    replace_block(
        ROOT / "ARCHITECTURE.md",
        f"""## EXP-096A architecture status

The FP64 margin certificate is an offline target-seeing capacity object. It enters the runtime architecture only after a causal checkpoint-derived generator emits the same coefficient words before target continuation and a physical resource trace charges scoring, certification, prefix state, and fallback. Promotion: `{d['promoted']}`.
""",
    )
    replace_block(
        ROOT / "HARDWARE_VALIDATION_PLAN.md",
        f"""## EXP-096A hardware status

The favorable target ledger totals `{d['hot_bytes']:,}` bytes (`{d['hot_gib']:.6f} GiB`) including `{d['coefficient_bytes']:,}` coefficient bytes. This is `PROJECTED`; allocator, packed residual scoring, HBM/PCIe/SSD, fine sweep throughput, target VRAM and p50/p95 remain `NOT TESTED`.
""",
    )
    replace_block(
        ROOT / "REPRODUCIBILITY.md",
        f"""## EXP-096A reproduction

```bash
python -m pip install --disable-pip-version-check -r requirements/fixed-public-dynamic-executor.txt
pytest -q tests/exp_096a
python experiments/exp_096a/run_experiment.py \\
  --config experiments/exp_096a/config.json \\
  --output-dir results/exp_096a/<source-commit>
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
