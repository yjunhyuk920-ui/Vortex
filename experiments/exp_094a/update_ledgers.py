from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- EXP-094A:START -->"
END = "<!-- EXP-094A:END -->"


def replace_block(path: Path, body: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"mandatory ledger missing: {path}")
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{body.rstrip()}\n{END}"
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise RuntimeError(f"malformed EXP-094A marker block in {path}")
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
    br = build["transported_rank"]
    hr = holdout["transported_rank"]
    hs = holdout["static_fine_rank"]
    projection = result["target_projection"]
    gates = result["gates"]
    return {
        "decision": result["authoritative_decision"],
        "integrity": bool(gates["integrity_passed"]),
        "chain": bool(gates["chain_signal"]),
        "tree": bool(gates["tree_signal"]),
        "build_accepted": list(build["accepted_lengths"]),
        "holdout_accepted": list(holdout["accepted_lengths"]),
        "holdout_min": int(holdout["accepted_min"]),
        "holdout_exact_positions": list(holdout["exact_positions"]),
        "build_p95": float(br["p95"]),
        "build_max": int(br["maximum"]),
        "holdout_p50": float(hr["p50"]),
        "holdout_p95": float(hr["p95"]),
        "holdout_max": int(hr["maximum"]),
        "holdout_top1": float(hr["top1_fraction"]),
        "holdout_top4": float(hr["top4_fraction"]),
        "holdout_top16": float(hr["top16_fraction"]),
        "static_p95": float(hs["p95"]),
        "static_max": int(hs["maximum"]),
        "p95_improvement": float(holdout["rank_p95_improvement"]),
        "top16_improvement": float(holdout["top16_fraction_improvement"]),
        "hot_bytes": int(projection["total_hot_bytes"]),
        "hot_gib": float(projection["total_hot_gib"]),
        "hot_pass": bool(projection["hot_limit_passed"]),
        "residual_bytes": int(projection["float64_residual_buffer_bytes"]),
        "full_block_raw": float(
            projection["one_fine_sweep_fraction_at_full_block_raw"]
        ),
        "full_block_compressed": float(
            projection["one_fine_sweep_fraction_at_full_block_compressed"]
        ),
        "observed_compressed": float(
            projection["one_fine_sweep_fraction_at_observed_holdout_min_compressed"]
        ),
        "coarse_fraction": float(
            projection["coarse_parameter_equivalent_fraction_of_405b"]
        ),
    }


def latest_markdown(
    result: dict[str, Any], result_path: Path, source_commit: str
) -> str:
    d = details(result)
    return f"""# EXP-094A latest result — Parareal Deep-Residual Transport Gate

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
- build accepted prefixes: `{d['build_accepted']}`
- untouched holdout accepted prefixes: `{d['holdout_accepted']}`
- holdout exact positions: `{d['holdout_exact_positions']}`
- transported holdout rank p50/p95/max: `{d['holdout_p50']} / {d['holdout_p95']} / {d['holdout_max']}`
- transported holdout top-1/top-4/top-16: `{pct(d['holdout_top1'])} / {pct(d['holdout_top4'])} / {pct(d['holdout_top16'])}`
- static fine-sweep holdout rank p95/max: `{d['static_p95']} / {d['static_max']}`
- Parareal p95 rank improvement: `{d['p95_improvement']}`
- Parareal top-16 coverage improvement: `{pct(d['top16_improvement'])}`
- exact-chain Gate: `{d['chain']}`
- branch-source Gate: `{d['tree']}`

## Favorable target projection

- full-block raw/compressed fine-sweep traffic/token: `{pct(d['full_block_raw'])} / {pct(d['full_block_compressed'])}`
- observed-min compressed source traffic/token: `{pct(d['observed_compressed'])}`
- coarse parameter-equivalent arithmetic fraction: `{pct(d['coarse_fraction'])}`
- float64 residual buffer: `{d['residual_bytes']:,}` bytes
- projected total hot state: `{d['hot_bytes']:,}` bytes (`{d['hot_gib']:.6f} GiB`)
- projected 8-GiB Gate: `{d['hot_pass']}`
- fine dense arithmetic/token: `100%`
- sound token certificate: `NOT CONSTRUCTED`

## Meaning

The candidate is generated before target continuation from the fixed equation `F(g)+G(c)-G(g)`. A rank or acceptance gain therefore measures a real nonlinear causal transport effect, not a post-target fit. A pass still requires a sound certificate and physical executor.

## Claim boundary

TARGET-W acceptance, packed kernels, exact token commitment, physical VRAM, and same-machine 4B-class p50/p95 remain `NOT TESTED`.
"""


def update(result_path: Path, source_commit: str) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    d = details(result)
    (ROOT / "docs/research/EXP_094A_LATEST_RESULT.md").write_text(
        latest_markdown(result, result_path, source_commit), encoding="utf-8"
    )

    rejected = d["decision"] == (
        "REJECT_PARAREAL_RESIDUAL_TRANSPORT_AS_405B_BLOCK_SOURCE"
    )
    if rejected:
        next_action = (
            "Do not tune the one-layer Parareal transport. The next admissible mechanism must change the correction dependency: test a branch-conditional/higher-order residual with a construction that does not add another fine sweep, or a sound finite-word token certificate that can commit transported decisions while reducing fine arithmetic."
        )
        failed = (
            "EXP-094A rejects the frozen one-sweep/one-layer Parareal residual transport as a 405B block source. The online position-aligned deep residual plus corrected-prefix coarse delta did not meet either the exact-chain or bounded-rank Gate."
        )
    else:
        next_action = (
            "Freeze the residual equation and construct a sound exact token certificate. Charge score storage, competitor exclusion, fallback, full prefix-state commitment, physical kernels, and the remaining 100% fine arithmetic before target promotion."
        )
        failed = (
            "EXP-094A did not reject the Parareal source; no failed-family closure is registered until an exact certificate or branch-successor Gate runs."
        )

    replace_block(
        ROOT / "RESEARCH_STATE.md",
        f"""## EXP-094A — Parareal deep-residual transport

- decision: `{d['decision']}`
- evidence: E2/E3 public-checkpoint nonlinear causal candidate-source Gate
- build accepted prefixes: `{d['build_accepted']}`
- untouched holdout accepted prefixes: `{d['holdout_accepted']}`
- holdout transported rank p50/p95/max: `{d['holdout_p50']} / {d['holdout_p95']} / {d['holdout_max']}`
- holdout top-1/top-4/top-16: `{pct(d['holdout_top1'])} / {pct(d['holdout_top4'])} / {pct(d['holdout_top16'])}`
- static-to-transport p95 improvement: `{d['p95_improvement']}`
- exact-chain / branch-source Gates: `{d['chain']} / {d['tree']}`
- raw evidence: `{result_path.as_posix()}`
- source commit: `{source_commit}`

The fixed candidate equation is `F(g)+G(c)-G(g)`: one official target block sweep supplies the deep residual, and one official first-layer/Q4-head coarse executor transports it over the corrected prefix before any target continuation.
""",
    )
    replace_block(
        ROOT / "NEXT_EXPERIMENT.md",
        f"""## Post EXP-094A handoff

Current decision: `{d['decision']}`.

{next_action}

The favorable target projection is `{d['hot_gib']:.6f} GiB`, but fine dense arithmetic remains `100%` and physical allocation is `NOT TESTED`.
""",
    )
    replace_block(
        ROOT / "DECISION_LOG.md",
        f"""## EXP-094A decision

`{d['decision']}`

One online fine block sweep and position-aligned float64 residual were frozen before target generation. Sequential coarse transport over the corrected prefix produced holdout accepted prefixes `{d['holdout_accepted']}` and rank p50/p95/max `{d['holdout_p50']}/{d['holdout_p95']}/{d['holdout_max']}`. Static p95 was `{d['static_p95']}`; transported p95 improvement was `{d['p95_improvement']}`.
""",
    )
    failed_body = f"""## EXP-094A — one-layer Parareal residual transport

{failed}

Frozen fingerprint: one K=128 official fine guessed sweep; sequential official first layer plus rowwise-Q4 head; exact float64 `F(g)-G(g)` residual; sequential corrected-prefix `G(c)`; candidate before delayed target; build plus untouched holdout; exact-chain and p95-top4/max-top16 Gates.
"""
    replace_block(ROOT / "FAILED_APPROACHES.md", failed_body)
    replace_block(ROOT / "FAILED_APPROACHES_RECENT.md", failed_body)
    replace_block(
        ROOT / "ASSUMPTION_REGISTER.md",
        f"""## EXP-094A assumptions and grants

- `MEASURED`: DEV-W fine guessed logits, old/candidate/true coarse logits, delayed target tokens/caches, candidate acceptance, transported true-path ranks.
- `DERIVED`: float64 residual transport, logical traffic, `{d['hot_gib']:.6f} GiB` target hot-state projection, coarse parameter-equivalent fraction.
- `GRANTED`: float64 residual arithmetic and score storage are exact and free of kernel overhead.
- `UNVERIFIED`: sound token certificate, packed Q4/coarse execution, fine block throughput, TARGET-W acceptance, physical VRAM and latency.
""",
    )
    replace_block(
        ROOT / "VALIDATION_MATRIX.md",
        f"""## EXP-094A validation row

| Boundary | Contract | Result | Evidence |
|---|---|---|---|
| Causal source | candidate complete before delayed target | `{d['integrity']}` | MEASURED |
| Exact chain | every build/holdout prefix `128` | holdout `{d['holdout_accepted']}` | E2/E3 |
| Branch source | p95 `<=4`, maximum `<=16` | p95 `{d['holdout_p95']}`, max `{d['holdout_max']}` | E2/E3 |
| Sound certificate | exact token commitment without second fine sweep | `NOT TESTED` | — |
| 405B / 8-GiB / 4B-class p50/p95 | complete target contract | `NOT TESTED` | — |
""",
    )
    replace_block(
        ROOT / "ARCHITECTURE.md",
        f"""## EXP-094A architecture status

The Parareal equation is a candidate-source object, not an accepted executor. It may enter the architecture only after a surviving source result is paired with a sound token certificate or exact branch successor, complete prefix-state commitment, and a charged physical resource trace. Architecture promotion: `{d['chain'] or d['tree']}`.
""",
    )
    replace_block(
        ROOT / "HARDWARE_VALIDATION_PLAN.md",
        f"""## EXP-094A hardware status

The target hot-state equation totals `{d['hot_bytes']:,}` bytes (`{d['hot_gib']:.6f} GiB`) including a `{d['residual_bytes']:,}`-byte float64 residual block. This is `PROJECTED`; allocator fragmentation, workspace, physical Q4 packing, HBM/PCIe, synchronization, fine-sweep throughput, and p50/p95 remain `NOT TESTED`.
""",
    )
    replace_block(
        ROOT / "REPRODUCIBILITY.md",
        f"""## EXP-094A reproduction

```bash
python -m pip install --disable-pip-version-check -r requirements/fixed-public-dynamic-executor.txt
pytest -q tests/exp_094a
python experiments/exp_094a/run_experiment.py \\
  --config experiments/exp_094a/config.json \\
  --output-dir results/exp_094a/<source-commit>
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
