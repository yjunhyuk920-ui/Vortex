from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- EXP-093A:START -->"
END = "<!-- EXP-093A:END -->"


def replace_block(path: Path, body: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"mandatory ledger missing: {path}")
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{body.rstrip()}\n{END}"
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise RuntimeError(f"malformed EXP-093A marker block in {path}")
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
    bm = build["metrics"]
    hm = holdout["metrics"]
    target = result["target_projection"]
    gates = result["gates"]
    return {
        "decision": result["authoritative_decision"],
        "integrity": bool(gates["integrity_passed"]),
        "promotion": bool(gates["promotion_gate_passed"]),
        "p95_limit": int(gates["p95_true_token_rank_limit"]),
        "maximum_limit": int(gates["maximum_true_token_rank_limit"]),
        "build_p50": float(bm["p50"]),
        "build_p95": float(bm["p95"]),
        "build_max": int(bm["maximum"]),
        "build_top16": float(bm["top16_fraction"]),
        "holdout_p50": float(hm["p50"]),
        "holdout_p95": float(hm["p95"]),
        "holdout_max": int(hm["maximum"]),
        "holdout_top1": float(hm["top1_fraction"]),
        "holdout_top4": float(hm["top4_fraction"]),
        "holdout_top16": float(hm["top16_fraction"]),
        "holdout_case_maxima": list(holdout["per_case_maximum"]),
        "holdout_path_bits": list(
            holdout["per_case_static_path_information_bits"]
        ),
        "candidate_bytes": int(target["candidate_id_bytes_per_block"]),
        "raw_fraction": float(target["one_sweep_no_compression_fraction"]),
        "compressed_fraction": float(
            target["one_sweep_favorable_compression_fraction"]
        ),
    }


def latest_markdown(
    result: dict[str, Any], result_path: Path, source_commit: str
) -> str:
    d = details(result)
    return f"""# EXP-093A latest result — One-Sweep True-Token Rank Gate

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
- frozen rank thresholds: p95 `<= {d['p95_limit']}`, maximum `<= {d['maximum_limit']}`
- build rank p50/p95/max: `{d['build_p50']} / {d['build_p95']} / {d['build_max']}`
- build top-16 coverage: `{pct(d['build_top16'])}`
- untouched holdout rank p50/p95/max: `{d['holdout_p50']} / {d['holdout_p95']} / {d['holdout_max']}`
- untouched holdout top-1/top-4/top-16 coverage: `{pct(d['holdout_top1'])} / {pct(d['holdout_top4'])} / {pct(d['holdout_top16'])}`
- holdout per-case maximum ranks: `{d['holdout_case_maxima']}`
- holdout per-case oracle static-path information bits: `{d['holdout_path_bits']}`
- promotion Gate: `{d['promotion']}`

## Favorable source accounting

- one-sweep raw logical checkpoint traffic/token: `{pct(d['raw_fraction'])}`
- one-sweep favorable-compressed logical traffic/token: `{pct(d['compressed_fraction'])}`
- static top-16 token-ID bytes/block: `{d['candidate_bytes']:,}`
- one-sweep dense arithmetic/token: `100%`
- branch successor construction and verification: `NOT MEASURED`

The target continuation is generated only after the guessed-context sweep completes. It is used to score the already-fixed logits and never to generate the source.

## Meaning

A rank above 16 proves that the exact token is absent from the frozen static top-16 source at that position. No tree layout, verifier, or kernel can recover that missing token without a new information source. A pass would still be only a necessary condition because alternate branch tokens change later hidden states and logits.

## Claim boundary

This Gate does not construct branch-dependent successor states, reduce dense arithmetic, execute TARGET-W, measure 8-GiB residency, or establish 4B-class latency. Those remain `NOT TESTED`.
"""


def update(result_path: Path, source_commit: str) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    d = details(result)
    (ROOT / "docs/research/EXP_093A_LATEST_RESULT.md").write_text(
        latest_markdown(result, result_path, source_commit), encoding="utf-8"
    )

    rejected = d["decision"] == (
        "REJECT_ONE_SWEEP_STATIC_TOPK_BRANCH_SOURCE_AS_405B_CORE"
    )
    if rejected:
        next_action = (
            "Do not enlarge or retune the static candidate list. The next admissible mechanism must change the token information source: construct a causal finite-word certificate that excludes all competitors, or generate branch-dependent exact successor logits without another dense checkpoint sweep."
        )
        failure = (
            "EXP-093A rejects the frozen one-sweep static top-k logits as a bounded exact branch source. At least one required true token lies outside the registered static candidate width, so layout, verification, and kernel work cannot repair the source."
        )
    else:
        next_action = (
            "Freeze every source threshold and build an exact branch-dependent successor generator. Measure branch growth, duplicate-state merging, full prefix-state commitment, dense arithmetic, checkpoint traffic, and 128-step correctness before any target projection."
        )
        failure = (
            "EXP-093A did not reject the static source; no failed-family closure is registered until exact branch-dependent successor construction runs."
        )

    replace_block(
        ROOT / "RESEARCH_STATE.md",
        f"""## EXP-093A — one-sweep true-token rank Gate

- decision: `{d['decision']}`
- evidence: E2/E3 public-checkpoint discrete branch-source Gate
- frozen thresholds: p95 `<= {d['p95_limit']}`, maximum `<= {d['maximum_limit']}`
- holdout rank p50/p95/max: `{d['holdout_p50']} / {d['holdout_p95']} / {d['holdout_max']}`
- holdout top-1/top-4/top-16 coverage: `{pct(d['holdout_top1'])} / {pct(d['holdout_top4'])} / {pct(d['holdout_top16'])}`
- per-case holdout maxima: `{d['holdout_case_maxima']}`
- promotion Gate: `{d['promotion']}`
- raw evidence: `{result_path.as_posix()}`
- source commit: `{source_commit}`

One official guessed-context sweep completes before any target continuation. Exact target tokens are later used only to measure stable ranks in those immutable logits. The Gate is discrete and does not reuse EXP-092A's rejected linear activation span.
""",
    )
    replace_block(
        ROOT / "NEXT_EXPERIMENT.md",
        f"""## Post EXP-093A handoff

Current decision: `{d['decision']}`.

{next_action}

The one-sweep source still performs 100% of dense arithmetic per represented token. TARGET-W, physical branch execution, 8-GiB residency, and target latency remain `NOT TESTED`.
""",
    )
    replace_block(
        ROOT / "DECISION_LOG.md",
        f"""## EXP-093A decision

`{d['decision']}`

The frozen official sweep produced full-vocabulary logits before target generation. Later incremental target tokens had holdout p50/p95/max ranks `{d['holdout_p50']}/{d['holdout_p95']}/{d['holdout_max']}` against the preregistered p95/max limits `{d['p95_limit']}/{d['maximum_limit']}`. Top-16 holdout coverage was `{pct(d['holdout_top16'])}`.
""",
    )
    failed_body = f"""## EXP-093A — one-sweep static top-k branch source

{failure}

Frozen fingerprint: exact prompt prefix and boundary; `prompt_suffix_cycle_16`; one official BF16 K=128 guessed-context sweep; complete FP32 vocabulary logits; delayed official incremental target; stable descending-logit/ascending-token-ID ranks; build plus untouched holdout; p95 top-4 and worst-case top-16 thresholds.
"""
    replace_block(ROOT / "FAILED_APPROACHES.md", failed_body)
    replace_block(ROOT / "FAILED_APPROACHES_RECENT.md", failed_body)
    replace_block(
        ROOT / "ASSUMPTION_REGISTER.md",
        f"""## EXP-093A assumptions and grants

- `MEASURED`: official DEV-W guessed-sweep logits, delayed incremental target tokens/caches, exact stable ranks, ordering, and split controls.
- `DERIVED`: top-k coverage, oracle static-path information volume, `{pct(d['raw_fraction'])}` raw and `{pct(d['compressed_fraction'])}` favorable-compressed logical traffic/token.
- `GRANTED`: extraction and storage of static candidate IDs are free.
- `UNVERIFIED`: branch-dependent successor transitions, branch merging, exact verification, arithmetic reduction, TARGET-W, target VRAM/traffic/latency.
""",
    )
    replace_block(
        ROOT / "VALIDATION_MATRIX.md",
        f"""## EXP-093A validation row

| Boundary | Contract | Result | Evidence |
|---|---|---|---|
| Source causality | guessed sweep completes before target starts | `{d['integrity']}` | MEASURED |
| First exact position | proposal equals target and rank one | `{d['integrity']}` | MEASURED |
| Static branch source | holdout p95 `<= {d['p95_limit']}`, max `<= {d['maximum_limit']}` | p95 `{d['holdout_p95']}`, max `{d['holdout_max']}` | E2/E3 |
| Branch-dependent exact successors | all branches and prefix-state commitment | `NOT TESTED` | — |
| 405B / 8-GiB / 4B-class p50/p95 | complete target contract | `NOT TESTED` | — |
""",
    )
    replace_block(
        ROOT / "ARCHITECTURE.md",
        f"""## EXP-093A architecture status

The static top-k table is an oracle-evaluated candidate source, not a runtime component. It enters the architecture only after the source Gate passes and an exact branch-dependent successor generator commits the same behavioral prefix state under a complete online resource trace. Architecture promotion: `{d['promotion']}`.
""",
    )
    replace_block(
        ROOT / "HARDWARE_VALIDATION_PLAN.md",
        f"""## EXP-093A hardware status

No target hardware run was performed. The logical one-sweep traffic values `{pct(d['raw_fraction'])}` raw and `{pct(d['compressed_fraction'])}` favorable-compressed exclude physical storage, PCIe/HBM movement, branch transitions, verification, rollback, and synchronization. Dense arithmetic remains `100%`; VRAM and p50/p95 remain `NOT TESTED`.
""",
    )
    replace_block(
        ROOT / "REPRODUCIBILITY.md",
        f"""## EXP-093A reproduction

```bash
python -m pip install --disable-pip-version-check -r requirements/fixed-public-dynamic-executor.txt
pytest -q tests/exp_093a
python experiments/exp_093a/run_experiment.py \\
  --config experiments/exp_093a/config.json \\
  --output-dir results/exp_093a/<source-commit>
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
