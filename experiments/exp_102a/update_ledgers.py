from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- EXP-102A:START -->"
END = "<!-- EXP-102A:END -->"


def replace_block(path: Path, body: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"mandatory canonical file missing: {path}")
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{body.rstrip()}\n{END}"
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise RuntimeError(f"malformed EXP-102A marker in {path}")
        prefix, remainder = text.split(START, 1)
        _, suffix = remainder.split(END, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def arm_line(name: str, arm: dict[str, Any]) -> str:
    gate = arm["holdout_gate"]
    return (
        f"- `{name}`: K `{arm['selected_block_length_from_build_only']}`, "
        f"minimum A `{gate['minimum_committed_tokens']}`, "
        f"latency p50/p95 `{gate['latency_ratio_p50']:.6f} / {gate['latency_ratio_p95']:.6f}`, "
        f"N/A p95 `{gate['candidate_ratio_p95']:.6f}`, "
        f"exact token+state `{gate['exact_token_and_state']}`, pass `{gate['gate_passed']}`"
    )


def update(result_path: Path, source_commit: str) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    decision = result["authoritative_decision"]
    arms = result["arms"]
    lines = [arm_line(name, arm) for name, arm in sorted(arms.items())]
    requirements = result["requirements"]

    latest = f"""# EXP-102A latest result — Reality-First Causal Draft/Verify Gate

- source commit: `{source_commit}`
- raw result: `{result_path.as_posix()}`
- decision: `{decision}`
- deterministic core: `{result['deterministic_core_sha256']}`
- authoritative arm: `{result['authoritative_arm']}`
- forbidden grants: `{result['forbidden_grants']}`
- integrity failures: `{result['integrity_failures']}`
- raw p50/p95 minimum committed tokens: `{requirements['raw_p50']['minimum_committed_tokens']} / {requirements['raw_p95']['minimum_committed_tokens']}`

## Untouched holdout

{chr(10).join(lines)}

## Claim boundary

This is an executable E2/E3 CPU Gate with real public draft and target checkpoints. It charges draft generation, target candidate positions, mismatch repair, draft-state rebuild, N/A, cache bytes, and wall time. Target 405B execution, physical 8-GiB residency, target storage/H2D, CUDA/SASS, and same-machine native-4B-Q4 acceptance remain `NOT TESTED`.
"""
    latest_path = ROOT / "docs/research/EXP_102A_LATEST_RESULT.md"
    latest_path.write_text(latest, encoding="utf-8")

    if decision == "PROMOTE_REAL_CAUSAL_DRAFT_BLOCK_TO_LARGER_SCALE_FULLY_CHARGED_GATE":
        next_action = (
            "Promote only to a larger draft/target size-ratio rung and then the measured target-hardware rung. "
            "Keep every causal, state, N/A, byte, memory, and wall-time cost. Do not reintroduce a free FMM block."
        )
        failure_text = "EXP-102A did not reject the real causal block source."
    elif decision == "RETAIN_SAME_FAMILY_CAUSAL_DRAFT_AS_RESTRICTED_AUXILIARY_NOT_UNIVERSAL":
        next_action = (
            "Retain the same-family source only as a restricted auxiliary. The next core Gate must supply a real "
            "cross-family/arbitrary-checkpoint causal source or a different executable dependency; same-family "
            "retuning cannot establish the fixed universal mission."
        )
        failure_text = (
            "The universal text-interface arm failed while the same-family arm survived. Do not promote the "
            "restricted arm as arbitrary-checkpoint evidence."
        )
    elif decision == "REJECT_FROZEN_REAL_CAUSAL_EXTERNAL_DRAFTS_AS_85_TOKEN_AMORTIZATION_SOURCE":
        next_action = (
            "Close the frozen greedy external-draft block source. Do not sweep K, prompts, sampling, model revisions, "
            "or thresholds. The next core principle must change the causal information source or execution dependency "
            "and remain fully executable and charged."
        )
        failure_text = (
            "The frozen real causal draft/verify source failed the raw A, N/A, latency, or exact-state Gate even before "
            "405B scaling. Reopening requires a materially different causal source."
        )
    else:
        next_action = (
            "Repair only the listed integrity or infrastructure failure without changing models, prompts, K, "
            "thresholds, or equations. Do not interpret an invalid run scientifically."
        )
        failure_text = "No scientific rejection is recorded because the run was invalid."

    state_block = f"""## EXP-102A — Reality-First causal block

- decision: `{decision}`
- source: `{source_commit}`
- result: `{result_path.as_posix()}`
- deterministic core: `{result['deterministic_core_sha256']}`
- integrity failures: `{result['integrity_failures']}`

{chr(10).join(lines)}

The result uses no future target tokens, perfect selector, free repair, free fallback, free N/A, or unmeasured compression in the authoritative arm.
"""
    replace_block(ROOT / "RESEARCH_STATE.md", state_block)

    next_block = f"""## EXP-102A handoff

Current decision: `{decision}`.

{next_action}

Every next core Gate remains subject to `docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md`.
"""
    replace_block(ROOT / "NEXT_EXPERIMENT.md", next_block)

    decision_block = f"""## EXP-102A decision

`{decision}`

The authoritative arm executed real causal draft prefill/generation, tokenizer bridge, real target block verification, mismatch repair, real draft-cache crop/replay or rebuild, exact terminal target KV comparison, and same-run latency/N/A accounting. No impossible promotion grant was used.
"""
    replace_block(ROOT / "DECISION_LOG.md", decision_block)

    failed_block = f"""## EXP-102A — real causal greedy draft block

{failure_text}

Frozen scope: pinned SmolLM2-360M target; same-family SmolLM2-135M and cross-family TinyStories-33M drafts; K `64,96`; build-only K selection; raw no-compression traffic threshold; exact terminal KV; all online work charged.
"""
    replace_block(ROOT / "FAILED_APPROACHES.md", failed_block)
    replace_block(ROOT / "FAILED_APPROACHES_RECENT.md", failed_block)

    assumption_block = f"""## EXP-102A reality assumptions

- `MEASURED`: public-checkpoint CPU wall time, accepted/committed tokens, target positions, N/A, cache bytes, RSS, token equality, terminal KV equality.
- `DERIVED`: raw one-sweep p50/p95 minimum A = `{requirements['raw_p50']['minimum_committed_tokens']} / {requirements['raw_p95']['minimum_committed_tokens']}`.
- `PROJECTED`: none used for promotion.
- `UNVERIFIED`: 405B scaling, target SSD/H2D, target GPU kernels, physical 8-GiB plan, native 4B-Q4 p50/p95.
- No compression ratio, peak utilization, perfect selector, future block, or zero-cost component is granted.
"""
    replace_block(ROOT / "ASSUMPTION_REGISTER.md", assumption_block)

    validation_block = f"""## EXP-102A validation row

| Boundary | Contract | Result | Evidence |
|---|---|---|---|
| Causal candidate | no target continuation observed | `{decision}` | E2/E3 CPU |
| Exact commitment | tokens + complete terminal target KV | see raw result | E2 |
| Runtime cost | draft + bridge + verify + repair + rebuild + N/A + RSS | measured | E2 |
| 405B / 8 GiB / same-machine 4B | complete target contract | `NOT TESTED` | — |
"""
    replace_block(ROOT / "VALIDATION_MATRIX.md", validation_block)

    architecture_block = f"""## EXP-102A architecture status

The only evaluated component is a real causal draft/verify loop. Architecture promotion follows `{decision}`. No multiplication oracle, free future activation block, or unimplemented transform circuit is an accepted runtime component.
"""
    replace_block(ROOT / "ARCHITECTURE.md", architecture_block)

    hardware_block = f"""## EXP-102A hardware status

The run is CPU E2/E3 evidence. Quadro M5000 execution, target storage, H2D, CUDA/SASS, physical peak VRAM, power, thermal state, 405B, and same-machine native 4B-Q4 latency remain `NOT TESTED`.
"""
    replace_block(ROOT / "HARDWARE_VALIDATION_PLAN.md", hardware_block)

    reproduce_block = f"""## EXP-102A reproduction

```bash
python -m pip install --disable-pip-version-check -r requirements/fixed-public-dynamic-executor.txt
python -m pytest -q tests/exp_102a
python experiments/exp_102a/run_experiment.py \\
  --config experiments/exp_102a/config.json \\
  --output-dir results/exp_102a/<source-commit>
```

Verify `checksums.sha256` before using the processed result. Source commit: `{source_commit}`.
"""
    replace_block(ROOT / "REPRODUCIBILITY.md", reproduce_block)

    handoff_block = f"""## EXP-102A current handoff

- decision: `{decision}`
- result: `{result_path.as_posix()}`
- source: `{source_commit}`
- next: {next_action}
- reality contract: `docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md`
"""
    replace_block(ROOT / "docs/research/VORTEX_RESEARCH_HANDOFF.md", handoff_block)

    readme_block = f"""## EXP-102A reality-first result

Authoritative decision:

```text
{decision}
```

The Gate used real public draft and target checkpoints and charged draft generation, bridge, target candidate positions, N/A, mismatch repair, draft-state rebuild, cache bytes, RSS, and same-run wall time. No compression or impossible oracle credit was used. See `docs/research/EXP_102A_LATEST_RESULT.md`.

`README_CURRENT=true`  
`README_UPDATED=latest authoritative Gate, active handoff, reality-first contract`
"""
    replace_block(ROOT / "README.md", readme_block)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    update(args.result, args.source_commit)
