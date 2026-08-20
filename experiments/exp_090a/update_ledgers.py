from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- EXP-090A:START -->"
END = "<!-- EXP-090A:END -->"


def replace_block(path: Path, body: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"mandatory ledger missing: {path}")
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{body.rstrip()}\n{END}"
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise RuntimeError(f"malformed EXP-090A marker block in {path}")
        prefix, remainder = text.split(START, 1)
        _old, suffix = remainder.split(END, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def percent(value: float) -> str:
    return f"{100.0 * float(value):.9f}%"


def latest_markdown(result: dict[str, Any], result_path: Path, source_commit: str) -> str:
    selection = result["selection"]
    holdout = result["holdout_aggregate"]
    target = result["target_projection"]
    decision = result["authoritative_decision"]
    return f"""# EXP-090A latest result — Causal Suffix-Action Cache Gate

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

- selected mode: `{selection['mode']}`
- build minimum accepted length: `{selection['minimum_accepted_length']} / 128`
- build total accepted length: `{selection['total_accepted_length']}`
- build true-path rank p95/max: `{selection['rank_p95']} / {selection['rank_max']}`

## Untouched holdout

- accepted lengths: `{holdout['accepted_lengths']}`
- minimum/p50/maximum: `{holdout['accepted_min']} / {holdout['accepted_p50']} / {holdout['accepted_max']}`
- target-token rank p50/p95/max: `{holdout['rank']['p50']} / {holdout['rank']['p95']} / {holdout['rank']['max']}`
- top-1/top-4/top-16 fractions: `{percent(holdout['rank']['top1_fraction'])} / {percent(holdout['rank']['top4_fraction'])} / {percent(holdout['rank']['top16_fraction'])}`
- candidate-before-target control: `{holdout['all_candidates_precede_target']}`
- manual first-layer replay equality: `{holdout['manual_first_layer_exact']}`

## Resource screen

- projected hot state: `{target['total_hot_bytes']:,}` bytes / `{target['total_hot_gib']:.9f}` GiB
- projected 8-GiB margin: `{target['margin_bytes']:,}` bytes
- hot-state screen: `{target['hot_limit_passed']}`
- actual-min-acceptance verification fraction, no compression: `{percent(target['verification_fraction_no_compression'])}`
- favorable compressed fraction: `{percent(target['verification_fraction_favorable_compression'])}`
- registered p50 fraction limit: `{percent(target['whole_model_fraction_limit'])}`
- verification arithmetic fraction: `100.000000000%`

## Meaning

The draft is causal and training-free: one official BF16 layer, a Q4 logit lens, and suffix actions from already verified prompt positions. Candidate chains were completed and hashed before the corresponding target continuation. Wrong candidates are not committed; a future exact target block verifier would retain the official successor state.

This Gate measures only whether the candidate source is strong enough to justify that verifier or a bounded candidate tree. It does not reduce the exact target arithmetic and does not establish target latency.

## Claim boundary

TARGET-W acceptance, packed Q4 execution, exact physical block verification, actual 8-GiB residency, SSD/PCIe/HBM traffic, target arithmetic throughput, and same-machine 4B-class p50/p95 remain `NOT TESTED`.
"""


def update(result_path: Path, source_commit: str) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    decision = result["authoritative_decision"]
    selection = result["selection"]
    holdout = result["holdout_aggregate"]
    target = result["target_projection"]
    gates = result["gates"]

    (ROOT / "docs/research/EXP_090A_LATEST_RESULT.md").write_text(
        latest_markdown(result, result_path, source_commit), encoding="utf-8"
    )

    state_body = f"""## EXP-090A — Causal Suffix-Action Cache Gate

- decision: `{decision}`
- evidence: E3 causal held-out drafting/source Gate
- selected mode: `{selection['mode']}`
- build minimum accepted length: `{selection['minimum_accepted_length']} / 128`
- holdout accepted lengths: `{holdout['accepted_lengths']}`
- holdout target-rank p95/max: `{holdout['rank']['p95']} / {holdout['rank']['max']}`
- projected hot state: `{target['total_hot_gib']:.9f} GiB` (`PROJECTED`)
- no-compression logical verification fraction: `{percent(target['verification_fraction_no_compression'])}`
- exact verifier arithmetic fraction: `100.000000000%`
- raw evidence: `{result_path.as_posix()}`
- source commit: `{source_commit}`

The candidate source uses one complete BF16 first layer, a checkpoint-derived row-wise Q4 head, and BF16 suffix actions from already verified prefix states. Candidate chains precede target continuation. No target output from the evaluated block enters the draft. A wrong draft would be rejected by the unchanged target verifier and cannot silently change the exact prefix-state contract.
"""
    replace_block(ROOT / "RESEARCH_STATE.md", state_body)

    if decision == "PROMOTE_CAUSAL_SUFFIX_ACTION_CACHE_TO_EXACT_BLOCK_EXECUTOR_GATE":
        next_action = """Implement the complete existing-ISA exact block verifier for the frozen 128-token chain. The verifier must stream the lossless checkpoint once, preserve official BF16/FP32 reduction semantics, return the exact accepted prefix and official KV state, and measure SSD, RAM, PCIe, GPU bytes, arithmetic, workspace, and p50/p95. The result is rejected if physical arithmetic or synchronization remains outside the 4B-class envelope."""
    elif decision == "PROMOTE_CAUSAL_SUFFIX_ACTION_CACHE_TO_CANDIDATE_TREE_GATE":
        next_action = """Freeze a bounded candidate-tree compiler using only the selected suffix-action score. Measure complete node growth, first-layer branch KV bytes, Q4-head calls, true-token coverage, exact target verification work, and accepted tokens per streamed checkpoint sweep. Do not infer a tree from true-path ranks alone; the full generated tree must fit the 8-GiB and whole-model traffic equations."""
    else:
        next_action = """Do not sweep history length, shallow depth, polynomial order, Q4 format, prompt subset, block length, nearest metric, mode order, or thresholds. The next candidate must add a materially new causal suffix representation or a sound exact token certificate. It must change the missing information source rather than tune this carry/secant/nearest family."""

    next_body = f"""## Post EXP-090A handoff

Current decision: `{decision}`.

{next_action}

The exact target arithmetic remains `100%` in EXP-090A. No result here is a 405B latency or hardware pass.
"""
    replace_block(ROOT / "NEXT_EXPERIMENT.md", next_body)

    decision_body = f"""## EXP-090A decision

`{decision}`

The compiler selected `{selection['mode']}` from the frozen build population. Untouched holdout acceptance was `{holdout['accepted_lengths']}` out of 128 and true-path target rank p95/max was `{holdout['rank']['p95']}/{holdout['rank']['max']}`. The projected hot-state screen was `{target['hot_limit_passed']}` at `{target['total_hot_gib']:.9f} GiB`; the actual-min-acceptance logical verification fraction was `{percent(target['verification_fraction_no_compression'])}`. Exact verification arithmetic was not reduced.
"""
    replace_block(ROOT / "DECISION_LOG.md", decision_body)

    assumption_body = f"""## EXP-090A assumptions and grants

- `MEASURED`: DEV-W causal draft chains, exact target chains, accepted prefixes, true-path target ranks, call counts, Q4-head simulation, and first-layer equality controls.
- `DERIVED`: accepted-length logical checkpoint fractions and the frozen target hot-state equation.
- `PROJECTED`: `{target['total_hot_gib']:.9f} GiB` for one 405B BF16 layer, Q4 head, 64 suffix actions/keys, one-layer maximum-context KV, and final norm.
- `GRANTED`: packed Q4 kernel/addressing and physical block verifier overhead are absent from the favorable screen; the remaining `{target['margin_bytes']:,}` bytes must cover them.
- `UNVERIFIED`: TARGET-W acceptance, actual 8-GiB allocation, lossless target stream, block arithmetic throughput, and 4B-class latency.
- True-path ranks after a chain mismatch are oracle diagnostics only. They do not count as accepted tokens.
"""
    replace_block(ROOT / "ASSUMPTION_REGISTER.md", assumption_body)

    validation_body = f"""## EXP-090A validation row

| Boundary | Contract | Result | Evidence |
|---|---|---|---|
| Causal one-layer suffix-action draft | candidate completed before target | `{gates['integrity_passed']}` | E3 control |
| Untouched 128-token chain | exact common prefix | `{holdout['accepted_lengths']}` | MEASURED DEV-W |
| Untouched target rank | p95/max | `{holdout['rank']['p95']} / {holdout['rank']['max']}` | favorable oracle diagnostic |
| 405B hot state | <=8 GiB | `{target['hot_limit_passed']}` (`PROJECTED`) | equation only |
| Exact physical verifier | full bytes/MACs/state/p50/p95 | `NOT TESTED` | — |
| 405B / same-machine 4B-class target | complete mission | `NOT TESTED` | — |
"""
    replace_block(ROOT / "VALIDATION_MATRIX.md", validation_body)

    architecture_body = f"""## EXP-090A architecture status

The suffix-action object is an auxiliary causal draft component unless the authoritative decision explicitly promotes it. It may never commit state independently. Exact state comes only from the accepted token prefix and the unchanged target verifier's official cache. Architecture promotion: chain=`{gates['chain_signal']}`, candidate-tree=`{gates['tree_signal']}`. Verification arithmetic remains unreduced.
"""
    replace_block(ROOT / "ARCHITECTURE.md", architecture_body)

    hardware_body = f"""## EXP-090A hardware status

No target GPU run was performed. The `{target['total_hot_gib']:.9f} GiB` value is a dimension-derived favorable projection, leaving `{target['margin_bytes']:,}` bytes before packed-kernel workspace, allocator fragmentation, on-demand embedding traffic, token buffers, and synchronization. Exact target arithmetic remains `100%`. SSD, PCIe, HBM, peak VRAM, TTFT, and p50/p95 are `NOT TESTED`.
"""
    replace_block(ROOT / "HARDWARE_VALIDATION_PLAN.md", hardware_body)

    reproduce_body = f"""## EXP-090A reproduction

```bash
python -m pip install --disable-pip-version-check -r requirements/fixed-public-dynamic-executor.txt
pytest -q tests/exp_090a
python experiments/exp_090a/run_experiment.py \\
  --config experiments/exp_090a/config.json \\
  --output-dir results/exp_090a/<source-commit>
```

Checkpoint: `{result['checkpoint']['model_id']}@{result['checkpoint']['requested_revision']}`. Source commit: `{source_commit}`. Verify `checksums.sha256` before using summaries.
"""
    replace_block(ROOT / "REPRODUCIBILITY.md", reproduce_body)

    if decision.startswith("REJECT_"):
        failed_body = f"""## EXP-090A — causal suffix-action carry/secant/nearest draft

EXP-090A rejected the frozen one-layer/Q4-logit-lens suffix-action family as a 128-token core. The selected mode was `{selection['mode']}`; untouched accepted lengths were `{holdout['accepted_lengths']}`, and true-path rank p95/max was `{holdout['rank']['p95']}/{holdout['rank']['max']}`. This result already grants exact prefill history and free packed-kernel construction.

Do not reopen with history length, shallow depth, polynomial order, Q4 range/scales, prompt selection, block length, nearest metric, mode order, or threshold sweeps. Reopening requires a materially new causal suffix representation, sound exact token certificate, or independently generated bounded tree that changes the information source. This does not reject all speculative verification or every online residual state.
"""
        replace_block(ROOT / "FAILED_APPROACHES.md", failed_body)
        replace_block(ROOT / "FAILED_APPROACHES_RECENT.md", failed_body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    update(args.result, args.source_commit)
