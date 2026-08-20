from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- EXP-089A:START -->"
END = "<!-- EXP-089A:END -->"


def replace_block(path: Path, body: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"mandatory ledger missing: {path}")
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{body.rstrip()}\n{END}"
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise RuntimeError(f"malformed EXP-089A markers in {path}")
        prefix, remainder = text.split(START, 1)
        _old, suffix = remainder.split(END, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def latest_markdown(result: dict[str, Any], result_path: Path, source_commit: str) -> str:
    totals = result["totals"]
    decision = result["authoritative_decision"]
    return f"""# EXP-089A latest result — Prefix-State Bisimulation Gate

## Identity

- source commit: `{source_commit}`
- raw result: `{result_path.as_posix()}`
- public checkpoint: `{result['identity']['model_id']}@{result['identity']['requested_revision']}`
- checkpoint tensor SHA-256: `{result['identity']['checkpoint_tensor_sha256']}`
- deterministic core: `{result['deterministic_core_sha256']}`

## Authoritative decision

`{decision}`

## Measured result

- configured exact token decisions: `{totals['transitions']}`
- token mismatches: `{totals['token_mismatches']}`
- logits-byte mismatches: `{totals['logits_byte_mismatches']}`
- cache-byte/digest mismatches: `{totals['cache_digest_mismatches']}`
- RNG-state mismatches: `{totals['rng_mismatches']}`
- official incremental reference target calls: `{totals['reference_target_calls']}`
- replay target calls: `{totals['replay_target_calls']}`
- replayed token positions: `{totals['replayed_token_positions']}`
- model-free fault controls: `{result['gates']['controls']}`

## Meaning

The accepted relation, when the Gate passes, is:

```text
compiled state = exact prompt/generated token prefix + exact RNG bytes
```

Replaying that state through the pinned official checkpoint reproduces the official incremental logits and complete KV cache bytes. This establishes a checkpoint/ABI-specific behavioral state witness. It does **not** make full replay an efficient executor.

## Next boundary

A pass authorizes research on:

```text
ExactTokenDecision(checkpoint, prefix_state) -> exact token + proof/resource trace
```

The first promoted Gate is a hot-prefix early-release oracle: measure whether a fixed shallow prefix of the actual decoder can identify or tightly rank the final exact token for long blocks, while the unchanged suffix remains the exact state constructor. No replay, dense suffix, or target-output oracle may be hidden in the final selector.

TARGET-W, a cheap exact token program, 128-step performance promotion, 8-GiB residency, and 4B-class p50/p95 remain `NOT TESTED`.
"""


def update(result_path: Path, source_commit: str) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    totals = result["totals"]
    decision = result["authoritative_decision"]
    accepted = decision == "ACCEPT_PREFIX_STATE_BISIMULATION_WITNESS_FOR_TOKEN_DECISION_RESEARCH"

    latest_path = ROOT / "docs/research/EXP_089A_LATEST_RESULT.md"
    latest_path.write_text(
        latest_markdown(result, result_path, source_commit), encoding="utf-8"
    )

    state_body = f"""## EXP-089A — Prefix-State Bisimulation Gate

- decision: `{decision}`
- evidence: E2 checkpoint-specific state witness
- public checkpoint: `{result['identity']['model_id']}@{result['identity']['requested_revision']}`
- exact decisions: `{totals['transitions']}`
- token/logits/cache/RNG mismatches: `{totals['token_mismatches']}/{totals['logits_byte_mismatches']}/{totals['cache_digest_mismatches']}/{totals['rng_mismatches']}`
- official reference calls: `{totals['reference_target_calls']}`
- replay calls: `{totals['replay_target_calls']}`
- raw evidence: `{result_path.as_posix()}`
- source commit: `{source_commit}`

The state relation stores only exact prefix token IDs and exact RNG bytes. It contains no hidden/KV table, output lookup, or model response table. Replay is a correctness witness and remains prohibited as the final fast core.
"""
    replace_block(ROOT / "RESEARCH_STATE.md", state_body)

    if accepted:
        next_text = """The prefix-state behavioral witness passed. Freeze the next cheapest decisive Gate at the token-decision boundary:

```text
checkpoint-static hot prefix -> candidate token/rank trace
unchanged exact suffix       -> final token and official successor state
```

Measure, on a frozen real-checkpoint population, the earliest decoder depth whose logit-lens top-1 equals the final token, the final-token rank after a one-layer hot prefix, exact consecutive release lengths, and candidate-tree node growth for target block lengths 64 and 128. Promotion requires one causal checkpoint-static shallow decision rule whose final token is present on every held-out state and whose tree/resource equation can meet the registered 405B traffic fraction. A dense suffix remains a charged exact constructor, not a free fallback."""
    else:
        next_text = """The prefix-state witness was invalid. Fix only the first recorded byte/RNG/control mismatch without changing checkpoint, runtime ABI, population, or equality contract. Do not start token-decision compression until the behavioral state relation is executable."""
    next_body = f"""## Post EXP-089A handoff

Current decision: `{decision}`.

{next_text}

This Gate makes no latency or target-hardware claim.
"""
    replace_block(ROOT / "NEXT_EXPERIMENT.md", next_body)

    decision_body = f"""## EXP-089A decision

`{decision}`

The official incremental decoder and prefix replay were compared for `{totals['transitions']}` token decisions plus each case's terminal successor state. Raw logits, complete cache tensors, selected tokens, and RNG bytes had `{totals['logits_byte_mismatches']}/{totals['cache_digest_mismatches']}/{totals['token_mismatches']}/{totals['rng_mismatches']}` mismatches. Replay consumed `{totals['replay_target_calls']}` target calls and is not promoted as an executor.
"""
    replace_block(ROOT / "DECISION_LOG.md", decision_body)

    assumption_body = f"""## EXP-089A assumptions and scope

- `MEASURED`: pinned DEV-W official incremental and replay logits/cache/RNG/token byte equality.
- `DERIVED`: induction from initialization, output preservation, and transition preservation for the declared decoder ABI.
- `GRANTED`: none for equality; replay cost is fully counted as target calls and replayed positions.
- `UNVERIFIED`: a cheap exact token-decision program, cross-ABI equivalence, TARGET-W, 8-GiB VRAM, and target latency.
- The relation is checkpoint- and ABI-specific. Changing dtype, attention backend, reduction schedule, tokenizer, sampling algorithm, or revision requires a new witness.
"""
    replace_block(ROOT / "ASSUMPTION_REGISTER.md", assumption_body)

    validation_body = f"""## EXP-089A validation row

| Boundary | Contract | Result | Evidence |
|---|---|---|---|
| Prefix state -> official replay | raw logits + complete KV + RNG bytes | `{decision}` | E2 |
| Token decisions | `{totals['transitions']}` consecutive configured transitions | `{totals['token_mismatches']} mismatches` | MEASURED |
| Cheap exact token-decision compiler | no hidden dense replay/fallback | `NOT TESTED` | — |
| 405B / 8-GiB / 4B-class latency | complete mission | `NOT TESTED` | — |
"""
    replace_block(ROOT / "VALIDATION_MATRIX.md", validation_body)

    architecture_body = f"""## EXP-089A architecture status

Prefix token IDs plus RNG bytes are accepted as a behavioral-state witness only when `{decision}` is the authoritative result. The production executor architecture is unchanged: replay is a reference oracle, not a runtime component. The next architecture candidate must compute the exact token from this state with an explicit proof and resource trace, then append it; it may use the unchanged suffix only as a fully charged exact state constructor.
"""
    replace_block(ROOT / "ARCHITECTURE.md", architecture_body)

    hardware_body = f"""## EXP-089A hardware status

No target GPU measurement was made. DEV-W replay performed `{totals['replay_target_calls']}` official target calls and `{totals['replayed_token_positions']}` replayed token positions; these are correctness costs, not speed records. TARGET-W storage, SSD/PCIe/HBM traffic, peak VRAM, TTFT, and p50/p95 remain `NOT TESTED`.
"""
    replace_block(ROOT / "HARDWARE_VALIDATION_PLAN.md", hardware_body)

    reproduce_body = f"""## EXP-089A reproduction

```bash
python -m pip install --disable-pip-version-check -r requirements/fixed-public-dynamic-executor.txt
pytest -q tests/exp_089a
python experiments/exp_089a/run_experiment.py \\
  --config experiments/exp_089a/config.json \\
  --output-dir results/exp_089a/<source-commit>
```

Checkpoint: `{result['identity']['model_id']}@{result['identity']['requested_revision']}`. Source commit: `{source_commit}`. Verify `checksums.sha256` before using summaries.
"""
    replace_block(ROOT / "REPRODUCIBILITY.md", reproduce_body)

    if not accepted:
        failed_body = f"""## EXP-089A — invalid prefix-state witness

The frozen prefix-only state relation failed its byte/RNG controls with decision `{decision}`. This is an implementation/ABI witness failure, not a lower bound on exact token-decision programs. Reopening may correct only the first recorded contract mismatch while preserving the frozen checkpoint, runtime, population, and byte-equality obligations.
"""
        replace_block(ROOT / "FAILED_APPROACHES.md", failed_body)
        replace_block(ROOT / "FAILED_APPROACHES_RECENT.md", failed_body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    update(args.result, args.source_commit)
