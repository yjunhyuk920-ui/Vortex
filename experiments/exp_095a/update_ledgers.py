from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- EXP-095A:START -->"
END = "<!-- EXP-095A:END -->"


def replace_block(path: Path, body: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"mandatory ledger missing: {path}")
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{body.rstrip()}\n{END}"
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise RuntimeError(f"malformed EXP-095A marker block in {path}")
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
    projection = result["target_projection"]
    hot = projection["hot"]
    gates = result["gates"]
    return {
        "decision": result["authoritative_decision"],
        "integrity": bool(gates["integrity_passed"]),
        "exact_chain": bool(gates["exact_chain_signal"]),
        "causal_branch": bool(gates["causal_branch_signal"]),
        "oracle_span": bool(gates["oracle_span_signal"]),
        "build_accepted": list(build["accepted_lengths"]),
        "holdout_accepted": list(holdout["accepted_lengths"]),
        "holdout_exact_positions": list(holdout["exact_positions"]),
        "causal": holdout["causal_rank"],
        "aligned": holdout["position_aligned_rank"],
        "single": holdout["single_residual_oracle_rank"],
        "span": holdout["span_oracle_rank"],
        "build_span": build["span_oracle_rank"],
        "span_ranks": list(holdout["span_effective_ranks"]),
        "span_l2": list(holdout["span_projection_relative_l2"]),
        "hot_bytes": int(hot["total_hot_bytes"]),
        "hot_gib": float(hot["total_hot_gib"]),
        "hot_pass": bool(hot["hot_limit_passed"]),
        "hidden_bank_bytes": int(hot["float64_hidden_bank_bytes"]),
        "gram_bytes": int(hot["float64_gram_bytes"]),
        "full_raw": float(projection["one_fine_sweep_fraction_at_full_block_raw"]),
        "full_compressed": float(
            projection["one_fine_sweep_fraction_at_full_block_compressed"]
        ),
        "observed_compressed": float(
            projection["one_fine_sweep_fraction_at_observed_holdout_min_compressed"]
        ),
        "causal_work_fraction": float(
            projection["causal_scalar_work_fraction_of_405b"]
        ),
    }


def latest_markdown(
    result: dict[str, Any], result_path: Path, source_commit: str
) -> str:
    d = details(result)
    return f"""# EXP-095A latest result — Online Deep-Residual Affine-Hull Gate

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
- untouched holdout exact positions: `{d['holdout_exact_positions']}`
- causal affine holdout rank p50/p95/max: `{d['causal']['p50']} / {d['causal']['p95']} / {d['causal']['maximum']}`
- causal affine top-1/top-4/top-16: `{pct(d['causal']['top1_fraction'])} / {pct(d['causal']['top4_fraction'])} / {pct(d['causal']['top16_fraction'])}`
- position-aligned EXP-094A-equivalent p95/max: `{d['aligned']['p95']} / {d['aligned']['maximum']}`
- best-single-residual oracle p95/max/top-16: `{d['single']['p95']} / {d['single']['maximum']} / {pct(d['single']['top16_fraction'])}`
- full residual-span L2 oracle p50/p95/max: `{d['span']['p50']} / {d['span']['p95']} / {d['span']['maximum']}`
- full residual-span oracle top-1/top-4/top-16: `{pct(d['span']['top1_fraction'])} / {pct(d['span']['top4_fraction'])} / {pct(d['span']['top16_fraction'])}`
- holdout residual-bank effective ranks: `{d['span_ranks']}`
- holdout projection relative-L2 values: `{d['span_l2']}`
- exact-chain / causal-branch / oracle-span Gates: `{d['exact_chain']} / {d['causal_branch']} / {d['oracle_span']}`

## Favorable target projection

- full-block raw/compressed fine-sweep traffic/token: `{pct(d['full_raw'])} / {pct(d['full_compressed'])}`
- observed-min compressed source traffic/token: `{pct(d['observed_compressed'])}`
- causal affine scalar-work fraction of 405B: `{pct(d['causal_work_fraction'])}`
- added float64 hidden bank: `{d['hidden_bank_bytes']:,}` bytes
- added float64 Gram matrix: `{d['gram_bytes']:,}` bytes
- projected total hot state: `{d['hot_bytes']:,}` bytes (`{d['hot_gib']:.6f} GiB`)
- projected 8-GiB Gate: `{d['hot_pass']}`
- fine dense arithmetic/token: `100%`
- sound token certificate: `NOT CONSTRUCTED`

## Meaning

The causal chain uses only one guessed-context target sweep, the official first-layer/Q4 coarse state, and a branch-conditioned affine combination of the online residual bank. The single-row and complete-span oracles are computed only after the delayed target and are favorable capacity diagnostics, not deployable candidates.

## Claim boundary

TARGET-W acceptance, a sound coefficient/token certificate, exact successor commitment, physical VRAM, fine arithmetic reduction, and same-machine 4B-class p50/p95 remain `NOT TESTED`.
"""


def update(result_path: Path, source_commit: str) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    d = details(result)
    (ROOT / "docs/research/EXP_095A_LATEST_RESULT.md").write_text(
        latest_markdown(result, result_path, source_commit), encoding="utf-8"
    )

    if d["decision"] == "PROMOTE_ONLINE_RESIDUAL_SPAN_TO_CAUSAL_COEFFICIENT_GATE":
        next_action = (
            "Freeze the online residual basis and construct a causal finite-word coefficient generator from the corrected shallow state. It must reproduce the oracle rank signal on untouched states without target logits, then add a sound token-margin certificate and charge coefficient generation, scoring, state reconstruction, and the remaining fine sweep."
        )
        failed = (
            "The causal affine router was not promoted, but the full online residual span retained a decisive token-source signal. Do not close the whole residual-bank source until the causal coefficient Gate runs."
        )
    elif d["exact_chain"] or d["causal_branch"]:
        next_action = (
            "Freeze the causal affine source and build a sound exact token certificate plus physical block executor. Charge prefix-state commitment, verifier arithmetic, rollback, packed Q4, residual scoring, fine sweep, and target hardware resources."
        )
        failed = (
            "EXP-095A was promoted; no failed-family closure is registered for the causal source."
        )
    else:
        next_action = (
            "Do not tune the affine residual hull. The next admissible mechanism must change the information dependency: a finite-word token-margin certificate, a checkpoint-derived branch-specific higher-order residual, or an exact symbolic transition that reduces the fine dense arithmetic rather than projecting delayed target residuals."
        )
        failed = (
            "EXP-095A rejects the frozen online deep-residual affine hull as a 405B token source. Neither the causal eight-neighbor affine transport nor the full-bank target-residual L2 projection met the bounded-rank Gate under the frozen scope."
        )

    replace_block(
        ROOT / "RESEARCH_STATE.md",
        f"""## EXP-095A — online deep-residual affine hull

- decision: `{d['decision']}`
- evidence: E2/E3 public-checkpoint causal candidate plus favorable online-span oracle
- build accepted prefixes: `{d['build_accepted']}`
- untouched holdout accepted prefixes: `{d['holdout_accepted']}`
- causal holdout rank p50/p95/max: `{d['causal']['p50']} / {d['causal']['p95']} / {d['causal']['maximum']}`
- single-residual oracle p95/max: `{d['single']['p95']} / {d['single']['maximum']}`
- complete residual-span oracle p50/p95/max: `{d['span']['p50']} / {d['span']['p95']} / {d['span']['maximum']}`
- exact-chain / causal-branch / oracle-span: `{d['exact_chain']} / {d['causal_branch']} / {d['oracle_span']}`
- projected hot state: `{d['hot_gib']:.6f} GiB` (`PROJECTED`)
- raw evidence: `{result_path.as_posix()}`
- source commit: `{source_commit}`

One current-block target sweep generated the complete online deep-residual bank. The candidate combined eight residual rows according to the corrected official first-layer hidden state before target continuation. The full-row-span oracle used delayed target residuals only as a nondeployable capacity screen.
""",
    )
    replace_block(
        ROOT / "NEXT_EXPERIMENT.md",
        f"""## Post EXP-095A handoff

Current decision: `{d['decision']}`.

{next_action}

Fine target arithmetic remains `100%`; the `{d['hot_gib']:.6f} GiB` hot figure is favorable and physical allocation remains `NOT TESTED`.
""",
    )
    replace_block(
        ROOT / "DECISION_LOG.md",
        f"""## EXP-095A decision

`{d['decision']}`

The frozen causal affine source produced holdout prefixes `{d['holdout_accepted']}` and rank p95/max `{d['causal']['p95']}/{d['causal']['maximum']}`. The target-seeing best-single residual reached p95/max `{d['single']['p95']}/{d['single']['maximum']}`. The complete online residual row-span L2 oracle reached p95/max `{d['span']['p95']}/{d['span']['maximum']}` with effective ranks `{d['span_ranks']}`.
""",
    )
    failed_body = f"""## EXP-095A — online deep-residual affine hull

{failed}

Frozen fingerprint: one K=128 fine guessed sweep; official first-layer/Q4 shallow states; float64 128-row residual bank; exact hidden-row reuse plus eight-neighbor affine ridge; target-seeing best-single residual; full residual-row-span L2 projection at `2^-40`; delayed incremental target; build plus untouched holdout.
"""
    replace_block(ROOT / "FAILED_APPROACHES.md", failed_body)
    replace_block(ROOT / "FAILED_APPROACHES_RECENT.md", failed_body)
    replace_block(
        ROOT / "ASSUMPTION_REGISTER.md",
        f"""## EXP-095A assumptions and grants

- `MEASURED`: DEV-W guessed fine logits, guessed/corrected/true shallow states, causal candidate, delayed target logits/tokens/caches and exact stable ranks.
- `DERIVED`: affine-ridge coefficients, best-single residual ranks, float64 Gram pseudoinverse, L2 span projection, logical traffic and hot-state equations.
- `GRANTED`: the target-seeing single-residual and span oracles may inspect delayed target residuals; they are not runtime constructions.
- `PROJECTED`: `{d['hot_gib']:.6f} GiB` target hot state and `{pct(d['causal_work_fraction'])}` causal scalar-work fraction.
- `UNVERIFIED`: sound coefficient/token certificate, physical kernels, fine arithmetic reduction, TARGET-W acceptance, target VRAM and latency.
""",
    )
    replace_block(
        ROOT / "VALIDATION_MATRIX.md",
        f"""## EXP-095A validation row

| Boundary | Contract | Result | Evidence |
|---|---|---|---|
| Causal affine candidate | complete before delayed target | `{d['integrity']}` | MEASURED |
| Exact chain | every build/holdout prefix `128` | holdout `{d['holdout_accepted']}` | E2/E3 |
| Causal branch source | p95 `<=4`, maximum `<=16` | `{d['causal']['p95']} / {d['causal']['maximum']}` | E2/E3 |
| Full residual-span capacity | p95 `<=4`, maximum `<=16` | `{d['span']['p95']} / {d['span']['maximum']}` | favorable oracle |
| Sound certificate / 405B / target latency | complete target contract | `NOT TESTED` | — |
""",
    )
    replace_block(
        ROOT / "ARCHITECTURE.md",
        f"""## EXP-095A architecture status

The online residual bank and its affine/span programs remain candidate-source objects. Only a causal source with a sound finite-word token certificate, exact prefix-state commitment, and charged physical resource trace may enter the executor architecture. Promotion status: exact-chain `{d['exact_chain']}`, causal-branch `{d['causal_branch']}`, oracle-span `{d['oracle_span']}`.
""",
    )
    replace_block(
        ROOT / "HARDWARE_VALIDATION_PLAN.md",
        f"""## EXP-095A hardware status

The favorable target ledger totals `{d['hot_bytes']:,}` bytes (`{d['hot_gib']:.6f} GiB`) and the 8-GiB arithmetic comparison reports `{d['hot_pass']}`. This is `PROJECTED`; allocator fragmentation, packed kernels, HBM/PCIe/SSD, fine sweep throughput, certificate work and p50/p95 remain `NOT TESTED`.
""",
    )
    replace_block(
        ROOT / "REPRODUCIBILITY.md",
        f"""## EXP-095A reproduction

```bash
python -m pip install --disable-pip-version-check -r requirements/fixed-public-dynamic-executor.txt
pytest -q tests/exp_095a
python experiments/exp_095a/run_experiment.py \\
  --config experiments/exp_095a/config.json \\
  --output-dir results/exp_095a/<source-commit>
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
