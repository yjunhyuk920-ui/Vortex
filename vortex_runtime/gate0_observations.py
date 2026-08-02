from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_RESULT_PATH = Path(
    "results/tinyllama_1_1b_exact_span_warm_decode.json"
)


def _portable_source(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def apply_real_model_observation(
    report: dict[str, Any],
    result_path: str | Path = DEFAULT_RESULT_PATH,
) -> dict[str, Any]:
    """Apply a committed E1 result without changing analytic assumptions."""

    path = Path(result_path)
    if not path.exists():
        return report

    result = json.loads(path.read_text(encoding="utf-8"))
    observed = float(
        result["warm_decode_repair"][
            "tokens_per_full_repair_equivalent"
        ]
    )
    required = float(
        report["mechanism"][
            "required_tokens_per_full_repair_equivalent"
        ]
    )

    report["mechanism"].update(
        {
            "observed": observed,
            "observed_source": _portable_source(path),
            "pass": observed >= required,
            "shortfall_factor": required / observed,
            "observed_exact_token_match": bool(
                result["exact_token_match"]
            ),
            "observed_warm_decode_fast_fraction": float(
                result["aggregate"]["warm_decode_fast_fraction"]
            ),
        }
    )
    report["gates"]["observed_repair_efficiency"] = observed >= required
    report["observed_component_decision"] = result["decision"]

    if observed < 300.0:
        report["status"] = "rejected-exact-span-hot-path"
    elif observed < required:
        report["status"] = "blocked-mechanism-below-gate"
    else:
        report["status"] = "gate0-candidate-ready-for-e2-falsification"
    return report
