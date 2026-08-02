import json
from pathlib import Path

from vortex_runtime.feasibility import default_gate0_report


def test_committed_gate0_budget_matches_generator() -> None:
    root = Path(__file__).resolve().parents[1]
    committed = json.loads(
        (root / "architecture_gate0_budget.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (root / "validation_results.json").read_text(encoding="utf-8")
    )
    observed = float(validation["jacobi"]["mean_committed_block"])
    assert committed == default_gate0_report(observed)
