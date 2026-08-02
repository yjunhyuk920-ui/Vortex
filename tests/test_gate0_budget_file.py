import json
from pathlib import Path

from scripts.run_architecture_gate0 import generate_report


def test_committed_gate0_budget_matches_generator() -> None:
    root = Path(__file__).resolve().parents[1]
    committed = json.loads(
        (root / "architecture_gate0_budget.json").read_text(encoding="utf-8")
    )
    assert committed == generate_report(root)
