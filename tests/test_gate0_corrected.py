from vortex_runtime.feasibility import default_gate0_report
from vortex_runtime.gate0_corrected import corrected_gate0_report


def test_compatibility_report_uses_active_accounting_engine() -> None:
    active = default_gate0_report(1.0)
    compatibility = corrected_gate0_report()

    assert compatibility["accounting_contract"] == active["accounting_contract"]
    assert compatibility["memory"] == active["memory"]
    assert compatibility["traffic"] == active["traffic"]
    assert compatibility["compute"] == active["compute"]
    assert compatibility["status"] == "rejected-analytic-compute"
