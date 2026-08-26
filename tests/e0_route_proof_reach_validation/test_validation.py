from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "e0_route_proof_reach_validation"
    / "run_validation.py"
)
spec = importlib.util.spec_from_file_location("e0_route_proof_reach_validation", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("failed to load validation module")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

SUMMARY = module.build_summary()


def test_routecell_complete_search_rejects_registered_grammar() -> None:
    route = SUMMARY["routecell"]
    assert route["registered_scope"]["unique_tag_partitions"] == 2262
    assert route["integrity"]["complete_encoder_tag_pairs"] == 3_474_432
    assert route["integrity"]["offset_invariance_failures"] == 0
    assert route["results"]["full_two_probe_encoders"] == 0
    assert route["results"]["best_two_probe_coverage"] == 16
    assert route["results"]["best_worst_query_depth"] == 5


def test_routecell_positive_controls() -> None:
    integrity = SUMMARY["routecell"]["integrity"]
    assert integrity["positive_one_coordinate_depth"] == 1
    assert integrity["positive_two_coordinate_parity_depth"] == 2


def test_proofwave_easy_and_dense_controls() -> None:
    proof = SUMMARY["proofwave"]
    assert proof["controls"]["one_coordinate"]["p50"] == 1
    assert proof["controls"]["easy_or_certificate"]["p95"] == 5
    assert proof["controls"]["balanced_dense_margin"]["p50"] == 10
    assert proof["controls"]["balanced_dense_margin"]["p95"] == 12
    assert proof["controls"]["parity_diagnostic_only"]["p50"] == 12
    assert proof["real_checkpoint_evidence"]["causal_coefficient_generator"] == "NOT_CONSTRUCTED"


def test_reachquotient_positive_and_adversarial_controls() -> None:
    reach = SUMMARY["reachquotient"]
    assert reach["positive_control"]["states"] == 32
    assert reach["positive_control"]["classes"] == 8
    assert reach["results"]["shift_register_full_class_cases"] == 7
    assert reach["results"]["shift_register_cases"] == 7
    assert all(value == 1.0 for value in reach["results"]["random_p50_class_fraction_all_sizes"])
    assert reach["contract_weakening_control"]["weaker_contract_merges_more"] is True


def test_overall_claim_boundary() -> None:
    assert SUMMARY["overall"]["decision"] == "NO_ROUTE_PROOF_REACH_CORE_PROMOTED"
    assert SUMMARY["overall"]["numbered_experiment_authorized"] is False
    assert SUMMARY["claim_boundary"]["405b_execution"] == "NOT_TESTED"
    assert SUMMARY["claim_boundary"]["general_adaptive_nonlinear_cell_probe_impossibility"] == "NOT_ESTABLISHED"
    assert len(SUMMARY["deterministic_core_sha256"]) == 64
