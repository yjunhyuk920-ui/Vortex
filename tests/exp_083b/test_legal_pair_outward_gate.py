from __future__ import annotations

import inspect
import math

import numpy as np

from experiments.exp_083b.run_experiment import (
    build_legal_query_state,
    build_selected_page_candidate,
)
from vortex_runtime.causal_residual_atlas_legal_execution import (
    bound_legal_projection_candidate,
    compile_stored_pair_certificate,
    propagate_last_down_to_final_hidden,
)
from vortex_runtime.causal_residual_atlas_legal_gate import (
    PROMOTE_LEGAL_DECISION,
    REJECT_LEGAL_DECISION,
    round_to_bfloat16_values,
    select_max_residual_energy_page,
    summarize_legal_pair_outward_gate,
    verified_cholesky_spectral_upper_bound,
)


def _bf16(values: np.ndarray) -> np.ndarray:
    return round_to_bfloat16_values(np.asarray(values, dtype=np.float64))


def test_verified_cholesky_bound_is_strict_for_wide_and_tall_matrices() -> None:
    rng = np.random.default_rng(8310)
    for shape in ((7, 11), (11, 7), (9, 9)):
        weight = rng.normal(size=shape)
        proof = verified_cholesky_spectral_upper_bound(weight)
        assert proof.operator_norm_upper > np.linalg.norm(weight, ord=2)
        assert proof.positive_definite_margin_lower > 0
        assert proof.inverse_residual_operator_upper < 1
        assert proof.proof_method == "outward_gram_verified_cholesky_residual"


def test_numpy_bfloat16_rounding_is_ties_to_even() -> None:
    one = np.float32(1.0)
    upper = np.nextafter(one, np.float32(math.inf), dtype=np.float32)
    values = np.asarray([one, upper, -upper, 3.1415927], dtype=np.float32)
    rounded = _bf16(values)
    assert rounded.dtype == np.float64
    assert rounded[0] == 1.0
    assert rounded[1] == 1.0
    assert rounded[2] == -1.0
    assert rounded[3] == float(np.float32(3.140625))


def test_pair_certificate_has_no_weight_argument_and_bounds_true_defect() -> None:
    assert "weight" not in inspect.signature(
        compile_stored_pair_certificate
    ).parameters
    rng = np.random.default_rng(8311)
    weight = _bf16(rng.normal(scale=0.2, size=(12, 20)))
    inputs = _bf16(rng.normal(size=(24, 20)))
    images = _bf16(inputs @ weight.T)
    spectral = verified_cholesky_spectral_upper_bound(weight)
    pair = compile_stored_pair_certificate(
        inputs,
        images,
        maximum_rank=8,
        operator_norm_upper=spectral.operator_norm_upper,
        frobenius_norm_upper=spectral.frobenius_norm_upper,
    )
    assert pair.rank == 8
    assert all(
        lower > upper
        for lower, upper in zip(
            pair.accepted_residual_norm_lowers,
            pair.accepted_threshold_uppers,
        )
    )
    actual = np.linalg.norm(
        weight @ pair.stored_basis - pair.stored_image,
        ord=2,
    )
    assert actual <= pair.pair_image_defect_operator


def test_current_query_interfaces_cannot_receive_native_evaluator_state() -> None:
    query_parameters = set(inspect.signature(build_legal_query_state).parameters)
    assert query_parameters == {"stored_basis", "current_input", "page_columns"}
    candidate_parameters = set(
        inspect.signature(build_selected_page_candidate).parameters
    )
    forbidden = {
        "native_output",
        "native_logits",
        "baseline_output",
        "baseline_logits",
        "target_output",
        "target_logits",
    }
    assert not candidate_parameters & forbidden


def test_projection_radius_contains_simulated_bfloat16_candidate() -> None:
    rng = np.random.default_rng(8312)
    weight = _bf16(rng.normal(scale=0.15, size=(10, 18)))
    inputs = _bf16(rng.normal(size=(22, 18)))
    images = _bf16(inputs @ weight.T)
    spectral = verified_cholesky_spectral_upper_bound(weight)
    pair = compile_stored_pair_certificate(
        inputs,
        images,
        maximum_rank=6,
        operator_norm_upper=spectral.operator_norm_upper,
        frobenius_norm_upper=spectral.frobenius_norm_upper,
    )
    current = _bf16(rng.normal(size=18))
    coordinates = _bf16(pair.stored_basis.T @ current)
    basis_application = _bf16(pair.stored_basis @ coordinates)
    residual = _bf16(current - basis_application)
    page_columns = 5
    selected = select_max_residual_energy_page(
        residual,
        page_columns=page_columns,
    )
    start = selected * page_columns
    page = weight[:, start : start + page_columns]
    stored_image_application = _bf16(pair.stored_image @ coordinates)
    selected_page_application = _bf16(
        page @ residual[start : start + page.shape[1]]
    )
    candidate = _bf16(stored_image_application + selected_page_application)
    native = _bf16(weight @ current)
    bound = bound_legal_projection_candidate(
        stored_basis=pair.stored_basis,
        stored_image=pair.stored_image,
        pair_image_defect=pair.pair_image_defect_operator,
        operator_norm_upper=spectral.operator_norm_upper,
        frobenius_norm_upper=spectral.frobenius_norm_upper,
        current_input=current,
        coordinates=coordinates,
        basis_application=basis_application,
        residual=residual,
        selected_page=selected,
        page_columns=page_columns,
        selected_weight_page=page,
        stored_image_application=stored_image_application,
        selected_page_application=selected_page_application,
    )
    assert np.linalg.norm(candidate - native) <= bound.projection_output_radius
    assert bound.unread_radius > 0
    assert bound.decomposition_radius > 0


def test_final_radius_contains_simulated_bfloat16_rmsnorm_paths() -> None:
    rng = np.random.default_rng(8313)
    residual = _bf16(rng.normal(size=32))
    candidate_down = _bf16(rng.normal(scale=0.2, size=32))
    target_down = _bf16(candidate_down + rng.normal(scale=1e-3, size=32))
    projection_radius = math.nextafter(
        float(np.linalg.norm(candidate_down - target_down)),
        math.inf,
    )
    candidate_pre = _bf16(residual + candidate_down)
    target_pre = _bf16(residual + target_down)
    gain = _bf16(rng.uniform(0.8, 1.2, size=32))
    epsilon = 1e-6

    def native_norm(value: np.ndarray) -> np.ndarray:
        exact = value / math.sqrt(float(np.mean(value * value)) + epsilon)
        return _bf16(exact * gain)

    bounds = propagate_last_down_to_final_hidden(
        residual_branch=residual,
        candidate_down=candidate_down,
        candidate_pre_norm=candidate_pre,
        projection_radius=projection_radius,
        gain_abs_max=float(np.max(np.abs(gain))),
        epsilon=epsilon,
    )
    assert np.linalg.norm(native_norm(target_pre) - native_norm(candidate_pre)) <= (
        bounds.final_hidden_radius
    )


def _successful_rows() -> list[dict[str, object]]:
    families = ("english", "korean", "code", "json", "math", "adversarial")
    return [
        {
            "prompt_id": f"{family}-{index}",
            "family": family,
            "success": True,
            "certificate_resolved": True,
            "fallback_executed": False,
            "false_accept": False,
            "scientific_failure": False,
            "target_to_candidate_kl": 0.001,
        }
        for family in families
        for index in range(4)
    ]


def test_gate_aggregator_promotes_only_a_complete_zero_fallback_population() -> None:
    families = ("english", "korean", "code", "json", "math", "adversarial")
    rows = _successful_rows()
    result = summarize_legal_pair_outward_gate(
        rows,
        expected_token_states=24,
        expected_families=families,
        prompts_per_family=4,
        maximum_mean_kl=0.02,
        maximum_p95_kl=0.05,
        execution_complete=True,
    )
    assert result["decision"] == PROMOTE_LEGAL_DECISION
    rows[0]["success"] = False
    rows[0]["certificate_resolved"] = False
    rows[0]["fallback_executed"] = True
    rows[0]["dense_completion_matches_baseline"] = True
    rows[0]["scientific_failure"] = True
    result = summarize_legal_pair_outward_gate(
        rows[:1],
        expected_token_states=24,
        expected_families=families,
        prompts_per_family=4,
        maximum_mean_kl=0.02,
        maximum_p95_kl=0.05,
        execution_complete=False,
    )
    assert result["decision"] == REJECT_LEGAL_DECISION
    assert result["fallbacks"] == 1
