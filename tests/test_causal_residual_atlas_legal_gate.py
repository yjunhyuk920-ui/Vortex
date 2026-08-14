from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from vortex_runtime.causal_residual_atlas_legal_gate import (
    certify_top1_from_row_norms,
    certified_unread_radius,
    compile_pair_only_mgs,
    derive_legal_pair_outward_gate,
    LEGAL_PROMPT_PATH,
    LEGAL_PROMPT_SHA256,
    native_linear_l2_rounding_bound,
    pair_image_defect_bound,
    projection_output_radius,
    rmsnorm_output_radius,
    select_max_residual_energy_page,
    verify_exact_spectral_upper_bound,
)


def test_frozen_gate_is_unseen_last_down_and_zero_failure() -> None:
    gate = derive_legal_pair_outward_gate()
    assert gate.rank == 16
    assert gate.page_columns == 64
    assert gate.layer_index == 23
    assert gate.projection == "down_proj"
    assert gate.teacher_position_index == 1
    assert gate.evaluation_prompts == 24
    assert gate.required_token_successes == 24
    assert gate.required_family_successes == 4
    assert gate.maximum_failures == 0
    assert gate.minimum_coverage > 0.999
    prompt_path = Path(LEGAL_PROMPT_PATH)
    assert hashlib.sha256(prompt_path.read_bytes()).hexdigest() == LEGAL_PROMPT_SHA256
    rows = json.loads(prompt_path.read_text(encoding="utf-8"))["evaluation"]
    assert len(rows) == 24
    assert len({row["id"] for row in rows}) == 24
    assert {
        family: sum(row["family"] == family for row in rows)
        for family in {row["family"] for row in rows}
    } == {
        "english": 4,
        "korean": 4,
        "code": 4,
        "structured_json": 4,
        "math": 4,
        "adversarial_low_acceptance": 4,
    }


def test_fully_charged_favorable_target_equations_remain_below_gate() -> None:
    accounting = derive_legal_pair_outward_gate().accounting
    assert accounting.proof_metadata_bytes == 1_055_320
    assert accounting.model_service_tokens == 20_000_000
    assert accounting.charged_traffic_fraction < (8 / 675)
    assert accounting.charged_operation_fraction < (8 / 675)
    assert accounting.minimum_static_service_tokens_operations is not None
    assert accounting.model_service_tokens >= accounting.minimum_static_service_tokens_operations
    assert accounting.capsule_and_metadata_gib < 1.0
    assert accounting.unallocated_hot_gib > 7.0


def test_selector_signature_cannot_receive_weights_outputs_or_logits() -> None:
    parameters = set(inspect.signature(select_max_residual_energy_page).parameters)
    assert parameters == {"residual", "page_columns"}


def test_selector_chooses_maximum_energy_and_lowest_tie() -> None:
    residual = np.asarray([3.0, 4.0, 0.0, 0.0, -5.0, 0.0])
    assert select_max_residual_energy_page(residual, page_columns=2) == 0
    residual[4] = 6.0
    assert select_max_residual_energy_page(residual, page_columns=2) == 2


def test_spectral_unread_radius_never_understates_exact_error() -> None:
    rng = np.random.default_rng(8401)
    for _ in range(30):
        weight = rng.normal(size=(9, 13))
        residual = rng.normal(size=13)
        selected = select_max_residual_energy_page(residual, page_columns=4)
        beta = math.nextafter(float(np.linalg.norm(weight, ord=2)), math.inf)
        radius = certified_unread_radius(
            beta,
            residual,
            page_columns=4,
            selected_page=selected,
        )
        unread = residual.copy()
        unread[selected * 4 : selected * 4 + 4] = 0.0
        assert np.linalg.norm(weight @ unread) <= radius


def test_exact_dyadic_spectral_certificate_accepts_only_strict_upper() -> None:
    weight = np.asarray([[3.0, 0.0], [0.0, 4.0]])
    assert verify_exact_spectral_upper_bound(weight, math.nextafter(4.0, math.inf))
    assert not verify_exact_spectral_upper_bound(weight, 4.0)
    assert not verify_exact_spectral_upper_bound(weight, 3.9)
    wide = np.asarray([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    assert verify_exact_spectral_upper_bound(
        wide,
        math.nextafter(4.0, math.inf),
        maximum_columns=2,
    )


def test_pair_and_projection_radius_charge_every_term() -> None:
    rng = np.random.default_rng(8403)
    weight = rng.normal(size=(7, 11))
    prefix_inputs = rng.normal(size=(8, 11))
    prefix_images = prefix_inputs @ weight.T
    pair = compile_pair_only_mgs(
        prefix_inputs,
        prefix_images,
        maximum_rank=5,
    )
    assert pair.basis.shape == (11, 5)
    assert np.allclose(pair.basis, prefix_inputs.T @ pair.coefficient_map)
    assert np.allclose(pair.image, prefix_images.T @ pair.coefficient_map)
    assert np.allclose(pair.image, weight @ pair.basis, atol=1e-12)
    assert "weight" not in inspect.signature(compile_pair_only_mgs).parameters

    defect = pair_image_defect_bound(
        operator_norm_upper=2.0,
        stored_basis_error_operator=0.1,
        prefix_image_error_frobenius=0.2,
        pair_coefficient_operator=3.0,
        stored_image_error_operator=0.4,
    )
    assert defect >= 1.2
    radius = projection_output_radius(
        pair_image_defect=defect,
        coordinate_norm_upper=2.0,
        unread_radius=3.0,
        selected_page_arithmetic_radius=0.1,
        candidate_cast_radius=0.2,
        native_dense_arithmetic_radius=0.3,
    )
    assert radius >= 6.0


def _rmsnorm(value: np.ndarray, gain: np.ndarray, epsilon: float) -> np.ndarray:
    return value / math.sqrt(float(np.mean(value * value)) + epsilon) * gain


def test_local_rmsnorm_radius_is_sound_on_random_ball_samples() -> None:
    rng = np.random.default_rng(8402)
    center = rng.normal(size=16)
    gain = rng.uniform(0.5, 1.5, size=16)
    input_radius = 0.15
    bound = rmsnorm_output_radius(
        center,
        input_radius=input_radius,
        gain_abs_max=float(np.max(np.abs(gain))),
        epsilon=1e-6,
    )
    reference = _rmsnorm(center, gain, 1e-6)
    for _ in range(200):
        direction = rng.normal(size=16)
        direction /= np.linalg.norm(direction)
        distance = rng.random() * input_radius
        actual = _rmsnorm(center + distance * direction, gain, 1e-6)
        assert np.linalg.norm(actual - reference) <= bound


def test_row_norm_top1_certificate_is_strict_and_has_no_false_accept() -> None:
    weight = np.asarray([[2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    center = np.asarray([2.0, 0.0])
    logits = weight @ center
    radius = 0.1
    certificate = certify_top1_from_row_norms(
        logits,
        hidden_radius=radius,
        row_norm_uppers=np.linalg.norm(weight, axis=1),
    )
    assert certificate.certified
    assert certificate.winner == 0
    for angle in np.linspace(0, 2 * math.pi, 1000, endpoint=False):
        perturbation = radius * np.asarray([math.cos(angle), math.sin(angle)])
        assert int(np.argmax(weight @ (center + perturbation))) == certificate.winner

    tied = certify_top1_from_row_norms(
        np.asarray([1.0, 1.0]),
        hidden_radius=0.0,
        row_norm_uppers=np.asarray([0.0, 0.0]),
    )
    assert not tied.certified
    one_ulp_gap = certify_top1_from_row_norms(
        np.asarray([math.nextafter(1.0, math.inf), 1.0]),
        hidden_radius=0.0,
        row_norm_uppers=np.asarray([0.0, 0.0]),
    )
    assert not one_ulp_gap.certified


def test_native_rounding_bound_rejects_malformed_and_is_positive() -> None:
    bound = native_linear_l2_rounding_bound(
        input_norm_upper=3.0,
        frobenius_norm_upper=5.0,
        operator_norm_upper=4.0,
        rows=8,
        columns=16,
    )
    assert math.isfinite(bound) and bound > 0
    with pytest.raises(ValueError):
        native_linear_l2_rounding_bound(
            input_norm_upper=3.0,
            frobenius_norm_upper=5.0,
            operator_norm_upper=4.0,
            rows=8,
            columns=0,
        )
