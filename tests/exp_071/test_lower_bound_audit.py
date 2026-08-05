from __future__ import annotations

from vortex_runtime.lower_bound_audit import (
    GIB,
    cgl2015_unit_constant_indicator,
    ckl2018_applicability,
    direct_sum_audit,
    exhaustive_binary_projection_reduction,
    llama_405b_tensor_plan,
    parameter_total,
    q4_field_embedding_supported,
)


def test_registered_llama_plan_total() -> None:
    specs = llama_405b_tensor_plan()
    assert len(specs) == 9
    assert sum(spec.count for spec in specs) == 884
    assert parameter_total(specs) == 405_849_243_648
    assert max(spec.square_subproblem_dimension for spec in specs) == 16_384


def test_ckl_redundancy_branches() -> None:
    assert ckl2018_applicability(64, 63).branch == "small_redundancy"
    assert ckl2018_applicability(64, 64).branch == "tradeoff"
    assert ckl2018_applicability(64, 1024).branch == "tradeoff"
    assert not ckl2018_applicability(64, 1025).applicable
    audit = ckl2018_applicability(16_384, 8 * GIB * 8)
    assert not audit.applicable
    assert audit.side_bits / (audit.dimension * audit.dimension // 4) == 1024


def test_q4_field_embedding_boundary() -> None:
    assert q4_field_embedding_supported(2)
    assert q4_field_embedding_supported(13)
    assert not q4_field_embedding_supported(17)


def test_exhaustive_binary_reduction() -> None:
    result = exhaustive_binary_projection_reduction(3)
    assert result["cases"] == 4164
    assert result["mismatches"] == 0
    assert result["float32_replay_cases"] == result["cases"]


def test_cgl_indicator_is_not_zero() -> None:
    row = cgl2015_unit_constant_indicator(
        dimension=16_384,
        field_size=13,
        side_bits=8 * GIB * 8,
        word_bits=64,
    )
    assert row["minimum_term_probes_unit_constant"] > 0
    assert row["space_overhead_alpha"] > 1


def test_direct_sum_is_not_assumed() -> None:
    audit = direct_sum_audit()
    assert not audit["certified"]
    assert not audit["pigeonhole_division_allowed"]
