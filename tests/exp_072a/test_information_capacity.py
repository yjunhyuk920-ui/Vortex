from __future__ import annotations

import pytest

from vortex_runtime.information_capacity import (
    GIB,
    InformationCapacityError,
    audit_finite_domain,
    audit_target_capacity,
    exact_matvec,
    matrix_population_size,
    minimum_universal_maximum_bits,
    power_of_two_alphabet_information_bits,
    standard_basis_signature,
    variable_length_binary_capacity,
)


def test_variable_length_capacity_boundary_for_q4_maps() -> None:
    for coefficients in range(1, 6):
        population = 16 ** coefficients
        required = minimum_universal_maximum_bits(population)
        assert required == 4 * coefficients
        assert variable_length_binary_capacity(required - 1) < population
        assert variable_length_binary_capacity(required) >= population


@pytest.mark.parametrize(
    ("rows", "columns", "alphabet_size"),
    ((1, 2, 4), (2, 2, 4), (1, 2, 16)),
)
def test_finite_domain_basis_signatures_are_injective(rows: int, columns: int, alphabet_size: int) -> None:
    audit = audit_finite_domain(rows, columns, alphabet_size)
    assert audit.matrix_count == matrix_population_size(rows, columns, alphabet_size)
    assert audit.injective
    assert audit.boundary_valid


def test_standard_basis_recovers_columns_and_mutation_changes_signature() -> None:
    matrix = ((-8, 3, 7), (2, 0, -1))
    assert standard_basis_signature(matrix) == ((-8, 2), (3, 0), (7, -1))
    mutated = ((-8, 3, 6), (2, 0, -1))
    assert standard_basis_signature(mutated) != standard_basis_signature(matrix)
    assert exact_matvec(matrix, (0, 0, 1)) == (7, -1)


def test_structured_zero_map_does_not_weaken_universal_bound() -> None:
    zero = ((0, 0), (0, 0))
    assert standard_basis_signature(zero) == ((0, 0), (0, 0))
    assert matrix_population_size(2, 2, 16) == 16 ** 4
    assert minimum_universal_maximum_bits(16 ** 4) == 16


def test_registered_405b_hot_capacity_fails_before_overhead() -> None:
    audit = audit_target_capacity(
        coefficient_count=405_849_243_648,
        alphabet_size=16,
        hot_allowance_bytes=8 * GIB,
        former_static_gate_fraction=0.10,
    )
    assert audit.q4_information_bits == 1_623_396_974_592
    assert audit.q4_information_gib == 188.98828125
    assert audit.hot_allowance_bits == 68_719_476_736
    assert audit.hot_allowance_fraction == pytest.approx(0.042330666997375005)
    assert audit.required_over_hot_factor == pytest.approx(23.62353515625)
    assert audit.former_static_gate_gib == pytest.approx(18.898828124976717)
    assert not audit.former_static_gate_fits_hot
    assert not audit.universal_self_contained_hot_gate_pass


def test_invalid_domains_fail_closed() -> None:
    with pytest.raises(InformationCapacityError):
        matrix_population_size(0, 1, 16)
    with pytest.raises(InformationCapacityError):
        variable_length_binary_capacity(-1)
    with pytest.raises(InformationCapacityError):
        power_of_two_alphabet_information_bits(4, 15)
    with pytest.raises(InformationCapacityError):
        exact_matvec(((1, 2),), (1,))
