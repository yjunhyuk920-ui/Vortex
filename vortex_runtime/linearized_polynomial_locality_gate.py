"""Cheapest locality Gate for linearized-polynomial matrix encodings.

Every binary ``n x n`` linear map has a unique representation

    L(x) = sum_{i=0}^{n-1} a_i x^(2^i)

over ``GF(2^n)``.  This module charges that representation honestly.  The
identity is exact, but the coefficient list still contains ``n^2`` base-field
bits.  Published fast-operation algorithms consume that list; importing a
dense ordinary-polynomial evaluation data structure instead makes the degree
parameter exponential.  This is a representation/locality screen, not a
lower bound for every possible preprocessed MatVec data structure.
"""

from __future__ import annotations

from decimal import Decimal, localcontext


REGISTERED_PARAMETERS = 405_849_243_648
REGISTERED_SQUARE_SIDE = 16_384
REGISTERED_BATCH_QUERIES = 32
EFFECTIVE_LINK_BYTES_PER_SECOND = 32_000_000_000
TARGET_SECONDS_PER_TOKEN = Decimal("0.020")
REGISTERED_COMPRESSED_CHECKPOINT_BYTES = 551_220_000_000

DECISION = (
    "REJECT_LINEARIZED_POLYNOMIAL_RENAMING_AS_LOCAL_EVALUATOR_"
    "REJECT_DENSE_POLYNOMIAL_DATA_STRUCTURE_IMPORT_"
    "CHANGE_MECHANISM_CLASS"
)


def linearized_representation_accounting(side: int) -> dict[str, int | str]:
    """Return the exact information accounting for one binary square."""

    if side <= 0:
        raise ValueError("side must be positive")
    return {
        "square_side": side,
        "extension_field": f"GF(2^{side})",
        "linearized_coefficients": side,
        "base_bits_per_coefficient": side,
        "total_base_bits": side * side,
        "matrix_base_bits": side * side,
        "q_degree_at_most": side - 1,
        "ordinary_degree_top_exponent": f"2^{side - 1}",
        "equation": "L_W(x)=sum_i a_i x^(2^i)",
        "evaluation_map_bijective": True,
        "representation_compresses_arbitrary_matrix": False,
    }


def full_coefficient_sweep_floor() -> dict[str, str | int | bool]:
    """Charge one favorable coefficient sweep shared by all 32 queries."""

    with localcontext() as context:
        context.prec = 50
        batch = Decimal(REGISTERED_BATCH_QUERIES)
        bandwidth = Decimal(EFFECTIVE_LINK_BYTES_PER_SECOND)
        one_bit_bytes = Decimal(REGISTERED_PARAMETERS) / Decimal(8)
        one_bit_block_seconds = one_bit_bytes / bandwidth
        one_bit_token_seconds = one_bit_block_seconds / batch
        compressed_block_seconds = (
            Decimal(REGISTERED_COMPRESSED_CHECKPOINT_BYTES) / bandwidth
        )
        compressed_token_seconds = compressed_block_seconds / batch
        return {
            "ordered_batch_queries": REGISTERED_BATCH_QUERIES,
            "one_bit_model_bytes": str(one_bit_bytes),
            "one_bit_block_seconds": str(one_bit_block_seconds),
            "one_bit_milliseconds_per_token": str(
                one_bit_token_seconds * Decimal(1000)
            ),
            "one_bit_target_multiple": str(
                one_bit_token_seconds / TARGET_SECONDS_PER_TOKEN
            ),
            "compressed_checkpoint_bytes": (
                REGISTERED_COMPRESSED_CHECKPOINT_BYTES
            ),
            "compressed_block_seconds": str(compressed_block_seconds),
            "compressed_milliseconds_per_token": str(
                compressed_token_seconds * Decimal(1000)
            ),
            "compressed_target_multiple": str(
                compressed_token_seconds / TARGET_SECONDS_PER_TOKEN
            ),
            "all_compute_and_other_costs_granted_free": True,
            "passes_target": False,
        }


def dense_ordinary_polynomial_data_structure_screen(
    side: int = REGISTERED_SQUARE_SIDE,
) -> dict[str, int | str | bool]:
    """Show why a dense degree-parameter polynomial DS cannot be imported.

    The top ordinary exponent is ``2**(n-1)``.  A data structure whose input
    is the dense coefficient list of a degree-``D`` polynomial therefore sees
    more than ``2**(n-1)`` extension-field cells, not the ``n`` nonzero
    coefficients of the special linearized representation.
    """

    if side <= 1 or side & (side - 1):
        raise ValueError("registered screen requires a power-of-two side")
    log2_side = side.bit_length() - 1
    return {
        "square_side": side,
        "ordinary_degree": f"2^{side - 1}",
        "dense_field_cells_strictly_more_than": f"2^{side - 1}",
        "dense_base_bits_log2_strict_lower": side - 1 + log2_side,
        "source_base_bits": side * side,
        "source_base_bits_log2": 2 * log2_side,
        "dense_ds_parameter_matches_sparse_linearized_input": False,
        "dense_ds_import_rejected": True,
    }


def derive_audit() -> dict[str, object]:
    representation = linearized_representation_accounting(
        REGISTERED_SQUARE_SIDE
    )
    sweep = full_coefficient_sweep_floor()
    dense_ds = dense_ordinary_polynomial_data_structure_screen()
    return {
        "classification": "E0_LINEARIZED_POLYNOMIAL_LOCALITY_GATE",
        "representation": representation,
        "published_fast_operation_scope": {
            "input": "explicit q-coefficient list",
            "q_degree_parameter": REGISTERED_SQUARE_SIDE,
            "subquadratic_multipoint_evaluation_exists": True,
            "preprocessed_sublinear_cell_probe_evaluator_supplied": False,
            "arbitrary_binary_matvec_equivalence": True,
        },
        "full_coefficient_sweep_floor": sweep,
        "dense_ordinary_polynomial_ds": dense_ds,
        "claim_boundary": {
            "binary_square_representation_exact": True,
            "direct_representation_has_sublinear_locality": False,
            "published_fast_operations_pass_registered_io_gate": False,
            "dense_polynomial_ds_applicable_at_near_source_space": False,
            "all_preprocessed_linearized_polynomial_ds_rejected": False,
            "native_bf16_fp32_lift": False,
            "target_candidate": False,
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "decisions": [
            "PROMOTE_LINEARIZED_POLYNOMIAL_AS_EXACT_REPRESENTATION_IDENTITY",
            "REJECT_LINEARIZED_POLYNOMIAL_RENAMING_AS_LOCAL_EVALUATOR",
            "REJECT_DENSE_POLYNOMIAL_DATA_STRUCTURE_IMPORT",
            "RECOGNIZE_SPECIALIZED_LOCAL_DS_AS_ORIGINAL_ARBITRARY_MATVEC_GAP",
            "CHANGE_MECHANISM_CLASS",
            "NO_SURVIVING_CANDIDATE",
        ],
        "decision": DECISION,
    }
