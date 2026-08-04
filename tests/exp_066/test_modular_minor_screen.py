from __future__ import annotations

import numpy as np

from vortex_runtime.modular_minor_screen import (
    minor_first_threshold_rank_witness,
)


def test_minor_first_witness_proves_requested_rank() -> None:
    matrix = np.eye(8, dtype=np.int16)
    witness = minor_first_threshold_rank_witness(
        matrix, prime=251, required_rank=4
    )
    assert witness.reached_required_rank
    assert witness.rank_lower_bound == 4
    assert witness.certified_minor_determinant != 0


def test_minor_first_falls_back_for_singular_candidate_minors() -> None:
    matrix = np.zeros((8, 8), dtype=np.int16)
    matrix[0, 7] = 1
    matrix[7, 0] = 1
    witness = minor_first_threshold_rank_witness(
        matrix, prime=257, required_rank=2
    )
    assert witness.reached_required_rank
    assert witness.rank_lower_bound == 2
    assert witness.certified_minor_determinant != 0
