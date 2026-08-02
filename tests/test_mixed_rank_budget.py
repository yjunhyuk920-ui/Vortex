from vortex_runtime.rank_frontier import mixed_rank_budget_point


def test_global58_session45_at_4_8_bits_fits_gate() -> None:
    point = mixed_rank_budget_point(
        global_rank=58,
        session_rank=45,
        global_bits=4,
        session_bits=8,
    )
    assert point.weighted_bit_rank == 592
    assert point.total_rank == 103
    assert point.pass_all is True


def test_global80_session45_at_4_6_bits_fits_gate() -> None:
    point = mixed_rank_budget_point(
        global_rank=80,
        session_rank=45,
        global_bits=4,
        session_bits=6,
    )
    assert point.weighted_bit_rank == 590
    assert point.total_rank == 125
    assert point.pass_all is True


def test_global93_session45_at_4_5_bits_exceeds_traffic() -> None:
    point = mixed_rank_budget_point(
        global_rank=93,
        session_rank=45,
        global_bits=4,
        session_bits=5,
    )
    assert point.weighted_bit_rank == 597
    assert point.uniform_equivalent.traffic_pass is False
    assert point.pass_all is False


def test_global92_session45_at_4_5_bits_fits_gate() -> None:
    point = mixed_rank_budget_point(
        global_rank=92,
        session_rank=45,
        global_bits=4,
        session_bits=5,
    )
    assert point.weighted_bit_rank == 593
    assert point.total_rank == 137
    assert point.pass_all is True
