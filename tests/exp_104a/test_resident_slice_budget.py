from __future__ import annotations

from vortex_runtime.resident_slice_budget import (
    BF16,
    GIB,
    Q4_G128,
    Q4_IDEAL,
    Q8_G128,
    LlamaInventory,
    default_reserve,
    draft_weight_scan_bytes,
    maximum_full_layers_under_weight_budget,
    maximum_resident_layers,
)


def test_registered_population_matches_exactly() -> None:
    assert LlamaInventory().registered_parameters == 405_849_243_648


def test_inventory_shapes_match_registered_layer_population() -> None:
    inv = LlamaInventory()
    assert inv.layer_parameters == 3_187_671_040
    assert inv.lm_head_parameters == 2_101_346_304
    assert inv.embedding_parameters == 2_101_346_304


def test_even_ideal_q4_head_plus_one_layer_exceeds_p50_budget() -> None:
    inv = LlamaInventory()
    p50_budget = 2_400_000_000
    assert draft_weight_scan_bytes(inv, Q4_IDEAL, 1) == 2_644_508_672
    assert draft_weight_scan_bytes(inv, Q4_IDEAL, 1) > p50_budget
    assert maximum_full_layers_under_weight_budget(inv, Q4_IDEAL, p50_budget) == 0


def test_realistic_q4_head_plus_one_layer_is_over_1_4x_native_4b_bytes() -> None:
    inv = LlamaInventory()
    scan = draft_weight_scan_bytes(inv, Q4_G128, 1)
    assert scan == 2_809_790_464
    assert scan / 2_000_000_000 > 1.4


def test_fully_charged_8gib_ledger_fits_only_three_q4_layers() -> None:
    inv = LlamaInventory()
    reserve = default_reserve(inv, 339)
    assert maximum_resident_layers(inv, Q4_G128, 339, reserve, 8 * GIB) == 3


def test_bf16_and_q8_are_stricter_than_q4() -> None:
    inv = LlamaInventory()
    reserve = default_reserve(inv, 339)
    assert maximum_resident_layers(inv, BF16, 339, reserve) == 0
    assert maximum_resident_layers(inv, Q8_G128, 339, reserve) == 1


def test_target_kv_is_charged() -> None:
    inv = LlamaInventory()
    assert inv.target_kv_bytes(339) == 174_956_544
