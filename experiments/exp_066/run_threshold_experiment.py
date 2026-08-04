#!/usr/bin/env python3
"""Run EXP-066 with decisive exact minor certificates only."""
from __future__ import annotations

from experiments.exp_066 import run_experiment as base
from vortex_runtime import tensor_train_fast_screen as fast_screen
from vortex_runtime.modular_minor_screen import (
    minor_first_threshold_rank_witness,
)

fast_screen.threshold_rank_witness = minor_first_threshold_rank_witness


def _certify_mode_family(matrix, **kwargs):
    return fast_screen.certify_fast_screen_family(
        matrix,
        rejection_operation_fraction=0.25,
        rejection_storage_fraction=0.25,
        **kwargs,
    )


base.certify_mode_family = _certify_mode_family
base.select_favorable_tt_plan = fast_screen.select_favorable_fast_screen_plan
base.selected_certificate_rows = fast_screen.selected_fast_certificate_rows


if __name__ == "__main__":
    base.main()
