#!/usr/bin/env python3
"""Run EXP-066 with the precommitted threshold-censored cheap-kill screen."""
from __future__ import annotations

from experiments.exp_066 import run_experiment as base
from vortex_runtime.tensor_train_screen import (
    certify_screen_family,
    select_favorable_screen_plan,
    selected_screen_certificate_rows,
)


def _certify_mode_family(matrix, **kwargs):
    return certify_screen_family(
        matrix,
        rejection_operation_fraction=0.25,
        rejection_storage_fraction=0.25,
        **kwargs,
    )


base.certify_mode_family = _certify_mode_family
base.select_favorable_tt_plan = select_favorable_screen_plan
base.selected_certificate_rows = selected_screen_certificate_rows


if __name__ == "__main__":
    base.main()
