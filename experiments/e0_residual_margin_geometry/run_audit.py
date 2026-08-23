from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

VOCABULARY_SIZE = 49_152
MARGIN = 2.0 ** -20
K128_SEEDS = [2026082300, 2026082301, 2026082302, 2026082303, 2026082304, 2026082305]
PHASE_K = [8, 16, 32, 64]


def solve_instance(seed: int, k: int, vocabulary_size: int = VOCABULARY_SIZE) -> dict:
    rng = np.random.default_rng(seed)
    bank = rng.standard_normal((k, vocabulary_size), dtype=np.float64)
    coarse = rng.standard_normal(vocabulary_size, dtype=np.float64)
    target = int(rng.integers(vocabulary_size))

    mask = np.ones(vocabulary_size, dtype=bool)
    mask[target] = False
    delta = (bank[:, mask] - bank[:, [target]]).T
    rhs = coarse[target] - coarse[mask] - MARGIN

    # a = p - n, p,n >= 0. Minimize ||a||_1 under the same winner-margin
    # inequality shape used by EXP-096A.
    aub = np.concatenate((delta, -delta), axis=1)
    result = linprog(
        np.ones(2 * k, dtype=np.float64),
        A_ub=aub,
        b_ub=rhs,
        bounds=(0.0, None),
        method="highs-ds",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )

    row = {
        "seed": seed,
        "k": k,
        "v": vocabulary_size,
        "target": target,
        "status": int(result.status),
        "success": bool(result.success),
    }
    if not result.success:
        return row

    coeff = result.x[:k] - result.x[k:]
    scores = coarse + coeff @ bank
    competitor = np.delete(scores, target)
    achieved_margin = float(scores[target] - competitor.max())
    row.update(
        {
            "l1": float(np.abs(coeff).sum()),
            "linf": float(np.abs(coeff).max()),
            "support": int(np.count_nonzero(np.abs(coeff) > 1e-10)),
            "achieved_margin": achieved_margin,
            "argmax": int(np.argmax(scores)),
        }
    )
    if row["argmax"] != target or achieved_margin <= 0.0:
        raise RuntimeError("solver success did not survive full-vocabulary rescore")
    return row


def rounded(value: float) -> float:
    return float(f"{value:.12g}")


def run() -> dict:
    rows = [solve_instance(seed, 128) for seed in K128_SEEDS]
    rows.extend(solve_instance(2026082400 + k, k) for k in PHASE_K)

    k128 = [row for row in rows if row["k"] == 128]
    k128_ok = [row for row in k128 if row["success"]]
    supports = [row["support"] for row in k128_ok]
    l1s = [row["l1"] for row in k128_ok]

    compact_rows = []
    for row in rows:
        compact = dict(row)
        for key in ("l1", "linf", "achieved_margin"):
            if key in compact:
                compact[key] = rounded(float(compact[key]))
        compact_rows.append(compact)

    phase_rows = []
    for row in compact_rows[len(K128_SEEDS):]:
        phase_rows.append(
            {
                key: row[key]
                for key in (
                    "seed",
                    "k",
                    "success",
                    "support",
                    "l1",
                    "linf",
                    "achieved_margin",
                )
                if key in row
            }
        )

    return {
        "schema": "e0-residual-margin-random-geometry-v1",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "vocabulary_size": VOCABULARY_SIZE,
        "required_unique_margin": MARGIN,
        "k128_trials": len(k128),
        "k128_certified": len(k128_ok),
        "k128_support_p50": rounded(float(np.percentile(supports, 50))),
        "k128_support_p95": rounded(float(np.percentile(supports, 95))),
        "k128_l1_p50": rounded(float(np.percentile(l1s, 50))),
        "k128_l1_p95": rounded(float(np.percentile(l1s, 95))),
        "phase_rows": phase_rows,
        "decision": "DEMOTE_EXP096_SPARSE_SUPPORT_AS_MODEL_SPECIFIC_R_SIGNAL",
        "rows": compact_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
