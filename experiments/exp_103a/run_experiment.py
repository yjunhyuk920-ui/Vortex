from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from vortex_runtime.native_rounding_page_certificate import (
    bf16_round,
    certify_page_termwise_noop,
    float32_to_bf16_bits,
    native_separate_mul_add,
    simulate_row_page_certificate,
    traffic_ledger,
)

TARGET_FRACTION = 0.011851851851851851
PAGE_SIZES = (32, 64, 128, 256, 512, 1024)
RNG_SEED = 103_001


def deterministic_core(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def random_soundness_control(rng: np.random.Generator, cases: int = 200_000) -> dict[str, int]:
    certified = 0
    mismatches = 0
    for _ in range(cases):
        # Accumulators cover both tiny and large magnitudes, including negative values.
        exponent = int(rng.integers(-20, 26))
        accumulator = np.float32(rng.choice(np.array([-1.0, 1.0], dtype=np.float32)) * np.ldexp(1.0, exponent))
        length = int(rng.integers(1, 65))
        weights = bf16_round(rng.normal(0.0, 2.0 ** int(rng.integers(-24, 0)), size=length)).astype(np.float32)
        activations = bf16_round(rng.normal(0.0, 2.0 ** int(rng.integers(-8, 5)), size=length)).astype(np.float32)
        exponent_code = int((((float32_to_bf16_bits(weights) >> 7) & 0xFF).max(initial=0)))
        activation_max = float(np.max(np.abs(activations), initial=np.float32(0.0)))
        if not certify_page_termwise_noop(accumulator, exponent_code, activation_max):
            continue
        certified += 1
        candidate = accumulator
        for weight, activation in zip(weights, activations, strict=True):
            candidate = native_separate_mul_add(candidate, weight, activation)
        if candidate.view(np.uint32) != accumulator.view(np.uint32):
            mismatches += 1
    return {"cases": cases, "certified_pages": certified, "mismatches": mismatches}


def run() -> dict[str, Any]:
    rng = np.random.default_rng(RNG_SEED)

    traffic_rows = []
    for page_size in PAGE_SIZES:
        ledger = traffic_ledger(page_size, metadata_bytes_per_page=1, target_fraction=TARGET_FRACTION)
        traffic_rows.append({
            "page_size": page_size,
            "metadata_bytes_per_page": 1,
            "metadata_fraction": ledger.metadata_fraction,
            "maximum_full_page_read_fraction": ledger.maximum_full_page_read_fraction,
            "required_skip_fraction": ledger.required_skip_fraction,
        })

    soundness = random_soundness_control(rng)

    # Positive control: once the accumulator is large, tiny products are exact no-ops.
    tiny_weights = np.full(4096, np.float32(2.0 ** -20), dtype=np.float32)
    unit_activations = np.ones(4096, dtype=np.float32)
    positive_rows = {}
    for page_size in PAGE_SIZES:
        # Seed the dot with 2**24 by prepending one exact term, then test tiny pages.
        weights = np.concatenate((np.array([2.0 ** 24], dtype=np.float32), tiny_weights))
        activations = np.concatenate((np.array([1.0], dtype=np.float32), unit_activations))
        positive_rows[str(page_size)] = simulate_row_page_certificate(weights, activations, page_size)

    # Adversarial but legal finite-word dense row: every 1*1 term changes FP32 sum.
    adversarial_rows = {}
    ones = np.ones(16_384, dtype=np.float32)
    for page_size in PAGE_SIZES:
        adversarial_rows[str(page_size)] = simulate_row_page_certificate(ones, ones, page_size)

    # Realistic synthetic dense rows: BF16 Gaussian weights/activations, no cherry-picking.
    gaussian_rows: dict[str, list[dict[str, float | int | bool]]] = {str(size): [] for size in PAGE_SIZES}
    for row_index in range(64):
        weights = rng.normal(0.0, 0.02, size=4096).astype(np.float32)
        activations = rng.normal(0.0, 1.0, size=4096).astype(np.float32)
        for page_size in PAGE_SIZES:
            row = simulate_row_page_certificate(weights, activations, page_size)
            row["row_index"] = row_index
            gaussian_rows[str(page_size)].append(row)

    gaussian_summary = {}
    for page_size, rows in gaussian_rows.items():
        reads = np.array([float(row["page_read_fraction"]) for row in rows], dtype=np.float64)
        terms = np.array([float(row["term_execution_fraction"]) for row in rows], dtype=np.float64)
        gaussian_summary[page_size] = {
            "row_count": len(rows),
            "exact_all": all(bool(row["exact_match"]) for row in rows),
            "page_read_fraction_p50": float(np.quantile(reads, 0.50)),
            "page_read_fraction_p95": float(np.quantile(reads, 0.95)),
            "page_read_fraction_max": float(reads.max(initial=0.0)),
            "term_execution_fraction_p50": float(np.quantile(terms, 0.50)),
        }

    integrity_failures: list[str] = []
    if soundness["mismatches"] != 0:
        integrity_failures.append("certificate_soundness_mismatch")
    if not all(row["exact_match"] for row in adversarial_rows.values()):
        integrity_failures.append("adversarial_reference_mismatch")
    if not all(summary["exact_all"] for summary in gaussian_summary.values()):
        integrity_failures.append("gaussian_reference_mismatch")

    # Universal rejection: every registered page size must read all weight pages on
    # the legal all-ones dense row, while the traffic target allows <1.2% total.
    adversarial_fail = all(
        float(row["page_read_fraction"]) == 1.0 for row in adversarial_rows.values()
    )
    if integrity_failures:
        decision = "INVALID_NATIVE_ROUNDING_PAGE_CERTIFICATE_CONTROL_FAILURE"
    elif adversarial_fail:
        decision = "REJECT_HIERARCHICAL_NATIVE_ROUNDING_PAGE_CERTIFICATES_AS_UNIVERSAL_10X_CORE"
    else:
        decision = "SURVIVE_HIERARCHICAL_NATIVE_ROUNDING_PAGE_CERTIFICATE_GATE"

    result: dict[str, Any] = {
        "schema": "exp-103a-native-rounding-page-certificate-v1",
        "authoritative_arm": "REAL_EXECUTOR_ONLY_FINITE_WORD_PRIMITIVE",
        "decision": decision,
        "integrity_failures": integrity_failures,
        "target_fraction": TARGET_FRACTION,
        "weight_format": "BF16",
        "accumulator_abi": "separate BF16xBF16->FP32 multiply then RNE FP32 add, fixed left-to-right order",
        "metadata_abi": "one maximum absolute BF16 exponent code byte per output-row/input-page",
        "traffic_rows": traffic_rows,
        "soundness_control": soundness,
        "positive_control": positive_rows,
        "adversarial_all_ones": adversarial_rows,
        "gaussian_dense_summary": gaussian_summary,
        "claim_boundary": {
            "public_checkpoint_execution": "NOT_TESTED",
            "target_405b_execution": "NOT_TESTED",
            "physical_8gib": "NOT_TESTED",
            "native_kernel": "NOT_TESTED",
            "reason_no_model_run": "universal finite-word adversarial byte Gate is already decisive",
        },
        "three_principles_screened": [
            {
                "name": "hierarchical_native_rounding_page_certificate",
                "status": "implemented_and_rejected_as_universal_core",
                "premise_flip": "a weight product need not be evaluated when native rounding proves its addition is a no-op",
            },
            {
                "name": "prefix_transition_monoid_binary_lifting",
                "status": "rejected_before_implementation_as_closed_self_contained_transition_table_family",
                "premise_flip": "compose token transitions instead of evaluating them sequentially",
            },
            {
                "name": "deferred_mismatch_state_closure",
                "status": "retained_auxiliary_only",
                "premise_flip": "materialize correction state in the next sweep rather than a separate repair call",
            },
        ],
        "next_gate": "derive a cross-page or cross-matrix exact computation that changes values, not merely certifies native no-ops; it must expose a fully charged sub-dense decoder before implementation",
    }
    result["deterministic_core"] = deterministic_core(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "integrity_failures": result["integrity_failures"],
        "deterministic_core": result["deterministic_core"],
    }, indent=2))


if __name__ == "__main__":
    main()
