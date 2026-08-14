from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.query_adaptive_cold_equation import (
    GIB,
    KNOWN_SIX_CHECK_VERIFIER_TRAFFIC_FRACTION,
    P50_TARGET_FRACTION,
    P95_TARGET_FRACTION,
    REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES,
    REGISTERED_NON_EMBEDDING_Q4_BYTES,
    REGISTERED_TOTAL_Q4_BYTES,
    ResourceTerms,
    maximum_pages,
    maximum_payload_bytes_per_call,
    minimum_coverage,
    minimum_service_tokens,
    pcie_transfer_floor_seconds,
    resident_metadata_bytes,
    total_page_count,
)


def _text(value: Decimal | int) -> str | int:
    return str(value) if isinstance(value, Decimal) else value


def derive() -> dict[str, object]:
    free_terms = ResourceTerms()
    verifier_terms = ResourceTerms(
        common=KNOWN_SIX_CHECK_VERIFIER_TRAFFIC_FRACTION
    )
    page_rows = []
    for page_bytes in (4 * 1024, 64 * 1024, 1024**2, 2 * 1024**2):
        page_rows.append(
            {
                "page_bytes": page_bytes,
                "max_pages_free": maximum_pages(
                    P50_TARGET_FRACTION, page_bytes
                ),
                "max_pages_after_known_verifier": maximum_pages(
                    P50_TARGET_FRACTION,
                    page_bytes,
                    other_traffic_fraction=(
                        KNOWN_SIX_CHECK_VERIFIER_TRAFFIC_FRACTION
                    ),
                ),
                "full_checkpoint_page_count": total_page_count(
                    REGISTERED_TOTAL_Q4_BYTES, page_bytes
                ),
                "full_checkpoint_32b_metadata_gib": str(
                    Decimal(
                        resident_metadata_bytes(
                            REGISTERED_TOTAL_Q4_BYTES, page_bytes, 32
                        )
                    )
                    / Decimal(GIB)
                ),
            }
        )

    return {
        "classification": "E0_DERIVED_ACCOUNTING_ONLY",
        "normalization": {
            "non_embedding_q4_bytes": REGISTERED_NON_EMBEDDING_Q4_BYTES,
            "full_checkpoint_q4_bytes": REGISTERED_TOTAL_Q4_BYTES,
            "p50_fraction": _text(P50_TARGET_FRACTION),
            "p95_fraction": _text(P95_TARGET_FRACTION),
        },
        "coverage_frontier": {
            "zero_cost_fast_path_p50": _text(
                minimum_coverage(P50_TARGET_FRACTION, free_terms)
            ),
            "known_verifier_p50": _text(
                minimum_coverage(P50_TARGET_FRACTION, verifier_terms)
            ),
        },
        "page_frontier": page_rows,
        "conditional_one_call_per_matrix": {
            "matrix_instances": REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES,
            "max_payload_bytes_free": maximum_payload_bytes_per_call(
                P50_TARGET_FRACTION,
                REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES,
            ),
            "max_payload_bytes_after_known_verifier": (
                maximum_payload_bytes_per_call(
                    P50_TARGET_FRACTION,
                    REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES,
                    other_traffic_fraction=(
                        KNOWN_SIX_CHECK_VERIFIER_TRAFFIC_FRACTION
                    ),
                )
            ),
        },
        "compile_examples": {
            "one_scan_at_0.1pct_headroom_tokens": minimum_service_tokens(
                1, Decimal("0.001")
            ),
            "ten_scans_at_0.1pct_headroom_tokens": minimum_service_tokens(
                10, Decimal("0.001")
            ),
        },
        "physical_floor": {
            "gen2_x16_signaling_ceiling_gib_s": "7.450580596923828",
            "p50_allowance_transfer_seconds": _text(
                pcie_transfer_floor_seconds(
                    P50_TARGET_FRACTION, Decimal("7.450580596923828")
                )
            ),
        },
    }


def main() -> None:
    print(json.dumps(derive(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
