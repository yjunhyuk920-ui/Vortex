#!/usr/bin/env python3
"""Exact arithmetic certificate, not a benchmark or proof-assistant proof."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path


def ruling_upper(a: int, b: int, d: int) -> int:
    """Equation (3); a,b may be swapped without changing rank-one geometry."""
    if a > b:
        a, b = b, a
    if a < 1 or b < 1 or not 0 <= d <= a * b:
        raise ValueError("invalid shape or dimension")
    return sum((1 << (i - 1)) * ((1 << min(b, d // i)) - 1)
               for i in range(1, a + 1))


def case(a: int, b: int, cells: int, bits: int, probes: int,
         advice: int = 0) -> dict[str, object]:
    if min(cells, bits) < 1 or min(probes, advice) < 0:
        raise ValueError("invalid resource bound")
    required = ((1 << a) - 1) * ((1 << b) - 1)
    dimension = min(a * b, advice + bits * probes)
    upper = ruling_upper(a, b, dimension)
    capacity = cells**probes * upper
    ratio = Fraction(capacity, required)
    return {
        "a": a, "b": b, "source_bits": a * b,
        "cells": cells, "cell_bits": bits,
        "free_common_advice_bits": advice,
        "probes": probes, "rank_bound": dimension,
        "rank_one_queries": str(required),
        "elementary_ruling_upper": str(upper),
        "address_fiber_capacity": str(capacity),
        "capacity_ratio_exact": str(ratio),
        "capacity_ratio_approximate": float(ratio),
        "rejected_exactly": capacity < required,
    }


def explicit_nonlinear_witness() -> dict[str, object]:
    """Validate a specified scope witness, not search for a new encoding."""
    images = set()
    checks = 0
    for source in range(8):
        x, y, z = [(source >> i) & 1 for i in range(3)]
        e = x | (y << 1) | ((z ^ (x & y)) << 2)
        images.add(e)
        dx, dy = e & 1, (e >> 1) & 1
        dz = ((e >> 2) & 1) ^ (dx & dy)
        assert (dx, dy, dz) == (x, y, z)
        for query in range(8):
            u = [(query >> i) & 1 for i in range(3)]
            reference_parity = (source & query).bit_count() % 2
            decoded_parity = ((dx & u[0]) ^ (dy & u[1]) ^ (dz & u[2]))
            assert reference_parity == decoded_parity
            reference_integer = sum((1 + bit) * inp
                                    for bit, inp in zip((x, y, z), u))
            decoded_integer = sum((1 + bit) * inp
                                  for bit, inp in zip((dx, dy, dz), u))
            assert reference_integer == decoded_integer <= 6
            checks += 1
    assert len(images) == 8
    # An affine map fixing 0 obeys E(e1 XOR e2)=E(e1) XOR E(e2).
    def enc(s: int) -> int:
        x, y, z = [(s >> i) & 1 for i in range(3)]
        return x | (y << 1) | ((z ^ (x & y)) << 2)
    assert enc(0) == 0 and enc(1 ^ 2) != enc(1) ^ enc(2)
    return {"source_states": 8, "queries_per_source": 8,
            "checks": checks, "bijective": True,
            "nonlinear": True, "native_integer_maximum": 6,
            "native_kernel_executed": False}


def validate_docs(root: Path) -> dict[str, object]:
    links = []
    for filename in ("PLAN.md", "REPORT.md"):
        text = (root / filename).read_text(encoding="utf-8")
        assert text.count("```") % 2 == 0
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("http:", "https:")):
                continue
            destination = (root / target.split("#", 1)[0]).resolve()
            assert destination.exists(), (filename, target)
            links.append({"source": filename, "target": target})
    return {"utf8_and_fences_pass": True, "relative_links_pass": True,
            "checked_relative_links": links}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cases = [case(25, 108, 50, 64, t) for t in range(6)]
    assert [c["rejected_exactly"] for c in cases] == [True] * 5 + [False]
    assert cases[4]["elementary_ruling_upper"] == "973555815717932701149233290412033"
    assert cases[4]["address_fiber_capacity"] == "6084723848237079382182708065075206250000"
    assert cases[4]["rank_one_queries"] == "10889035416951477172401260654660528635905"
    original_accessible = [case(25, 108, 725, 64, t) for t in (2, 3)]
    assert [c["rejected_exactly"] for c in original_accessible] == [True, False]
    common_advice_boundary = [case(25, 108, 50, 64, 2, advice=h)
                              for h in (1383, 1384)]
    assert [c["rejected_exactly"] for c in common_advice_boundary] == [True, False]
    hot_full_source = case(25, 108, 50, 64, 0, advice=2700)
    assert not hot_full_source["rejected_exactly"]
    historical_traffic = Fraction(5 * 64, 4 * 25 * 108)
    assert historical_traffic == Fraction(4, 135)
    assert historical_traffic / Fraction(8, 675) == Fraction(5, 2)
    result = {
        "classification": "SCOPED_NONLINEAR_CELL_INTERFACE_REJECTED_THROUGH_FOUR_READS",
        "model": "exact deterministic full Cartesian binary source/query; uniform decoder",
        "five_read_construction": "NOT_ESTABLISHED",
        "target_status": "NOT_TESTED",
        "full_mission_O1_O6": "OPEN",
        "cases": cases,
        "accessible_original_bf16_cases": original_accessible,
        "full_source_hot_advice_scope_control": hot_full_source,
        "two_read_common_advice_boundary": common_advice_boundary,
        "historical_four_lane_traffic_lower_bound": str(historical_traffic),
        "ratio_to_historical_8_over_675": "5/2",
        "specified_nonlinear_witness": explicit_nonlinear_witness(),
        "proof_validation": "manual independent mathematical reviews; not formal verification",
    }
    (root / "certificate.json").write_text(json.dumps(result, indent=2) + "\n")
    # Pre-create the declared local validation target so its doc link is checkable.
    (root / "validation.json").write_text("{}\n")
    validation = validate_docs(root)
    validation.update({"exact_integer_certificate_pass": True,
                       "specific_witness_pass": True,
                       "runtime_tests": "NOT_RUN",
                       "full_repository_suite": "NOT_RUN",
                       "target_hardware": "NOT_TESTED"})
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    files = ["PLAN.md", "REPORT.md", "verify.py", "certificate.json", "validation.json"]
    lines = [hashlib.sha256((root / name).read_bytes()).hexdigest() + "  " + name
             for name in files]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n")
    print(json.dumps({"checks": "PASS", "rejected_maximum_reads": 4,
                      "first_not_excluded_reads": 5,
                      "construction_at_five_reads": False,
                      "model_or_hardware_run": False}, indent=2))


if __name__ == "__main__":
    main()
